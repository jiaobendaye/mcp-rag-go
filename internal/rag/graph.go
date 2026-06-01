package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

// retrievalState holds shared state for the retrieval graph.
type retrievalState struct {
	Query  string
	Vector []float64
}

// BuildRetrievalGraphAt returns a compose.Runnable that retrieves from
// the given ES index and produces an LLM answer. Per-request compile
// with closure capture of indexName, topK, and minScore via
// retriever.WithIndex / WithTopK / WithScoreThreshold.
//
// Graph topology:
//
//	string (query)
//	  ↓
//	[parse_input]         Lambda: write state.Query → passthrough string
//	                      (also embeds query via embedder, stores in state.Vector)
//	  ↓
//	[build_query]         Lambda: read state.Vector, build ES JSON body via
//	                      BuildHybridQueryJSON. Output: JSON body string.
//	  ↓
//	[retrieve]            retriever: string → []*schema.Document
//	                      (closes over per-request indexName)
//	  ↓
//	[assemble_prompt]     Lambda: docs + state.Query → map[string]any
//	  ↓
//	[chat_template]       ChatTemplate: map[string]any → []*schema.Message
//	  ↓
//	[chat_model]          ChatModel: []*schema.Message → *schema.Message
//	  ↓
//	[format_output]       Lambda: *schema.Message → string (content)
//
// The embedder is required because the eino-ext retriever is configured
// with SearchModeRawStringRequest, which expects the query arg to be a
// complete ES query body JSON (NOT plain text). We must build that body
// ourselves using the embedded query vector and BuildHybridQueryJSON.
func BuildRetrievalGraphAt(
	ctx context.Context,
	kbRetriever retriever.Retriever,
	llm LLMGenerator,
	embedder embedding.Embedder,
	indexName string,
	searchMode string,
	topK int,
	minScore float64,
) (compose.Runnable[string, string], error) {
	if indexName == "" {
		return nil, fmt.Errorf("BuildRetrievalGraphAt: empty indexName")
	}
	if kbRetriever == nil {
		return nil, fmt.Errorf("BuildRetrievalGraphAt: nil kbRetriever")
	}
	if llm == nil {
		return nil, fmt.Errorf("BuildRetrievalGraphAt: nil llm")
	}
	if embedder == nil {
		return nil, fmt.Errorf("BuildRetrievalGraphAt: nil embedder (needed to build ES query JSON)")
	}

	// Per-request retriever options: close over indexName + the per-request
	// topK/minScore. The adapter ignores caller-supplied options and uses
	// these instead, so the retriever always queries the resolved KB.
	retrieveOpts := []retriever.Option{
		retriever.WithIndex(indexName),
		retriever.WithTopK(topK),
		retriever.WithScoreThreshold(minScore),
	}
	frozenRetriever := newRetrieverWithFixedIndex(kbRetriever, retrieveOpts)

	chatTemplate := prompt.FromMessages(schema.FString,
		schema.SystemMessage("你是一个知识库助手，基于给定的参考资料回答问题。如果知识库内容不足以回答问题，请说明无法找到相关信息。"),
		schema.UserMessage("参考资料:\n{context}\n\n问题: {query}\n\n请提供准确、简洁的回答:"),
	)

	graph := compose.NewGraph[string, string](
		compose.WithGenLocalState(func(ctx context.Context) *retrievalState {
			return &retrievalState{}
		}),
	)

	// Node 1: parse_input — passthrough, record query in state, embed query
	// to produce dense vector (stored in state for build_query to use).
	if err := graph.AddLambdaNode("parse_input", compose.InvokableLambda(
		func(ctx context.Context, in string) (string, error) {
			return in, nil
		},
	), compose.WithStatePostHandler(func(ctx context.Context, out string, state *retrievalState) (string, error) {
		state.Query = out
		// Embed the query text for the upcoming build_query step.
		vectors, err := embedder.EmbedStrings(ctx, []string{out})
		if err != nil {
			return "", fmt.Errorf("embed query: %w", err)
		}
		if len(vectors) == 0 || len(vectors[0]) == 0 {
			return "", fmt.Errorf("embedder returned empty vector")
		}
		state.Vector = vectors[0]
		return out, nil
	})); err != nil {
		return nil, fmt.Errorf("add parse_input node: %w", err)
	}

	// Node 2: build_query — read state.Vector, build ES query body JSON
	// via BuildHybridQueryJSON. The output is the JSON body that the
	// eino-ext retriever (configured with SearchModeRawStringRequest)
	// expects as its query argument.
	if err := graph.AddLambdaNode("build_query", compose.InvokableLambda(
		func(ctx context.Context, in string) (string, error) {
			var vector []float64
			if err := compose.ProcessState[*retrievalState](ctx, func(ctx context.Context, s *retrievalState) error {
				vector = s.Vector
				return nil
			}); err != nil {
				return "", fmt.Errorf("read state: %w", err)
			}
			if len(vector) == 0 {
				return "", fmt.Errorf("build_query: empty vector in state (parse_input did not embed)")
			}
			body, err := BuildHybridQueryJSON(in, vector, topK, minScore, DefaultWeights, searchMode)
			if err != nil {
				return "", fmt.Errorf("build query body: %w", err)
			}
			return body, nil
		},
	)); err != nil {
		return nil, fmt.Errorf("add build_query node: %w", err)
	}

	// Node 3: retrieve — uses the per-request retriever that closes over
	// indexName, topK, minScore. Receives the JSON body from build_query.
	if err := graph.AddRetrieverNode("retrieve", frozenRetriever); err != nil {
		return nil, fmt.Errorf("add retrieve node: %w", err)
	}

	// Node 3: assemble_prompt — docs + state.Query → map[string]any{"query", "context"}
	if err := graph.AddLambdaNode("assemble_prompt", compose.InvokableLambda(
		func(ctx context.Context, docs []*schema.Document) (map[string]any, error) {
			var query string
			if err := compose.ProcessState[*retrievalState](ctx, func(ctx context.Context, s *retrievalState) error {
				query = s.Query
				return nil
			}); err != nil {
				return nil, fmt.Errorf("read state: %w", err)
			}

			var contextStr string
			if len(docs) == 0 {
				contextStr = "(无相关资料)"
			} else {
				var sb strings.Builder
				for i, doc := range docs {
					sb.WriteString(fmt.Sprintf("文档%d", i+1))
					if score, ok := doc.MetaData["score"]; ok {
						sb.WriteString(fmt.Sprintf(" (相关度: %.3f)", score))
					}
					sb.WriteString(":\n")
					sb.WriteString(doc.Content)
					sb.WriteString("\n\n")
				}
				contextStr = sb.String()
			}

			return map[string]any{
				"query":   query,
				"context": contextStr,
			}, nil
		},
	)); err != nil {
		return nil, fmt.Errorf("add assemble_prompt node: %w", err)
	}

	// Node 4: chat_template
	if err := graph.AddChatTemplateNode("chat_template", chatTemplate); err != nil {
		return nil, fmt.Errorf("add chat_template node: %w", err)
	}

	// Node 5: chat_model
	if err := graph.AddChatModelNode("chat_model", llm); err != nil {
		return nil, fmt.Errorf("add chat_model node: %w", err)
	}

	// Node 6: format_output
	if err := graph.AddLambdaNode("format_output", compose.InvokableLambda(
		func(ctx context.Context, msg *schema.Message) (string, error) {
			return msg.Content, nil
		},
	)); err != nil {
		return nil, fmt.Errorf("add format_output node: %w", err)
	}

	// Edges
	edges := []struct {
		from, to string
	}{
		{compose.START, "parse_input"},
		{"parse_input", "build_query"},
		{"build_query", "retrieve"},
		{"retrieve", "assemble_prompt"},
		{"assemble_prompt", "chat_template"},
		{"chat_template", "chat_model"},
		{"chat_model", "format_output"},
		{"format_output", compose.END},
	}
	for _, e := range edges {
		if err := graph.AddEdge(e.from, e.to); err != nil {
			return nil, fmt.Errorf("add edge %s->%s: %w", e.from, e.to, err)
		}
	}

	runnable, err := graph.Compile(ctx)
	if err != nil {
		return nil, fmt.Errorf("compile graph: %w", err)
	}
	return runnable, nil
}
