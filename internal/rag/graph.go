package rag

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

// retrievalState holds shared state for the retrieval graph.
type retrievalState struct {
	Query      string
	Vector     []float64
	IndexNames []string // per-request KB index names (single KB: []string{idx})
	TopK       int      // per-request top-K
	MinScore   float64  // per-request score threshold
	SearchMode string   // per-request search mode (hybrid/knn/rrf/keyword)
}

// BuildRetrievalGraph returns a compose.Runnable that retrieves from
// one or more KB ES indices and produces an LLM answer. The graph is
// compiled once at startup; per-request KB parameters (indexNames, topK,
// minScore, searchMode) are passed via context using WithKBParams, read in
// parse_input's StatePostHandler, and stored in retrievalState for
// downstream nodes (build_query, multi_retrieve) to consume.
//
// Graph topology:
//
//	string (query)
//	  ↓
//	[parse_input]         Lambda: write state.Query → passthrough string
//	                      (also embeds query via embedder, stores state.Vector,
//	                       reads KBParams from context → stores IndexNames/TopK/
//	                       MinScore/SearchMode in state)
//	  ↓
//	[build_query]         Lambda: read state.Vector/TopK/MinScore/SearchMode,
//	                      build ES JSON body via BuildHybridQueryJSON.
//	                      Output: JSON body string.
//	  ↓
//	[multi_retrieve]      Lambda: read IndexNames/TopK/MinScore from state,
//	                      fan-out across N ES indices via goroutines,
//	                      merge results by score desc, truncate to TopK.
//	                      Output: []*schema.Document
//	  ↓
//	[assemble_prompt]     Lambda: docs + state.Query → map[string]any
//	  ↓
//	[chat_template]       ChatTemplate: map[string]any → []*schema.Message
//	  ↓
//	[chat_model]          ChatModel: []*schema.Message → *schema.Message
//	  ↓
//	[format_output]       Lambda: *schema.Message → string (content)
func BuildRetrievalGraph(
	ctx context.Context,
	kbRetriever retriever.Retriever,
	llm LLMGenerator,
	embedder embedding.Embedder,
) (compose.Runnable[string, string], error) {
	if kbRetriever == nil {
		return nil, fmt.Errorf("BuildRetrievalGraph: nil kbRetriever")
	}
	if llm == nil {
		return nil, fmt.Errorf("BuildRetrievalGraph: nil llm")
	}
	if embedder == nil {
		return nil, fmt.Errorf("BuildRetrievalGraph: nil embedder (needed to build ES query JSON)")
	}

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
	// to produce dense vector, and read per-request KBParams from context
	// into state for downstream nodes.
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
		// Read per-request KB params from caller's context.
		if kb, ok := GetKBParams(ctx); ok {
			state.IndexNames = kb.IndexNames
			state.TopK = kb.TopK
			state.MinScore = kb.MinScore
			state.SearchMode = kb.SearchMode
		}
		return out, nil
	})); err != nil {
		return nil, fmt.Errorf("add parse_input node: %w", err)
	}

	// Node 2: build_query — read state.Vector, TopK, MinScore, SearchMode
	// and build ES query body JSON via BuildHybridQueryJSON. The output
	// is the JSON body that the eino-ext retriever (configured with
	// SearchModeRawStringRequest) expects as its query argument.
	if err := graph.AddLambdaNode("build_query", compose.InvokableLambda(
		func(ctx context.Context, in string) (string, error) {
			var (
				vector     []float64
				topK       int
				minScore   float64
				searchMode string
			)
			if err := compose.ProcessState[*retrievalState](ctx, func(ctx context.Context, s *retrievalState) error {
				vector = s.Vector
				topK = s.TopK
				minScore = s.MinScore
				searchMode = s.SearchMode
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

	// Node 3: multi_retrieve — reads IndexNames/TopK/MinScore from state,
	// fans out across N KB indices using goroutines, merges results by
	// score descending, and truncates to TopK.
	if err := graph.AddLambdaNode("multi_retrieve", compose.InvokableLambda(
		func(ctx context.Context, body string) ([]*schema.Document, error) {
			var indexNames []string
			var topK int
			var minScore float64
			if err := compose.ProcessState[*retrievalState](ctx, func(ctx context.Context, s *retrievalState) error {
				indexNames = s.IndexNames
				topK = s.TopK
				minScore = s.MinScore
				return nil
			}); err != nil {
				return nil, fmt.Errorf("read state: %w", err)
			}

			if len(indexNames) == 0 {
				return nil, fmt.Errorf("multi_retrieve: no index names in state")
			}

			// Fan-out across indices concurrently.
			results := make([][]*schema.Document, len(indexNames))
			var wg sync.WaitGroup
			for i, idx := range indexNames {
				wg.Add(1)
				go func(i int, idx string) {
					defer wg.Done()
					docs, err := kbRetriever.Retrieve(ctx, body,
						retriever.WithIndex(idx),
						retriever.WithTopK(topK),
						retriever.WithScoreThreshold(minScore),
					)
					if err != nil {
						// Log but don't fail — index-level errors are
						// non-fatal; downstream handles empty results.
						return
					}
					results[i] = docs
				}(i, idx)
			}
			wg.Wait()

			return mergeDocs(results, topK), nil
		},
	)); err != nil {
		return nil, fmt.Errorf("add multi_retrieve node: %w", err)
	}

	// Node 4: assemble_prompt — docs + state.Query → map[string]any{"query", "context"}
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

	// Node 5: chat_template
	if err := graph.AddChatTemplateNode("chat_template", chatTemplate); err != nil {
		return nil, fmt.Errorf("add chat_template node: %w", err)
	}

	// Node 6: chat_model
	if err := graph.AddChatModelNode("chat_model", llm); err != nil {
		return nil, fmt.Errorf("add chat_model node: %w", err)
	}

	// Node 7: format_output
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
		{"build_query", "multi_retrieve"},
		{"multi_retrieve", "assemble_prompt"},
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

// mergeDocs flattens per-index document slices, sorts by score descending,
// and truncates to topK. It is used by the multi_retrieve lambda to
// combine results from concurrent per-index retrieval calls.
func mergeDocs(kbResults [][]*schema.Document, topK int) []*schema.Document {
	var all []*schema.Document
	for _, docs := range kbResults {
		all = append(all, docs...)
	}
	sort.Slice(all, func(i, j int) bool {
		si, _ := all[i].MetaData["score"].(float64)
		sj, _ := all[j].MetaData["score"].(float64)
		return si > sj
	})
	if topK > 0 && len(all) > topK {
		all = all[:topK]
	}
	return all
}
