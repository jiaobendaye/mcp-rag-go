package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

// retrievalState holds shared state for the retrieval graph.
type retrievalState struct {
	Query string
}

// BuildRetrievalGraph creates an eino Graph that chains retrieval + generation.
// The compiled Runnable takes a query string and returns an answer string.
//
// Graph topology:
//
//	string (query)
//	  ↓
//	[parseInput]         Lambda: write state.Query → passthrough string
//	  ↓
//	[retrieve]           ESRetriever: string → []*schema.Document
//	  ↓
//	[assemblePrompt]     Lambda: docs + state.Query → map[string]any
//	  ↓
//	[chatTemplate]       ChatTemplate: map[string]any → []*schema.Message
//	  ↓
//	[chatModel]          ChatModel: []*schema.Message → *schema.Message
//	  ↓
//	[formatOutput]       Lambda: *schema.Message → string (content)
func BuildRetrievalGraph(ctx context.Context, embedder Embedder, searcher Searcher, llm LLMGenerator, searchMode string) (compose.Runnable[string, string], error) {
	// Build retriever
	retriever := NewESRetriever(embedder, searcher, "", searchMode, 5, 0.7)

	// Build ChatTemplate
	chatTemplate := prompt.FromMessages(schema.FString,
		schema.SystemMessage("你是一个知识库助手，基于给定的参考资料回答问题。如果知识库内容不足以回答问题，请说明无法找到相关信息。"),
		schema.UserMessage("参考资料:\n{context}\n\n问题: {query}\n\n请提供准确、简洁的回答:"),
	)

	// Create graph with local state management
	graph := compose.NewGraph[string, string](
		compose.WithGenLocalState(func(ctx context.Context) *retrievalState {
			return &retrievalState{}
		}),
	)

	// Node 1: parseInput — pass through string and store query in state
	err := graph.AddLambdaNode("parse_input", compose.InvokableLambda(
		func(ctx context.Context, in string) (string, error) {
			return in, nil
		},
	), compose.WithStatePostHandler(func(ctx context.Context, out string, state *retrievalState) (string, error) {
		state.Query = out
		return out, nil
	}))
	if err != nil {
		return nil, fmt.Errorf("add parse_input node: %w", err)
	}

	// Node 2: retrieve — ESRetriever: string → []*schema.Document
	err = graph.AddRetrieverNode("retrieve", retriever)
	if err != nil {
		return nil, fmt.Errorf("add retrieve node: %w", err)
	}

	// Node 3: assemblePrompt — docs + state.Query → map[string]any{"query", "context"}
	err = graph.AddLambdaNode("assemble_prompt", compose.InvokableLambda(
		func(ctx context.Context, docs []*schema.Document) (map[string]any, error) {
			// Read query from state
			var query string
			err := compose.ProcessState[*retrievalState](ctx, func(ctx context.Context, s *retrievalState) error {
				query = s.Query
				return nil
			})
			if err != nil {
				return nil, fmt.Errorf("read state: %w", err)
			}

			// Build context string
			contextStr := ""
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
	))
	if err != nil {
		return nil, fmt.Errorf("add assemble_prompt node: %w", err)
	}

	// Node 4: chatTemplate — ChatTemplate: map[string]any → []*schema.Message
	err = graph.AddChatTemplateNode("chat_template", chatTemplate)
	if err != nil {
		return nil, fmt.Errorf("add chat_template node: %w", err)
	}

	// Node 5: chatModel — ChatModel: []*schema.Message → *schema.Message
	err = graph.AddChatModelNode("chat_model", llm)
	if err != nil {
		return nil, fmt.Errorf("add chat_model node: %w", err)
	}

	// Node 6: formatOutput — *schema.Message → string
	err = graph.AddLambdaNode("format_output", compose.InvokableLambda(
		func(ctx context.Context, msg *schema.Message) (string, error) {
			return msg.Content, nil
		},
	))
	if err != nil {
		return nil, fmt.Errorf("add format_output node: %w", err)
	}

	// Add edges: START → parse_input → retrieve → assemble_prompt → chat_template → chat_model → format_output → END
	err = graph.AddEdge(compose.START, "parse_input")
	if err != nil {
		return nil, fmt.Errorf("add edge START->parse_input: %w", err)
	}
	err = graph.AddEdge("parse_input", "retrieve")
	if err != nil {
		return nil, fmt.Errorf("add edge parse_input->retrieve: %w", err)
	}
	err = graph.AddEdge("retrieve", "assemble_prompt")
	if err != nil {
		return nil, fmt.Errorf("add edge retrieve->assemble_prompt: %w", err)
	}
	err = graph.AddEdge("assemble_prompt", "chat_template")
	if err != nil {
		return nil, fmt.Errorf("add edge assemble_prompt->chat_template: %w", err)
	}
	err = graph.AddEdge("chat_template", "chat_model")
	if err != nil {
		return nil, fmt.Errorf("add edge chat_template->chat_model: %w", err)
	}
	err = graph.AddEdge("chat_model", "format_output")
	if err != nil {
		return nil, fmt.Errorf("add edge chat_model->format_output: %w", err)
	}
	err = graph.AddEdge("format_output", compose.END)
	if err != nil {
		return nil, fmt.Errorf("add edge format_output->END: %w", err)
	}

	// Compile
	runnable, err := graph.Compile(ctx)
	if err != nil {
		return nil, fmt.Errorf("compile graph: %w", err)
	}

	return runnable, nil
}
