package rag

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"

	elastic_retriever "github.com/cloudwego/eino-ext/components/retriever/es8"
	elastic_search_mode "github.com/cloudwego/eino-ext/components/retriever/es8/search_mode"
	elasticsearch "github.com/elastic/go-elasticsearch/v8"
)

// BuildRetrievalGraph returns a compose.Runnable that retrieves from
// one or more KB ES indices and produces an LLM answer. The graph is
// compiled per-request so that per-KB retrievers (bound to actual index
// names) and per-request parameters (TopK, MinScore, SearchMode) are
// baked in directly — no placeholder, no KBParams, no WithIndex routing.
//
// Graph topology:
//
//	                     START (string: query)
//	                    /                      \
//	[retrieve_pipeline] /                        \ [format_query]
//	Chain:              |                        | Lambda:
//	embed+build→        |                        | string →
//	retriever(s)→       |                        | map[string]any
//	format_context      |                        | {"query": "..."}
//	    ↓               |                        ↓
//	map[string]any      |           map[string]any
//	{"context":"..."}   |           {"query":"..."}
//	                    \                      /
//	                     \                    /
//	               [chat_template] (AllPredecessor)
//	               merged: {"query":..., "context":...}
//	                       ↓
//	                 [chat_model]
//	                       ↓
//	                 [format_output]
//	                       ↓
//	                     END (string)
func BuildRetrievalGraph(
	ctx context.Context,
	esClient *elasticsearch.Client,
	llm LLMGenerator,
	embedder embedding.Embedder,
	indexNames []string,
	topK int,
	minScore float64,
	searchMode string,
) (compose.Runnable[string, string], error) {
	if esClient == nil {
		return nil, fmt.Errorf("BuildRetrievalGraph: nil esClient")
	}
	if llm == nil {
		return nil, fmt.Errorf("BuildRetrievalGraph: nil llm")
	}
	if embedder == nil {
		return nil, fmt.Errorf("BuildRetrievalGraph: nil embedder (needed to build ES query JSON)")
	}
	if len(indexNames) == 0 {
		return nil, fmt.Errorf("BuildRetrievalGraph: empty indexNames")
	}

	chatTemplate := prompt.FromMessages(schema.FString,
		schema.SystemMessage("你是一个知识库助手，基于给定的参考资料回答问题。如果知识库内容不足以回答问题，请说明无法找到相关信息。"),
		schema.UserMessage("参考资料:\n{context}\n\n问题: {query}\n\n请提供准确、简洁的回答:"),
	)

	// ── Sub-chain: retrieve pipeline (string → map[string]any{"context":...}) ──
	retrieveChain := compose.NewChain[string, map[string]any]()

	// Node: embed query + build ES query JSON body.
	// Closure-captures: embedder, topK, minScore, searchMode.
	retrieveChain.AppendLambda(compose.InvokableLambda(
		func(ctx context.Context, query string) (string, error) {
			vectors, err := embedder.EmbedStrings(ctx, []string{query})
			if err != nil {
				return "", fmt.Errorf("embed query: %w", err)
			}
			if len(vectors) == 0 || len(vectors[0]) == 0 {
				return "", fmt.Errorf("embedder returned empty vector")
			}
			body, err := BuildHybridQueryJSON(query, vectors[0], topK, minScore, DefaultWeights, searchMode)
			if err != nil {
				return "", fmt.Errorf("build query body: %w", err)
			}
			return body, nil
		},
	))

	// Node: retrieve — one elastic_retriever per index. Single index uses
	// direct retriever; multi-index uses Parallel fan-out + merge.
	if len(indexNames) == 1 {
		scoreThreshold := minScore
		retriever, err := elastic_retriever.NewRetriever(ctx, &elastic_retriever.RetrieverConfig{
			Client:         esClient,
			Index:          indexNames[0],
			TopK:           topK,
			ScoreThreshold: &scoreThreshold,
			SearchMode:     elastic_search_mode.SearchModeRawStringRequest(),
			ResultParser:   ProjectResultParser(),
		})
		if err != nil {
			return nil, fmt.Errorf("BuildRetrievalGraph: create retriever for %q: %w", indexNames[0], err)
		}
		retrieveChain.AppendRetriever(retriever)
	} else {
		parallel := compose.NewParallel()
		for _, idx := range indexNames {
			scoreThreshold := minScore
			retriever, err := elastic_retriever.NewRetriever(ctx, &elastic_retriever.RetrieverConfig{
				Client:         esClient,
				Index:          idx,
				TopK:           topK,
				ScoreThreshold: &scoreThreshold,
				SearchMode:     elastic_search_mode.SearchModeRawStringRequest(),
				ResultParser:   ProjectResultParser(),
			})
			if err != nil {
				return nil, fmt.Errorf("BuildRetrievalGraph: create retriever for %q: %w", idx, err)
			}
			parallel.AddRetriever(idx, retriever)
		}
		retrieveChain.AppendParallel(parallel)

		// Merge: flatten parallel output, sort by score desc, truncate to topK.
		retrieveChain.AppendLambda(compose.InvokableLambda(
			func(ctx context.Context, docsMap map[string]any) ([]*schema.Document, error) {
				return mergeDocsMap(docsMap, topK), nil
			},
		))
	}

	// Node: format context → map[string]any{"context": "..."}
	retrieveChain.AppendLambda(compose.InvokableLambda(
		func(ctx context.Context, docs []*schema.Document) (map[string]any, error) {
			return map[string]any{"context": formatDocsContext(docs)}, nil
		},
	))

	// ── Main graph ──
	graph := compose.NewGraph[string, string]()

	_ = graph.AddGraphNode("retrieve_pipeline", retrieveChain, compose.WithNodeName("RetrievePipeline"))
	_ = graph.AddLambdaNode("format_query", compose.InvokableLambda(
		func(ctx context.Context, query string) (map[string]any, error) {
			return map[string]any{"query": query}, nil
		},
	), compose.WithNodeName("FormatQuery"))
	_ = graph.AddChatTemplateNode("chat_template", chatTemplate)
	_ = graph.AddChatModelNode("chat_model", llm)
	_ = graph.AddLambdaNode("format_output", compose.InvokableLambda(
		func(ctx context.Context, msg *schema.Message) (string, error) {
			return msg.Content, nil
		},
	), compose.WithNodeName("FormatOutput"))

	// Edges: fan-out from START, fan-in to chat_template (AllPredecessor).
	_ = graph.AddEdge(compose.START, "retrieve_pipeline")
	_ = graph.AddEdge(compose.START, "format_query")
	_ = graph.AddEdge("retrieve_pipeline", "chat_template")
	_ = graph.AddEdge("format_query", "chat_template")
	_ = graph.AddEdge("chat_template", "chat_model")
	_ = graph.AddEdge("chat_model", "format_output")
	_ = graph.AddEdge("format_output", compose.END)

	runnable, err := graph.Compile(ctx,
		compose.WithNodeTriggerMode(compose.AllPredecessor),
		compose.WithGraphName("RetrievalGraph"),
	)
	if err != nil {
		return nil, fmt.Errorf("BuildRetrievalGraph: compile graph: %w", err)
	}
	return runnable, nil
}

// formatDocsContext builds the context string from retrieved documents.
func formatDocsContext(docs []*schema.Document) string {
	if len(docs) == 0 {
		return "(无相关资料)"
	}
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
	return sb.String()
}

// mergeDocsMap flattens the per-index document slices from a Parallel
// output map, sorts by score descending, and truncates to topK.
func mergeDocsMap(docsMap map[string]any, topK int) []*schema.Document {
	var all []*schema.Document
	for _, v := range docsMap {
		docs, ok := v.([]*schema.Document)
		if !ok {
			continue
		}
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
