package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

// ChatRequest contains the input for a chat request.
type ChatRequest struct {
	Query      string  `json:"query"`
	Collection string  `json:"collection,omitempty"`
	KBID       *int64  `json:"kb_id,omitempty"`
	KBIDs      []int64 `json:"kb_ids,omitempty"`
	Limit      int     `json:"limit,omitempty"`
	UserID     *int64  `json:"user_id,omitempty"`
	AgentID    *int64  `json:"agent_id,omitempty"`
	Scope      string  `json:"scope,omitempty"`
	APIKey     string  `json:"api_key,omitempty"`
}

// ChatResponse contains the result of a chat request (compatible with Python MCP-RAG format).
type ChatResponse struct {
	Query      string       `json:"query"`
	Collection string       `json:"collection"`
	Response   string       `json:"response"`
	Sources    []SearchHit  `json:"sources"`
}

// SearchRequest contains the input for a search request.
type SearchRequest struct {
	Query    string `json:"query"`
	Limit    int    `json:"limit,omitempty"`
	MinScore float64 `json:"min_score,omitempty"`
}

// SearchResponse contains the result of a search request (compatible with Python MCP-RAG format).
type SearchResponse struct {
	Query      string       `json:"query"`
	Collection string       `json:"collection"`
	Results    []SearchHit  `json:"results"`
}

// ChatService provides RAG-based chat functionality.
type ChatService struct {
	searcher Searcher
	embedder Embedder
	llm      LLMGenerator
	graph    compose.Runnable[string, string] // optional: Eino Graph mode
}

// NewChatService creates a new ChatService.
// graph is optional; if nil, the service falls back to raw mode.
func NewChatService(searcher Searcher, embedder Embedder, llm LLMGenerator, graph compose.Runnable[string, string]) *ChatService {
	return &ChatService{searcher: searcher, embedder: embedder, llm: llm, graph: graph}
}

// Chat performs RAG-based conversation: retrieve → build prompt → generate.
func (c *ChatService) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	if strings.TrimSpace(req.Query) == "" {
		return nil, fmt.Errorf("query is required")
	}

	// Use graph mode if available
	if c.graph != nil {
		return c.chatViaGraph(ctx, req)
	}

	// Fallback: raw mode (original hand-written logic)
	return c.chatViaRaw(ctx, req)
}

// chatViaGraph uses the Eino Graph for retrieval + generation.
func (c *ChatService) chatViaGraph(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	answer, err := c.graph.Invoke(ctx, req.Query)
	if err != nil {
		return nil, fmt.Errorf("graph invoke: %w", err)
	}

	collection := req.Collection
	if collection == "" {
		collection = "kb_1"
	}

	return &ChatResponse{
		Query:      req.Query,
		Collection: collection,
		Response:   answer,
		Sources:    nil,
	}, nil
}

// chatViaRaw uses the original hand-written retrieval pipeline.
func (c *ChatService) chatViaRaw(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {

	vecs, err := c.embedder.EmbedStrings(ctx, []string{req.Query})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	topK := 5
	if req.Limit > 0 {
		topK = req.Limit
	}

	hits, err := c.searcher.SearchHybrid(ctx, req.Query, toFloat32(vecs[0]), topK, 0.7)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	collection := req.Collection
	if collection == "" {
		collection = "kb_1"
	}

	if len(hits) == 0 {
		return &ChatResponse{
			Query: req.Query, Collection: collection,
			Response: fmt.Sprintf("未找到与 \"%s\" 相关的信息。", req.Query),
			Sources:  []SearchHit{},
		}, nil
	}

	prompt := buildChatPrompt(req.Query, hits)

	// Call eino ChatModel
	msg, err := c.llm.Generate(ctx, []*schema.Message{
		{Role: schema.User, Content: prompt},
	})
	if err != nil {
		answer := formatFallbackResponse(hits, err)
		return &ChatResponse{Query: req.Query, Collection: collection, Response: answer, Sources: hits}, nil
	}

	return &ChatResponse{
		Query: req.Query, Collection: collection,
		Response: msg.Content,
		Sources:  hits,
	}, nil
}

func buildChatPrompt(query string, hits []SearchHit) string {
	var sb strings.Builder
	sb.WriteString("基于以下知识库内容回答用户的问题。如果知识库内容不足以回答问题，请说明无法找到相关信息。\n\n")
	sb.WriteString("知识库内容:\n")
	for i, h := range hits {
		sb.WriteString(fmt.Sprintf("文档%d (相似度: %.3f):\n%s\n\n", i+1, h.Score, h.Content))
	}
	sb.WriteString(fmt.Sprintf("用户问题: %s\n\n", query))
	sb.WriteString("请提供准确、简洁的回答:")
	return sb.String()
}

func formatFallbackResponse(hits []SearchHit, err error) string {
	var sb strings.Builder
	sb.WriteString("### Retrieved Context\n\n")
	for i, h := range hits {
		sb.WriteString(fmt.Sprintf("文档%d (相似度: %.3f):\n%s\n\n", i+1, h.Score, h.Content))
	}
	sb.WriteString("### Note\n")
	sb.WriteString("LLM is not available. The above context was retrieved for your query.\n\n")
	sb.WriteString(fmt.Sprintf("LLM error: %s", err.Error()))
	return sb.String()
}

func toFloat32(f64 []float64) []float32 {
	result := make([]float32, len(f64))
	for i, v := range f64 {
		result[i] = float32(v)
	}
	return result
}
