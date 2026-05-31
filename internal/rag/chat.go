package rag

import (
	"context"
	"fmt"
	"strings"
)

// LLMGenerator is the interface for text generation models.
type LLMGenerator interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

// ChatRequest contains the input for a chat request.
type ChatRequest struct {
	Query string `json:"query"`
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
}

// NewChatService creates a new ChatService.
func NewChatService(searcher Searcher, embedder Embedder, llm LLMGenerator) *ChatService {
	return &ChatService{
		searcher: searcher,
		embedder: embedder,
		llm:      llm,
	}
}

// Chat performs RAG-based conversation: retrieve → build prompt → generate.
func (c *ChatService) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	if strings.TrimSpace(req.Query) == "" {
		return nil, fmt.Errorf("query is required")
	}

	// 1. Embed the query
	vecs, err := c.embedder.EmbedStrings(ctx, []string{req.Query})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	// 2. Search for relevant documents
	hits, err := c.searcher.Search(ctx, toFloat32(vecs[0]), 5, 0.7)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	// 3. Build response
	if len(hits) == 0 {
		return &ChatResponse{
			Query:      req.Query,
			Collection: "kb_1",
			Response:   fmt.Sprintf("未找到与 \"%s\" 相关的信息。", req.Query),
			Sources:    []SearchHit{},
		}, nil
	}

	// 4. Build prompt with retrieved context
	prompt := buildChatPrompt(req.Query, hits)

	// 5. Generate with LLM
	answer, err := c.llm.Generate(ctx, prompt)
	if err != nil {
		// Graceful degradation: return context with error note
		answer = formatFallbackResponse(hits, err)
	}

	return &ChatResponse{
		Query:      req.Query,
		Collection: "kb_1",
		Response:   answer,
		Sources:    hits,
	}, nil
}

// buildChatPrompt constructs a Chinese prompt with retrieved context.
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

// formatFallbackResponse builds a response when LLM is unavailable.
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
