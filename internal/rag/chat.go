package rag

import (
	"fmt"
	"strings"
)

// ChatRequest contains the input for a chat request.
type ChatRequest struct {
	Query               string  `json:"query"`
	Collection          string  `json:"collection,omitempty"`
	KBID                *int64  `json:"kb_id,omitempty"`
	KBIDs               []int64 `json:"kb_ids,omitempty"`
	Limit               int     `json:"limit,omitempty"`
	UserID              *int64  `json:"user_id,omitempty"`
	AgentID             *int64  `json:"agent_id,omitempty"`
	Scope               string  `json:"scope,omitempty"`
	APIKey              string  `json:"api_key,omitempty"`
	PreRetrievedContext string  `json:"-"` // pre-built context for multi-KB mode
}

// ChatResponse contains the result of a chat request (compatible with Python MCP-RAG format).
type ChatResponse struct {
	Query      string      `json:"query"`
	Collection string      `json:"collection"`
	Response   string      `json:"response"`
	Sources    []SearchHit `json:"sources"`
}

// SearchRequest contains the input for a search request.
type SearchRequest struct {
	Query    string  `json:"query"`
	Limit    int     `json:"limit,omitempty"`
	MinScore float64 `json:"min_score,omitempty"`
}

// SearchResponse contains the result of a search request (compatible with Python MCP-RAG format).
type SearchResponse struct {
	Query      string      `json:"query"`
	Collection string      `json:"collection"`
	Results    []SearchHit `json:"results"`
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

func buildChatPromptFromContext(query, context string) string {
	return fmt.Sprintf(
		"基于以下知识库内容回答用户的问题。如果知识库内容不足以回答问题，请说明无法找到相关信息。\n\n"+
			"知识库内容:\n%s\n\n"+
			"用户问题: %s\n\n"+
			"请提供准确、简洁的回答:",
		context, query,
	)
}
