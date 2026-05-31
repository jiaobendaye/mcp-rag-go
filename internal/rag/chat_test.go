package rag

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// mockLLM implements LLMGenerator for testing.
type mockLLM struct {
	generateFunc func(ctx context.Context, prompt string) (string, error)
}

func (m *mockLLM) Generate(ctx context.Context, prompt string) (string, error) {
	if m.generateFunc != nil {
		return m.generateFunc(ctx, prompt)
	}
	return "这是基于知识的回答。", nil
}

func TestChat(t *testing.T) {
	t.Run("successful chat", func(t *testing.T) {
		emb := &mockEmbedder{}
		searcher := &mockSearcher{
			searchFunc: func(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return []SearchHit{
					{ChunkID: "c1", Content: "RAG是检索增强生成技术", Score: 0.95, Filename: "doc1.md"},
				}, nil
			},
		}
		llm := &mockLLM{}

		svc := NewChatService(searcher, emb, llm)
		resp, err := svc.Chat(context.Background(), &ChatRequest{Query: "什么是RAG"})
		if err != nil {
			t.Fatalf("Chat error: %v", err)
		}

		if resp.Query != "什么是RAG" {
			t.Errorf("expected query in response")
		}
		if resp.Response == "" {
			t.Error("expected non-empty response")
		}
		if len(resp.Sources) != 1 {
			t.Errorf("expected 1 source, got %d", len(resp.Sources))
		}
	})

	t.Run("empty query", func(t *testing.T) {
		svc := NewChatService(&mockSearcher{}, &mockEmbedder{}, &mockLLM{})
		_, err := svc.Chat(context.Background(), &ChatRequest{Query: ""})
		if err == nil {
			t.Error("expected error for empty query")
		}
	})

	t.Run("no results found", func(t *testing.T) {
		emb := &mockEmbedder{}
		searcher := &mockSearcher{
			searchFunc: func(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return []SearchHit{}, nil
			},
		}
		llm := &mockLLM{}

		svc := NewChatService(searcher, emb, llm)
		resp, err := svc.Chat(context.Background(), &ChatRequest{Query: "unknown"})
		if err != nil {
			t.Fatalf("Chat error: %v", err)
		}
		if !strings.Contains(resp.Response, "未找到") {
			t.Errorf("expected 'not found' message, got %q", resp.Response)
		}
		if len(resp.Sources) != 0 {
			t.Errorf("expected 0 sources, got %d", len(resp.Sources))
		}
	})

	t.Run("LLM failure graceful degradation", func(t *testing.T) {
		emb := &mockEmbedder{}
		searcher := &mockSearcher{
			searchFunc: func(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return []SearchHit{
					{ChunkID: "c1", Content: "test content", Score: 0.9},
				}, nil
			},
		}
		llm := &mockLLM{
			generateFunc: func(ctx context.Context, prompt string) (string, error) {
				return "", errors.New("API unavailable")
			},
		}

		svc := NewChatService(searcher, emb, llm)
		resp, err := svc.Chat(context.Background(), &ChatRequest{Query: "test"})
		if err != nil {
			t.Fatalf("Chat should not error on LLM failure: %v", err)
		}
		if !strings.Contains(resp.Response, "LLM is not available") {
			t.Errorf("expected fallback message, got %q", resp.Response)
		}
		if !strings.Contains(resp.Response, "test content") {
			t.Errorf("expected context in fallback response")
		}
	})

	t.Run("search failure propagates", func(t *testing.T) {
		emb := &mockEmbedder{}
		searcher := &mockSearcher{
			searchFunc: func(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return nil, errors.New("connection refused")
			},
		}
		svc := NewChatService(searcher, emb, &mockLLM{})
		_, err := svc.Chat(context.Background(), &ChatRequest{Query: "test"})
		if err == nil {
			t.Error("expected search error to propagate")
		}
	})
}

func TestBuildChatPrompt(t *testing.T) {
	hits := []SearchHit{
		{Content: "RAG是一种技术", Score: 0.95},
		{Content: "它结合了检索和生成", Score: 0.85},
	}

	prompt := buildChatPrompt("什么是RAG", hits)

	if !strings.Contains(prompt, "什么是RAG") {
		t.Error("prompt should contain the query")
	}
	if !strings.Contains(prompt, "RAG是一种技术") {
		t.Error("prompt should contain hit content")
	}
	if !strings.Contains(prompt, "0.950") {
		t.Error("prompt should contain similarity scores")
	}
}

func TestFormatFallbackResponse(t *testing.T) {
	hits := []SearchHit{
		{Content: "content1", Score: 0.9},
	}
	err := errors.New("test error")

	resp := formatFallbackResponse(hits, err)
	if !strings.Contains(resp, "LLM is not available") {
		t.Error("should mention LLM unavailability")
	}
	if !strings.Contains(resp, "test error") {
		t.Error("should include error message")
	}
	if !strings.Contains(resp, "content1") {
		t.Error("should include retrieved context")
	}
}
