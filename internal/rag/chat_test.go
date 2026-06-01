package rag

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

// mockLLM implements model.BaseChatModel for testing.
type mockLLM struct {
	generateFunc func(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.Message, error)
}

func (m *mockLLM) Generate(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.Message, error) {
	if m.generateFunc != nil {
		return m.generateFunc(ctx, input, opts...)
	}
	return &schema.Message{Role: schema.Assistant, Content: "这是基于知识的回答。"}, nil
}

func (m *mockLLM) Stream(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	return nil, errors.New("not implemented")
}

func TestChat(t *testing.T) {
	t.Run("successful chat", func(t *testing.T) {
		emb := &mockEmbedder{}
		searcher := &mockSearcher{
			searchHybridFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return []SearchHit{{ChunkID: "c1", Content: "RAG是检索增强生成技术", Score: 0.95, Filename: "doc1.md"}}, nil
			},
		}
		llm := &mockLLM{}

		svc := NewChatService(searcher, emb, llm, nil)
		resp, err := svc.Chat(context.Background(), &ChatRequest{Query: "什么是RAG"})
		if err != nil {
			t.Fatalf("Chat error: %v", err)
		}
		if resp.Response == "" {
			t.Error("expected non-empty response")
		}
		if len(resp.Sources) != 1 {
			t.Errorf("expected 1 source, got %d", len(resp.Sources))
		}
	})

	t.Run("empty query", func(t *testing.T) {
		svc := NewChatService(&mockSearcher{}, &mockEmbedder{}, &mockLLM{}, nil)
		_, err := svc.Chat(context.Background(), &ChatRequest{Query: ""})
		if err == nil {
			t.Error("expected error for empty query")
		}
	})

	t.Run("no results found", func(t *testing.T) {
		emb := &mockEmbedder{}
		searcher := &mockSearcher{
			searchHybridFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return []SearchHit{}, nil
			},
		}
		svc := NewChatService(searcher, emb, &mockLLM{}, nil)
		resp, err := svc.Chat(context.Background(), &ChatRequest{Query: "unknown"})
		if err != nil {
			t.Fatalf("Chat error: %v", err)
		}
		if !strings.Contains(resp.Response, "未找到") {
			t.Errorf("expected 'not found' message, got %q", resp.Response)
		}
	})

	t.Run("LLM failure graceful degradation", func(t *testing.T) {
		emb := &mockEmbedder{}
		searcher := &mockSearcher{
			searchHybridFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return []SearchHit{{ChunkID: "c1", Content: "test content", Score: 0.9}}, nil
			},
		}
		llm := &mockLLM{generateFunc: func(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.Message, error) {
			return nil, errors.New("API unavailable")
		}}
		svc := NewChatService(searcher, emb, llm, nil)
		resp, err := svc.Chat(context.Background(), &ChatRequest{Query: "test"})
		if err != nil {
			t.Fatalf("should not error: %v", err)
		}
		if !strings.Contains(resp.Response, "LLM is not available") {
			t.Errorf("expected fallback, got %q", resp.Response)
		}
	})
}

func TestChat_DefaultCollectionIsNotKB1(t *testing.T) {
	// Regression: chat.go used to default the collection to "kb_1" when
	// the request didn't specify one. After the cleanup, the default is
	// "default" (a logical name, not a hardcoded ES index).
	emb := &mockEmbedder{}
	searcher := &mockSearcher{
		searchHybridFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
			return []SearchHit{{ChunkID: "c1", Content: "anything", Score: 0.9}}, nil
		},
	}
	llm := &mockLLM{}
	svc := NewChatService(searcher, emb, llm, nil)

	resp, err := svc.Chat(context.Background(), &ChatRequest{Query: "no-collection-set"})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if resp.Collection == "kb_1" {
		t.Errorf("default collection should no longer be the hardcoded 'kb_1', got %q", resp.Collection)
	}
	if resp.Collection != "default" {
		t.Errorf("expected default collection='default', got %q", resp.Collection)
	}
}

func TestBuildChatPrompt(t *testing.T) {
	hits := []SearchHit{
		{Content: "RAG是一种技术", Score: 0.95},
		{Content: "它结合了检索和生成", Score: 0.85},
	}
	prompt := buildChatPrompt("什么是RAG", hits)
	if !strings.Contains(prompt, "什么是RAG") {
		t.Error("prompt should contain query")
	}
	if !strings.Contains(prompt, "RAG是一种技术") {
		t.Error("prompt should contain content")
	}
}

func TestFormatFallbackResponse(t *testing.T) {
	hits := []SearchHit{{Content: "content1", Score: 0.9}}
	resp := formatFallbackResponse(hits, errors.New("test error"))
	if !strings.Contains(resp, "LLM is not available") {
		t.Error("should mention unavailability")
	}
}
