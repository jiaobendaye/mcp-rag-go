package rag

import (
	"context"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

func TestBuildRetrievalGraph(t *testing.T) {
	t.Run("graph compiles successfully", func(t *testing.T) {
		emb := &mockEmbedder{}
		search := &mockSearcher{
			searchHybridFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return []SearchHit{
					{ChunkID: "c1", Content: "RAG是检索增强生成技术", Score: 0.95, Filename: "doc1.md", Source: "doc1.md", ChunkIndex: 0},
				}, nil
			},
		}
		llm := &mockLLM{}

		runnable, err := BuildRetrievalGraph(context.Background(), emb, search, llm, "hybrid")
		if err != nil {
			t.Fatalf("BuildRetrievalGraph error: %v", err)
		}
		if runnable == nil {
			t.Fatal("expected non-nil runnable")
		}
	})

	t.Run("graph invocation with results", func(t *testing.T) {
		emb := &mockEmbedder{}
		search := &mockSearcher{
			searchHybridFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return []SearchHit{
					{
						ChunkID:    "c1",
						DocumentID: "d1",
						Content:    "RAG（Retrieval-Augmented Generation）是检索增强生成技术",
						Score:      0.95,
						Filename:   "intro.md",
						Source:     "intro.md",
						ChunkIndex: 0,
					},
				}, nil
			},
		}
		llm := &mockLLM{}

		runnable, err := BuildRetrievalGraph(context.Background(), emb, search, llm, "hybrid")
		if err != nil {
			t.Fatalf("BuildRetrievalGraph error: %v", err)
		}

		answer, err := runnable.Invoke(context.Background(), "什么是RAG？")
		if err != nil {
			t.Fatalf("Invoke error: %v", err)
		}
		if answer == "" {
			t.Error("expected non-empty answer")
		}
		// The mock LLM returns "这是基于知识的回答。"
		if answer != "这是基于知识的回答。" {
			t.Errorf("unexpected answer: %s", answer)
		}
	})

	t.Run("graph invocation with empty results", func(t *testing.T) {
		emb := &mockEmbedder{}
		search := &mockSearcher{
			searchHybridFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return []SearchHit{}, nil
			},
		}
		llm := &mockLLM{}

		runnable, err := BuildRetrievalGraph(context.Background(), emb, search, llm, "hybrid")
		if err != nil {
			t.Fatalf("BuildRetrievalGraph error: %v", err)
		}

		answer, err := runnable.Invoke(context.Background(), "未知问题")
		if err != nil {
			t.Fatalf("Invoke error: %v", err)
		}
		if answer == "" {
			t.Error("expected non-empty answer even with no results")
		}
		// The mock LLM returns "这是基于知识的回答。"
		if answer != "这是基于知识的回答。" {
			t.Errorf("unexpected answer: %s", answer)
		}
	})

	t.Run("state carries query to prompt assembly", func(t *testing.T) {
		var capturedQuery string

		emb := &mockEmbedder{}
		search := &mockSearcher{
			searchHybridFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				capturedQuery = query // capture the query as seen by the retriever
				return []SearchHit{
					{ChunkID: "c1", Content: "测试内容", Score: 0.9, Filename: "test.md", Source: "test.md", ChunkIndex: 0},
				}, nil
			},
		}

		// A custom LLM that returns the captured context to verify it was assembled
		llm := &mockLLM{
			generateFunc: func(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.Message, error) {
				// Find the user message which has the context
				for _, msg := range input {
					if msg.Role == schema.User {
						if !strings.Contains(msg.Content, "什么是AI") {
							t.Errorf("assembled prompt should contain the query '什么是AI', got: %s", msg.Content)
						}
						if !strings.Contains(msg.Content, "测试内容") {
							t.Errorf("assembled prompt should contain '测试内容', got: %s", msg.Content)
						}
					}
				}
				return &schema.Message{Role: schema.Assistant, Content: "回答: AI是人工智能"}, nil
			},
		}

		runnable, err := BuildRetrievalGraph(context.Background(), emb, search, llm, "hybrid")
		if err != nil {
			t.Fatalf("BuildRetrievalGraph error: %v", err)
		}

		answer, err := runnable.Invoke(context.Background(), "什么是AI")
		if err != nil {
			t.Fatalf("Invoke error: %v", err)
		}
		if capturedQuery != "什么是AI" {
			t.Errorf("retriever should receive the original query, got: %s", capturedQuery)
		}
		if answer != "回答: AI是人工智能" {
			t.Errorf("unexpected answer: %s", answer)
		}
	})
}
