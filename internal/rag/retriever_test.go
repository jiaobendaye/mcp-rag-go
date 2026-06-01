package rag

import (
	"context"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/retriever"
)

// mockRetrieverEmbedder implements embedding.Embedder for retriever tests.
type mockRetrieverEmbedder struct {
	embedFunc func(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error)
}

func (m *mockRetrieverEmbedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
	if m.embedFunc != nil {
		return m.embedFunc(ctx, texts, opts...)
	}
	vecs := make([][]float64, len(texts))
	for i := range vecs {
		vecs[i] = []float64{0.1, 0.2, 0.3}
	}
	return vecs, nil
}

// mockRetrieverSearcher implements Searcher with custom SearchWithMode behavior.
type mockRetrieverSearcher struct {
	searchWithModeFunc func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error)
}

func (m *mockRetrieverSearcher) Search(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
	return nil, nil
}

func (m *mockRetrieverSearcher) SearchHybrid(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
	return nil, nil
}

func (m *mockRetrieverSearcher) SearchWithMode(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error) {
	if m.searchWithModeFunc != nil {
		return m.searchWithModeFunc(ctx, query, queryVector, topK, minScore, mode)
	}
	return nil, nil
}

func (m *mockRetrieverSearcher) SearchWithWeights(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string, weights SearchWeights) ([]SearchHit, error) {
	return m.SearchWithMode(ctx, query, queryVector, topK, minScore, mode)
}

func TestESRetrieverRetrieve(t *testing.T) {
	t.Run("basic retrieval", func(t *testing.T) {
		emb := &mockRetrieverEmbedder{}
		search := &mockRetrieverSearcher{
			searchWithModeFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error) {
				if query != "什么是AI" {
					t.Errorf("expected query '什么是AI', got %q", query)
				}
				if topK != 5 {
					t.Errorf("expected topK=5, got %d", topK)
				}
				if minScore != 0.7 {
					t.Errorf("expected minScore=0.7, got %f", minScore)
				}
				return []SearchHit{
					{ChunkID: "c1", DocumentID: "d1", Content: "RAG是检索增强生成技术", Score: 0.95, Filename: "doc1.md", Source: "doc1.md", ChunkIndex: 0},
				}, nil
			},
		}

		r := NewESRetriever(emb, search, "test_index", "hybrid", 5, 0.7)
		docs, err := r.Retrieve(context.Background(), "什么是AI")
		if err != nil {
			t.Fatalf("Retrieve error: %v", err)
		}
		if len(docs) != 1 {
			t.Fatalf("expected 1 doc, got %d", len(docs))
		}
		if docs[0].Content != "RAG是检索增强生成技术" {
			t.Errorf("unexpected content: %s", docs[0].Content)
		}
		if docs[0].MetaData["filename"] != "doc1.md" {
			t.Errorf("unexpected filename: %v", docs[0].MetaData["filename"])
		}
		if docs[0].MetaData["score"] != 0.95 {
			t.Errorf("unexpected score: %v", docs[0].MetaData["score"])
		}
		if docs[0].MetaData["document_id"] != "d1" {
			t.Errorf("unexpected document_id: %v", docs[0].MetaData["document_id"])
		}
		if docs[0].MetaData["chunk_id"] != "c1" {
			t.Errorf("unexpected chunk_id: %v", docs[0].MetaData["chunk_id"])
		}
		if docs[0].MetaData["chunk_index"] != 0 {
			t.Errorf("unexpected chunk_index: %v", docs[0].MetaData["chunk_index"])
		}
	})

	t.Run("empty results", func(t *testing.T) {
		emb := &mockRetrieverEmbedder{}
		search := &mockRetrieverSearcher{
			searchWithModeFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error) {
				return []SearchHit{}, nil
			},
		}

		r := NewESRetriever(emb, search, "test_index", "hybrid", 5, 0.7)
		docs, err := r.Retrieve(context.Background(), "unknown")
		if err != nil {
			t.Fatalf("Retrieve error: %v", err)
		}
		if len(docs) != 0 {
			t.Errorf("expected 0 docs, got %d", len(docs))
		}
	})

	t.Run("option override topK", func(t *testing.T) {
		emb := &mockRetrieverEmbedder{}
		search := &mockRetrieverSearcher{
			searchWithModeFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error) {
				if topK != 10 {
					t.Errorf("expected topK=10 from option override, got %d", topK)
				}
				if minScore != 0.5 {
					t.Errorf("expected minScore=0.5 from option override, got %f", minScore)
				}
				return []SearchHit{
					{ChunkID: "c1", Content: "test", Score: 0.9, Filename: "f1", Source: "s1", ChunkIndex: 0},
				}, nil
			},
		}

		r := NewESRetriever(emb, search, "test_index", "hybrid", 5, 0.7)
		_, err := r.Retrieve(context.Background(), "query",
			retriever.WithTopK(10),
			retriever.WithScoreThreshold(0.5),
		)
		if err != nil {
			t.Fatalf("Retrieve error: %v", err)
		}
	})

	t.Run("multi-hit retrieval", func(t *testing.T) {
		emb := &mockRetrieverEmbedder{}
		search := &mockRetrieverSearcher{
			searchWithModeFunc: func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error) {
				return []SearchHit{
					{ChunkID: "c1", DocumentID: "d1", Content: "first", Score: 0.95, Filename: "a.md", Source: "a.md", ChunkIndex: 0},
					{ChunkID: "c2", DocumentID: "d2", Content: "second", Score: 0.85, Filename: "b.md", Source: "b.md", ChunkIndex: 0},
					{ChunkID: "c3", DocumentID: "d3", Content: "third", Score: 0.75, Filename: "c.md", Source: "c.md", ChunkIndex: 0},
				}, nil
			},
		}

		r := NewESRetriever(emb, search, "test_index", "hybrid", 5, 0.7)
		docs, err := r.Retrieve(context.Background(), "multi")
		if err != nil {
			t.Fatalf("Retrieve error: %v", err)
		}
		if len(docs) != 3 {
			t.Fatalf("expected 3 docs, got %d", len(docs))
		}
		for i, doc := range docs {
			if doc.Content == "" {
				t.Errorf("doc %d has empty content", i)
			}
			if doc.MetaData == nil {
				t.Errorf("doc %d has nil MetaData", i)
			}
		}
	})
}
