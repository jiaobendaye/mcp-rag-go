package rag

import (
	"context"
	"errors"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
)

// mockEmbedder implements eino embedding.Embedder for testing.
type mockEmbedder struct {
	embedFunc func(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error)
}

func (m *mockEmbedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
	if m.embedFunc != nil {
		return m.embedFunc(ctx, texts, opts...)
	}
	vecs := make([][]float64, len(texts))
	for i := range vecs {
		vecs[i] = []float64{0.1, 0.2, 0.3}
	}
	return vecs, nil
}

// mockIndexer implements Indexer for testing.
type mockIndexer struct {
	ensureIndexFunc func(ctx context.Context, dims int) error
	indexChunksFunc func(ctx context.Context, chunks []Chunk, vectors [][]float32) error
}

func (m *mockIndexer) EnsureIndex(ctx context.Context, dims int) error {
	if m.ensureIndexFunc != nil {
		return m.ensureIndexFunc(ctx, dims)
	}
	return nil
}

func (m *mockIndexer) IndexChunks(ctx context.Context, chunks []Chunk, vectors [][]float32) error {
	if m.indexChunksFunc != nil {
		return m.indexChunksFunc(ctx, chunks, vectors)
	}
	return nil
}

// mockSearcher implements Searcher for testing.
type mockSearcher struct {
	searchFunc       func(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error)
	searchHybridFunc func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error)
}

func (m *mockSearcher) Search(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
	if m.searchFunc != nil {
		return m.searchFunc(ctx, queryVector, topK, minScore)
	}
	return nil, nil
}

func (m *mockSearcher) SearchHybrid(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
	if m.searchHybridFunc != nil {
		return m.searchHybridFunc(ctx, query, queryVector, topK, minScore)
	}
	return nil, nil
}

func (m *mockSearcher) SearchWithMode(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error) {
	return m.SearchHybrid(ctx, query, queryVector, topK, minScore) // Default to hybrid
}

func (m *mockSearcher) SearchWithWeights(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string, weights SearchWeights) ([]SearchHit, error) {
	return m.SearchWithMode(ctx, query, queryVector, topK, minScore, mode)
}

func TestGenerateChunkID(t *testing.T) {
	tests := []struct {
		name     string
		docID    string
		index    int
		expected string
	}{
		{"basic", "abc123", 0, "abc123_chunk_0000"},
		{"large index", "abc123", 42, "abc123_chunk_0042"},
		{"empty doc", "", 1, "_chunk_0001"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GenerateChunkID(tt.docID, tt.index)
			if got != tt.expected {
				t.Errorf("GenerateChunkID(%q, %d) = %q, want %q", tt.docID, tt.index, got, tt.expected)
			}
		})
	}
}

func TestGenerateDocID(t *testing.T) {
	id1 := GenerateDocID("hello world")
	id2 := GenerateDocID("hello world")
	id3 := GenerateDocID("different content")

	if id1 != id2 {
		t.Error("same content should generate same doc ID")
	}
	if id1 == id3 {
		t.Error("different content should generate different doc IDs")
	}
	if len(id1) != 32 { // MD5 hex is 32 chars
		t.Errorf("doc ID should be 32 hex chars, got %d", len(id1))
	}
}

func TestMockIndexerEnsureIndex(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		m := &mockIndexer{
			ensureIndexFunc: func(ctx context.Context, dims int) error {
				if dims != 1536 {
					t.Errorf("expected dims=1536, got %d", dims)
				}
				return nil
			},
		}
		if err := m.EnsureIndex(context.Background(), 1536); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("error", func(t *testing.T) {
		m := &mockIndexer{
			ensureIndexFunc: func(ctx context.Context, dims int) error {
				return errors.New("index creation failed")
			},
		}
		if err := m.EnsureIndex(context.Background(), 768); err == nil {
			t.Error("expected error, got nil")
		}
	})
}

func TestMockIndexerIndexChunks(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		m := &mockIndexer{
			indexChunksFunc: func(ctx context.Context, chunks []Chunk, vectors [][]float32) error {
				if len(chunks) != 2 {
					t.Errorf("expected 2 chunks, got %d", len(chunks))
				}
				if len(vectors) != 2 {
					t.Errorf("expected 2 vectors, got %d", len(vectors))
				}
				return nil
			},
		}
		chunks := []Chunk{{ID: "c1"}, {ID: "c2"}}
		vectors := [][]float32{{0.1, 0.2}, {0.3, 0.4}}
		if err := m.IndexChunks(context.Background(), chunks, vectors); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("empty chunks", func(t *testing.T) {
		m := &mockIndexer{}
		if err := m.IndexChunks(context.Background(), nil, nil); err != nil {
			t.Errorf("empty chunks should not error: %v", err)
		}
	})

	t.Run("mismatch error", func(t *testing.T) {
		m := &mockIndexer{
			indexChunksFunc: func(ctx context.Context, chunks []Chunk, vectors [][]float32) error {
				if len(chunks) != len(vectors) {
					return errors.New("chunks and vectors length mismatch")
				}
				return nil
			},
		}
		chunks := []Chunk{{ID: "c1"}, {ID: "c2"}}
		vectors := [][]float32{{0.1}}
		if err := m.IndexChunks(context.Background(), chunks, vectors); err == nil {
			t.Error("expected mismatch error, got nil")
		}
	})
}

func TestMockSearcherSearch(t *testing.T) {
	t.Run("returns results", func(t *testing.T) {
		m := &mockSearcher{
			searchFunc: func(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return []SearchHit{
					{ChunkID: "c1", Content: "hello", Score: 0.95},
					{ChunkID: "c2", Content: "world", Score: 0.85},
				}, nil
			},
		}
		hits, err := m.Search(context.Background(), []float32{0.1}, 5, 0.7)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(hits) != 2 {
			t.Errorf("expected 2 hits, got %d", len(hits))
		}
		if hits[0].Score != 0.95 {
			t.Errorf("expected score 0.95, got %f", hits[0].Score)
		}
	})

	t.Run("empty results", func(t *testing.T) {
		m := &mockSearcher{
			searchFunc: func(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return []SearchHit{}, nil
			},
		}
		hits, err := m.Search(context.Background(), []float32{0.1}, 5, 0.7)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(hits) != 0 {
			t.Errorf("expected 0 hits, got %d", len(hits))
		}
	})

	t.Run("search error", func(t *testing.T) {
		m := &mockSearcher{
			searchFunc: func(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
				return nil, errors.New("connection refused")
			},
		}
		_, err := m.Search(context.Background(), []float32{0.1}, 5, 0.7)
		if err == nil {
			t.Error("expected error, got nil")
		}
	})
}
