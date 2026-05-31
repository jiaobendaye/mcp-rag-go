package rag

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// mockEmbedder implements Embedder for testing.
type mockEmbedder struct {
	embedFunc func(ctx context.Context, texts []string) ([][]float64, error)
}

func (m *mockEmbedder) EmbedStrings(ctx context.Context, texts []string) ([][]float64, error) {
	if m.embedFunc != nil {
		return m.embedFunc(ctx, texts)
	}
	// Default: return dummy vectors
	vecs := make([][]float64, len(texts))
	for i := range vecs {
		vecs[i] = []float64{0.1, 0.2, 0.3}
	}
	return vecs, nil
}

func TestSplitText(t *testing.T) {
	t.Run("short text", func(t *testing.T) {
		text := "hello"
		chunks := splitText(text, 100, 20)
		if len(chunks) != 1 {
			t.Errorf("expected 1 chunk, got %d", len(chunks))
		}
		if chunks[0] != "hello" {
			t.Errorf("expected 'hello', got %q", chunks[0])
		}
	})

	t.Run("paragraph split", func(t *testing.T) {
		text := "para1\n\npara2\n\npara3"
		chunks := splitText(text, 10, 2)
		if len(chunks) < 2 {
			t.Errorf("expected at least 2 chunks, got %d", len(chunks))
		}
	})

	t.Run("large text", func(t *testing.T) {
		// Generate 10000-character text
		text := strings.Repeat("hello world ", 1000)
		chunks := splitText(text, 200, 50)
		if len(chunks) < 2 {
			t.Errorf("expected multiple chunks, got %d", len(chunks))
		}
		// Each chunk should be <= chunkSize
		for i, c := range chunks {
			if len(c) > 200 {
				t.Errorf("chunk %d length %d > 200", i, len(c))
			}
		}
	})

	t.Run("empty text", func(t *testing.T) {
		chunks := splitText("", 100, 20)
		if len(chunks) != 1 {
			t.Errorf("expected 1 empty chunk, got %d", len(chunks))
		}
	})
}

func TestIndexText(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		emb := &mockEmbedder{}
		idx := &mockIndexer{
			indexChunksFunc: func(ctx context.Context, chunks []Chunk, vectors [][]float32) error {
				if len(chunks) == 0 {
					t.Error("expected chunks")
				}
				if len(vectors) == 0 {
					t.Error("expected vectors")
				}
				if chunks[0].DocumentID == "" {
					t.Error("expected document_id")
				}
				// Verify chunk_id format
				if !strings.Contains(chunks[0].ID, "_chunk_") {
					t.Errorf("chunk_id format wrong: %s", chunks[0].ID)
				}
				return nil
			},
		}

		pipeline := NewIndexPipeline(emb, idx, 4000, 200)
		result, err := pipeline.IndexText(context.Background(), "hello world", "test.txt")
		if err != nil {
			t.Fatalf("IndexText error: %v", err)
		}
		if result.DocumentID == "" {
			t.Error("expected document_id in result")
		}
		if result.ChunkCount == 0 {
			t.Error("expected chunk count > 0")
		}
	})

	t.Run("empty content", func(t *testing.T) {
		pipeline := NewIndexPipeline(&mockEmbedder{}, &mockIndexer{}, 4000, 200)
		_, err := pipeline.IndexText(context.Background(), "   ", "")
		if err == nil {
			t.Error("expected error for empty content")
		}
	})

	t.Run("embedding failure", func(t *testing.T) {
		emb := &mockEmbedder{
			embedFunc: func(ctx context.Context, texts []string) ([][]float64, error) {
				return nil, errors.New("api error")
			},
		}
		pipeline := NewIndexPipeline(emb, &mockIndexer{}, 4000, 200)
		_, err := pipeline.IndexText(context.Background(), "test content", "test.txt")
		if err == nil {
			t.Error("expected embedding error")
		}
	})

	t.Run("indexing failure", func(t *testing.T) {
		emb := &mockEmbedder{}
		idx := &mockIndexer{
			indexChunksFunc: func(ctx context.Context, chunks []Chunk, vectors [][]float32) error {
				return errors.New("es unavailable")
			},
		}
		pipeline := NewIndexPipeline(emb, idx, 4000, 200)
		_, err := pipeline.IndexText(context.Background(), "test content", "test.txt")
		if err == nil {
			t.Error("expected indexing error")
		}
	})
}

func TestIndexFile(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test.md")
	content := "# Hello\n\nThis is a test document."
	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	pipeline := NewIndexPipeline(&mockEmbedder{}, &mockIndexer{}, 4000, 200)
	result, err := pipeline.IndexFile(context.Background(), filePath)
	if err != nil {
		t.Fatalf("IndexFile error: %v", err)
	}
	if result.ChunkCount == 0 {
		t.Error("expected chunks")
	}
}

func TestIndexFileNotFound(t *testing.T) {
	pipeline := NewIndexPipeline(&mockEmbedder{}, &mockIndexer{}, 4000, 200)
	_, err := pipeline.IndexFile(context.Background(), "/nonexistent/file.txt")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestNewIndexPipelineDefaults(t *testing.T) {
	// Zero/negative values should use defaults
	p := NewIndexPipeline(&mockEmbedder{}, &mockIndexer{}, 0, -1)
	if p.chunkSize != 4000 {
		t.Errorf("expected default chunkSize=4000, got %d", p.chunkSize)
	}
	if p.overlap != 200 {
		t.Errorf("expected default overlap=200, got %d", p.overlap)
	}
}
