package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/rag"
)

// mockEmbedder implements rag.Embedder for testing
type mockEmbedder struct {
	embedFunc func(ctx context.Context, texts []string) ([][]float64, error)
}

func (m *mockEmbedder) EmbedStrings(ctx context.Context, texts []string) ([][]float64, error) {
	if m.embedFunc != nil {
		return m.embedFunc(ctx, texts)
	}
	return [][]float64{{0.1, 0.2}}, nil
}

// mockSearcher implements rag.Searcher for testing
type testMockSearcher struct {
	searchFunc       func(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]rag.SearchHit, error)
	searchHybridFunc func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]rag.SearchHit, error)
}

func (m *testMockSearcher) Search(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]rag.SearchHit, error) {
	if m.searchFunc != nil {
		return m.searchFunc(ctx, queryVector, topK, minScore)
	}
	return []rag.SearchHit{
		{ChunkID: "c1", Content: "test", Score: 0.95, Source: "doc1.md", Filename: "doc1.md"},
	}, nil
}

func (m *testMockSearcher) SearchHybrid(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]rag.SearchHit, error) {
	if m.searchHybridFunc != nil {
		return m.searchHybridFunc(ctx, query, queryVector, topK, minScore)
	}
	return []rag.SearchHit{
		{ChunkID: "c1", Content: "test", Score: 0.95, Source: "doc1.md", Filename: "doc1.md"},
	}, nil
}

func (m *testMockSearcher) SearchWithMode(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]rag.SearchHit, error) {
	return m.SearchHybrid(ctx, query, queryVector, topK, minScore)
}

// mockIndexer for pipeline
type mockIndexer struct {
	ensureFunc   func(ctx context.Context, dims int) error
	indexFunc    func(ctx context.Context, chunks []rag.Chunk, vectors [][]float32) error
}

func (m *mockIndexer) EnsureIndex(ctx context.Context, dims int) error {
	if m.ensureFunc != nil {
		return m.ensureFunc(ctx, dims)
	}
	return nil
}

func (m *mockIndexer) IndexChunks(ctx context.Context, chunks []rag.Chunk, vectors [][]float32) error {
	if m.indexFunc != nil {
		return m.indexFunc(ctx, chunks, vectors)
	}
	return nil
}

// mockEmbedder implementing rag.Embedder interface
type httpTestEmbedder struct{}

func (e *httpTestEmbedder) EmbedStrings(ctx context.Context, texts []string) ([][]float64, error) {
	vecs := make([][]float64, len(texts))
	for i := range vecs {
		vecs[i] = []float64{0.1, 0.2, 0.3}
	}
	return vecs, nil
}

// mockLLM for chat service
type mockLLM struct {
	generateFunc func(ctx context.Context, prompt string) (string, error)
}

func (m *mockLLM) Generate(ctx context.Context, prompt string) (string, error) {
	if m.generateFunc != nil {
		return m.generateFunc(ctx, prompt)
	}
	return "这是基于知识的回答。", nil
}

func setupTestServer() *gin.Engine {
	gin.SetMode(gin.TestMode)

	cfg := config.DefaultConfig()
	emb := &httpTestEmbedder{}
	pipeline := rag.NewIndexPipeline(emb, &mockIndexer{}, 4000, 200)
	chatSvc := rag.NewChatService(&testMockSearcher{}, emb, &mockLLM{})
	searcher := &testMockSearcher{}

	s := New(cfg, pipeline, chatSvc, searcher, emb)
	return s.Setup()
}

func TestHealthEndpoint(t *testing.T) {
	r := setupTestServer()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/health", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var body map[string]string
	json.NewDecoder(w.Body).Decode(&body)
	if body["status"] != "ok" {
		t.Errorf("expected status=ok, got %s", body["status"])
	}
}

func TestAddDocument(t *testing.T) {
	r := setupTestServer()

	t.Run("valid request", func(t *testing.T) {
		body := `{"content": "Hello world, this is a test document."}`
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/add-document", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != 200 {
			t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
		}

		var resp map[string]any
		json.NewDecoder(w.Body).Decode(&resp)
		if resp["document_id"] == nil || resp["document_id"] == "" {
			t.Error("expected document_id in response")
		}
	})

	t.Run("empty content", func(t *testing.T) {
		body := `{"content": ""}`
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/add-document", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != 400 {
			t.Errorf("expected 400, got %d", w.Code)
		}
	})

	t.Run("invalid json", func(t *testing.T) {
		body := `not json`
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/add-document", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != 400 {
			t.Errorf("expected 400, got %d", w.Code)
		}
	})
}

func TestSearchEndpoint(t *testing.T) {
	r := setupTestServer()

	t.Run("valid search", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/search?query=test&limit=3", nil)
		r.ServeHTTP(w, req)

		if w.Code != 200 {
			t.Errorf("expected 200, got %d", w.Code)
		}

		var resp map[string]any
		json.NewDecoder(w.Body).Decode(&resp)
		if resp["query"] != "test" {
			t.Errorf("expected query=test, got %v", resp["query"])
		}
		if results, ok := resp["results"].([]any); !ok || len(results) == 0 {
			t.Error("expected non-empty results")
		}
	})

	t.Run("missing query", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/search", nil)
		r.ServeHTTP(w, req)

		if w.Code != 400 {
			t.Errorf("expected 400, got %d", w.Code)
		}
	})
}

func TestChatEndpoint(t *testing.T) {
	r := setupTestServer()

	t.Run("valid chat", func(t *testing.T) {
		body := `{"query": "什么是RAG"}`
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/chat", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != 200 {
			t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
		}

		var resp map[string]any
		json.NewDecoder(w.Body).Decode(&resp)
		if resp["response"] == nil || resp["response"] == "" {
			t.Error("expected non-empty response")
		}
	})

	t.Run("empty query", func(t *testing.T) {
		body := `{"query": ""}`
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/chat", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != 400 {
			t.Errorf("expected 400, got %d", w.Code)
		}
	})
}
