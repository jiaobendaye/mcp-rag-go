package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/gin-gonic/gin"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/rag"
)

// httpTestEmbedder implements eino embedding.Embedder
type httpTestEmbedder struct{}

func (e *httpTestEmbedder) EmbedStrings(ctx context.Context, texts []string, _ ...embedding.Option) ([][]float64, error) {
	vecs := make([][]float64, len(texts))
	for i := range vecs {
		vecs[i] = []float64{0.1, 0.2, 0.3}
	}
	return vecs, nil
}

// mockLLM implements eino model.BaseChatModel
type mockLLM struct {
	generateFunc func(ctx context.Context, input []*schema.Message, _ ...model.Option) (*schema.Message, error)
}

func (m *mockLLM) Generate(ctx context.Context, input []*schema.Message, _ ...model.Option) (*schema.Message, error) {
	if m.generateFunc != nil {
		return m.generateFunc(ctx, input)
	}
	return &schema.Message{Role: schema.Assistant, Content: "这是基于知识的回答。"}, nil
}

func (m *mockLLM) Stream(ctx context.Context, input []*schema.Message, _ ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	return nil, nil
}

// testMockSearcher implements rag.Searcher
type testMockSearcher struct {
	searchFunc       func(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]rag.SearchHit, error)
	searchHybridFunc func(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]rag.SearchHit, error)
}

func (m *testMockSearcher) Search(ctx context.Context, qv []float32, tk int, ms float64) ([]rag.SearchHit, error) {
	if m.searchFunc != nil {
		return m.searchFunc(ctx, qv, tk, ms)
	}
	return []rag.SearchHit{}, nil
}

func (m *testMockSearcher) SearchHybrid(ctx context.Context, q string, qv []float32, tk int, ms float64) ([]rag.SearchHit, error) {
	if m.searchHybridFunc != nil {
		return m.searchHybridFunc(ctx, q, qv, tk, ms)
	}
	return []rag.SearchHit{{ChunkID: "c1", Content: "test", Score: 0.95}}, nil
}

func (m *testMockSearcher) SearchWithMode(ctx context.Context, q string, qv []float32, tk int, ms float64, mode string) ([]rag.SearchHit, error) {
	return m.SearchHybrid(ctx, q, qv, tk, ms)
}

func setupTestServer() *gin.Engine {
	gin.SetMode(gin.TestMode)
	cfg := config.DefaultConfig()
	emb := &httpTestEmbedder{}
	chatSvc := rag.NewChatService(&testMockSearcher{}, emb, &mockLLM{}, nil)
	s := New(cfg, nil, nil, nil, nil, chatSvc, &testMockSearcher{}, emb, nil, nil)
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

func TestAddDocumentValidation(t *testing.T) {
	r := setupTestServer()
	t.Run("empty content", func(t *testing.T) {
		body := `{"content":""}`
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
		if _, ok := resp["results"]; !ok {
			t.Error("expected results in response")
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
		body := `{"query":"什么是RAG"}`
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
		body := `{"query":""}`
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/chat", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)
		if w.Code != 400 {
			t.Errorf("expected 400, got %d", w.Code)
		}
	})
}
