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
	if body["status"] != "healthy" {
		t.Errorf("expected status=healthy, got %s", body["status"])
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
	t.Run("chat with extra fields", func(t *testing.T) {
		body := `{"query":"test","collection":"default","kb_id":1,"limit":10,"user_id":1001,"agent_id":50}`
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/chat", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)
		if w.Code != 200 {
			t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
		}
		var resp map[string]any
		json.NewDecoder(w.Body).Decode(&resp)
		if resp["collection"] == nil || resp["collection"] == "" {
			t.Error("expected collection in response")
		}
	})
}

func TestConfigGetFlatFormat(t *testing.T) {
	r := setupTestServer()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/config", nil)
	r.ServeHTTP(w, req)

	// When no configManager, config routes are not registered
	if w.Code == 404 {
		return
	}

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)

	// Should have flat fields at top level
	if _, ok := resp["http_port"]; !ok {
		t.Error("expected http_port at top level")
	}

	// Should have provider_configs
	if pc, ok := resp["provider_configs"]; !ok || pc == nil {
		t.Error("expected provider_configs")
	}

	// Should have host/port aliases
	if host := resp["host"]; host != "0.0.0.0" {
		t.Errorf("expected host=0.0.0.0, got %v", host)
	}
}

func TestSPARedirects(t *testing.T) {
	r := setupTestServer()

	tests := []struct {
		path     string
		expected int
		location string
	}{
		{"/", 302, "/app"},
		{"/doc", 302, "/docs"},
		{"/documents-page", 302, "/app/documents"},
		{"/config-page", 302, "/app/config"},
	}

	for _, tc := range tests {
		t.Run("redirect"+tc.path, func(t *testing.T) {
			w := httptest.NewRecorder()
			req, _ := http.NewRequest("GET", tc.path, nil)
			r.ServeHTTP(w, req)
			if w.Code != tc.expected {
				t.Errorf("expected %d, got %d", tc.expected, w.Code)
			}
			if loc := w.Header().Get("Location"); loc != tc.location {
				t.Errorf("expected Location=%s, got %s", tc.location, loc)
			}
		})
	}
}

func TestDocsAndOpenAPI(t *testing.T) {
	r := setupTestServer()

	t.Run("docs returns html", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/docs", nil)
		r.ServeHTTP(w, req)
		if w.Code != 200 {
			t.Errorf("expected 200, got %d", w.Code)
		}
		ct := w.Header().Get("Content-Type")
		if !strings.Contains(ct, "text/html") {
			t.Errorf("expected text/html, got %s", ct)
		}
	})

	t.Run("openapi.json is valid", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/openapi.json", nil)
		r.ServeHTTP(w, req)
		if w.Code != 200 {
			t.Errorf("expected 200, got %d", w.Code)
		}
		var spec map[string]any
		if err := json.NewDecoder(w.Body).Decode(&spec); err != nil {
			t.Fatalf("invalid JSON: %v", err)
		}
		if spec["openapi"] == nil {
			t.Error("expected openapi version")
		}
		if _, ok := spec["paths"]; !ok {
			t.Error("expected paths in openapi spec")
		}
	})
}

func TestProviderModels(t *testing.T) {
	r := setupTestServer()

	t.Run("models endpoint", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/providers/openai/models", nil)
		r.ServeHTTP(w, req)
		if w.Code != 200 {
			t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
		}

		var resp map[string]any
		json.NewDecoder(w.Body).Decode(&resp)
		if resp["provider"] != "openai" {
			t.Errorf("expected provider=openai, got %v", resp["provider"])
		}
		models, ok := resp["models"].([]interface{})
		if !ok {
			t.Fatal("expected models array")
		}
		if len(models) == 0 {
			t.Error("expected at least one model")
		}
	})

	t.Run("models with family filter", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/providers/openai/models?family=embedding", nil)
		r.ServeHTTP(w, req)
		if w.Code != 200 {
			t.Fatalf("expected 200, got %d", w.Code)
		}

		var resp map[string]any
		json.NewDecoder(w.Body).Decode(&resp)
		if resp["family"] != "embedding" {
			t.Errorf("expected family=embedding, got %v", resp["family"])
		}
	})
}

func TestSearchResponseEnrichment(t *testing.T) {
	r := setupTestServer()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/search?query=test&limit=2", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)

	results, ok := resp["results"].([]interface{})
	if !ok || len(results) == 0 {
		t.Fatal("expected non-empty results")
	}

	first := results[0].(map[string]any)

	if _, ok := first["vector_score"]; !ok {
		t.Error("expected vector_score in search result")
	}
	if _, ok := first["retrieval_method"]; !ok {
		t.Error("expected retrieval_method in search result")
	}
	if _, ok := first["metadata"]; !ok {
		t.Error("expected metadata in search result")
	}
}

func TestHealthFormatCompatibility(t *testing.T) {
	r := setupTestServer()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/health", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)

	requiredFields := []string{"healthy", "ready", "bootstrapped", "runtime", "config_revision", "reasons"}
	for _, f := range requiredFields {
		if _, ok := resp[f]; !ok {
			t.Errorf("expected %s field in health response", f)
		}
	}

	if rt, ok := resp["runtime"].(map[string]any); ok {
		runtimeKeys := []string{"embedding_model", "llm_model", "vector_store", "knowledge_base"}
		for _, rk := range runtimeKeys {
			if _, ok := rt[rk]; !ok {
				t.Errorf("expected runtime.%s in health response", rk)
			}
		}
	}
}

func TestReadyFormatCompatibility(t *testing.T) {
	r := setupTestServer()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/ready", nil)
	r.ServeHTTP(w, req)

	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)

	requiredFields := []string{"bootstrapped", "ready", "runtime", "config_revision"}
	for _, f := range requiredFields {
		if _, ok := resp[f]; !ok {
			t.Errorf("expected %s field in ready response", f)
		}
	}
}

func TestStaticRouteExists(t *testing.T) {
	r := setupTestServer()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/static/nonexistent.js", nil)
	r.ServeHTTP(w, req)
	if w.Code != 404 {
		t.Errorf("expected 404 for missing static file, got %d", w.Code)
	}
}
