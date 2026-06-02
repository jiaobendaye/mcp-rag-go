package server

import (
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
)

func TestDeriveTenantKey(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name       string
		query      string
		wantTenant string
	}{
		{
			name:       "full params",
			query:      "user_id=123&agent_id=456&collection=mykb",
			wantTenant: "u123_a456_mykb",
		},
		{
			name:       "no agent_id",
			query:      "user_id=123&collection=mykb",
			wantTenant: "u123_mykb",
		},
		{
			name:       "only collection",
			query:      "collection=mykb",
			wantTenant: "mykb",
		},
		{
			name:       "no params defaults to default",
			query:      "",
			wantTenant: "default",
		},
		{
			name:       "only user_id",
			query:      "user_id=123",
			wantTenant: "u123_default",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			req, _ := http.NewRequest("GET", "/health?"+tt.query, nil)
			c, _ := gin.CreateTestContext(w)
			c.Request = req

			got := deriveTenantKey(c)
			if got != tt.wantTenant {
				t.Errorf("deriveTenantKey = %q, want %q", got, tt.wantTenant)
			}
		})
	}
}

func setupSecureServer(securityCfg *config.Config) *gin.Engine {
	gin.SetMode(gin.TestMode)

	emb := &httpTestEmbedder{}

	s, _ := New(securityCfg, nil, nil, nil, emb, nil, &mockLLM{}, nil, nil, nil, 0)
	return s.Setup()
}

func TestAuthMiddleware(t *testing.T) {
	t.Run("security disabled allows all", func(t *testing.T) {
		cfg := config.DefaultConfig()
		cfg.SecurityEnabled = false
		r := setupSecureServer(cfg)

		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/health", nil)
		r.ServeHTTP(w, req)

		if w.Code != 200 {
			t.Errorf("expected 200, got %d", w.Code)
		}
	})

	t.Run("valid API key passes", func(t *testing.T) {
		cfg := config.DefaultConfig()
		cfg.SecurityEnabled = true
		cfg.SecurityAllowAnonymous = false
		cfg.SecurityAPIKeys = []string{"sk-secret"}
		r := setupSecureServer(cfg)

		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/health", nil)
		req.Header.Set("x-api-key", "sk-secret")
		r.ServeHTTP(w, req)

		if w.Code != 200 {
			t.Errorf("expected 200, got %d", w.Code)
		}
	})

	t.Run("missing API key returns 401", func(t *testing.T) {
		cfg := config.DefaultConfig()
		cfg.SecurityEnabled = true
		cfg.SecurityAllowAnonymous = false
		r := setupSecureServer(cfg)

		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/health", nil)
		r.ServeHTTP(w, req)

		if w.Code != 401 {
			t.Errorf("expected 401, got %d", w.Code)
		}
	})

	t.Run("invalid API key returns 403", func(t *testing.T) {
		cfg := config.DefaultConfig()
		cfg.SecurityEnabled = true
		cfg.SecurityAllowAnonymous = false
		cfg.SecurityAPIKeys = []string{"sk-correct"}
		r := setupSecureServer(cfg)

		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/health", nil)
		req.Header.Set("x-api-key", "wrong-key")
		r.ServeHTTP(w, req)

		if w.Code != 403 {
			t.Errorf("expected 403, got %d", w.Code)
		}
	})

	t.Run("Authorization Bearer header works", func(t *testing.T) {
		cfg := config.DefaultConfig()
		cfg.SecurityEnabled = true
		cfg.SecurityAllowAnonymous = false
		cfg.SecurityAPIKeys = []string{"sk-bearer"}
		r := setupSecureServer(cfg)

		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/health", nil)
		req.Header.Set("Authorization", "Bearer sk-bearer")
		r.ServeHTTP(w, req)

		if w.Code != 200 {
			t.Errorf("expected 200, got %d", w.Code)
		}
	})

	t.Run("query param API key works", func(t *testing.T) {
		cfg := config.DefaultConfig()
		cfg.SecurityEnabled = true
		cfg.SecurityAllowAnonymous = false
		cfg.SecurityAPIKeys = []string{"sk-query"}
		r := setupSecureServer(cfg)

		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/health?api_key=sk-query", nil)
		r.ServeHTTP(w, req)

		if w.Code != 200 {
			t.Errorf("expected 200, got %d", w.Code)
		}
	})
}

func TestRateLimitMiddleware(t *testing.T) {
	t.Run("rate limit blocks after burst exhausted", func(t *testing.T) {
		cfg := config.DefaultConfig()
		cfg.RateLimitRequestsPerWindow = 5
		cfg.RateLimitWindowSeconds = 60
		cfg.RateLimitBurst = 2 // capacity = 7
		r := setupSecureServer(cfg)

		// First 7 requests should pass
		for i := 0; i < 7; i++ {
			w := httptest.NewRecorder()
			req, _ := http.NewRequest("GET", "/health", nil)
			r.ServeHTTP(w, req)
			if w.Code != 200 {
				t.Errorf("request %d: expected 200, got %d", i+1, w.Code)
			}
		}

		// 8th request should be rate limited
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/health", nil)
		r.ServeHTTP(w, req)

		if w.Code != 429 {
			t.Errorf("expected 429, got %d", w.Code)
		}
		if w.Header().Get("Retry-After") == "" {
			t.Error("expected Retry-After header")
		}
	})

	t.Run("rate limit disabled when limit is 0", func(t *testing.T) {
		cfg := config.DefaultConfig()
		cfg.RateLimitRequestsPerWindow = 0
		r := setupSecureServer(cfg)

		// Many requests should all pass
		for i := 0; i < 20; i++ {
			w := httptest.NewRecorder()
			req, _ := http.NewRequest("GET", "/health", nil)
			r.ServeHTTP(w, req)
			if w.Code != 200 {
				t.Errorf("request %d: expected 200, got %d (rate limit should be disabled)", i+1, w.Code)
				break
			}
		}
	})
}

func TestQuotaMiddleware(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.QuotaMaxUploadFiles = 2 // Only 2 files allowed

	t.Run("upload exceeds file count quota", func(t *testing.T) {
		r := setupSecureServer(cfg)

		// Create multipart with 3 files
		body := &strings.Builder{}
		w := multipart.NewWriter(body)
		for i := 0; i < 3; i++ {
			part, _ := w.CreateFormFile("files", "test.txt")
			part.Write([]byte("content"))
		}
		w.Close()

		req := httptest.NewRequest("POST", "/upload-files", strings.NewReader(body.String()))
		req.Header.Set("Content-Type", w.FormDataContentType())
		resp := httptest.NewRecorder()
		r.ServeHTTP(resp, req)

		if resp.Code != 413 {
			t.Errorf("expected 413, got %d", resp.Code)
		}
	})

	t.Run("add-document exceeds char quota", func(t *testing.T) {
		cfg2 := config.DefaultConfig()
		cfg2.QuotaMaxIndexChars = 5 // Only 5 chars
		r := setupSecureServer(cfg2)

		body := `{"content":"this is a long text that exceeds limit"}`
		req := httptest.NewRequest("POST", "/add-document", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		r.ServeHTTP(resp, req)

		if resp.Code != 413 {
			t.Errorf("expected 413, got %d: %s", resp.Code, resp.Body.String())
		}
	})
}


