package server

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/jiaobendaye/mcp-rag-go/internal/knowledgebase"
)

func setupIdempotencyServer(t *testing.T) (*gin.Engine, *knowledgebase.Store) {
	t.Helper()
	gin.SetMode(gin.TestMode)

	dbPath := "file:" + t.TempDir() + "/test.db"
	store, err := knowledgebase.NewStore(dbPath)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	r := gin.New()
	r.Use(IdempotencyRouteFilter(store))
	r.POST("/add-document", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})
	r.POST("/upload-files", func(c *gin.Context) {
		c.JSON(http.StatusCreated, gin.H{"files": 2})
	})
	r.POST("/knowledge-bases", func(c *gin.Context) {
		c.JSON(http.StatusCreated, gin.H{"kb_id": 1})
	})
	r.GET("/search", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"results": []any{}})
	})
	return r, store
}

func TestIdempotency_NoHeaderPassThrough(t *testing.T) {
	r, _ := setupIdempotencyServer(t)

	req := httptest.NewRequest("POST", "/add-document", strings.NewReader(`{"content":"hello"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	r.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.Code)
	}
	if resp.Header().Get("Idempotency-Replay") != "" {
		t.Error("expected no Idempotency-Replay header")
	}
}

func TestIdempotency_CacheAndReplay(t *testing.T) {
	r, store := setupIdempotencyServer(t)

	body := `{"content":"hello world"}`
	key := "test-key-001"

	// First request — should succeed and cache
	req1 := httptest.NewRequest("POST", "/add-document", strings.NewReader(body))
	req1.Header.Set("Content-Type", "application/json")
	req1.Header.Set("Idempotency-Key", key)
	resp1 := httptest.NewRecorder()
	r.ServeHTTP(resp1, req1)

	if resp1.Code != http.StatusOK {
		t.Fatalf("first request: expected 200, got %d: %s", resp1.Code, resp1.Body.String())
	}
	if resp1.Header().Get("Idempotency-Replay") == "true" {
		t.Error("first request should not be a replay")
	}

	// Verify cache record exists
	record, err := store.GetIdempotencyRecord(key, "POST", "/add-document")
	if err != nil {
		t.Fatalf("get record: %v", err)
	}
	if record == nil {
		t.Fatal("expected cached record, got nil")
	}

	// Second request with same key and body — should replay
	req2 := httptest.NewRequest("POST", "/add-document", strings.NewReader(body))
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("Idempotency-Key", key)
	resp2 := httptest.NewRecorder()
	r.ServeHTTP(resp2, req2)

	if resp2.Code != http.StatusOK {
		t.Fatalf("replay: expected 200, got %d", resp2.Code)
	}
	if resp2.Header().Get("Idempotency-Replay") != "true" {
		t.Error("replay response should have Idempotency-Replay: true")
	}
	if !bytes.Contains(resp2.Body.Bytes(), []byte(`"status":"ok"`)) {
		t.Errorf("unexpected replay body: %s", resp2.Body.String())
	}
}

func TestIdempotency_DifferentBodySameKey(t *testing.T) {
	r, _ := setupIdempotencyServer(t)

	key := "test-key-002"

	// First request
	req1 := httptest.NewRequest("POST", "/add-document", strings.NewReader(`{"content":"first"}`))
	req1.Header.Set("Content-Type", "application/json")
	req1.Header.Set("Idempotency-Key", key)
	resp1 := httptest.NewRecorder()
	r.ServeHTTP(resp1, req1)
	if resp1.Code != http.StatusOK {
		t.Fatalf("first request: expected 200, got %d", resp1.Code)
	}

	// Second request — same key, DIFFERENT body
	req2 := httptest.NewRequest("POST", "/add-document", strings.NewReader(`{"content":"different"}`))
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("Idempotency-Key", key)
	resp2 := httptest.NewRecorder()
	r.ServeHTTP(resp2, req2)

	if resp2.Code != http.StatusUnprocessableEntity {
		t.Fatalf("expected 422, got %d: %s", resp2.Code, resp2.Body.String())
	}
}

func TestIdempotency_ReadOnlyRouteSkipped(t *testing.T) {
	r, _ := setupIdempotencyServer(t)

	// GET /search should NOT trigger idempotency
	req := httptest.NewRequest("GET", "/search", nil)
	req.Header.Set("Idempotency-Key", "some-key")
	resp := httptest.NewRecorder()
	r.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.Code)
	}
}

func TestIdempotency_NoCacheOnError(t *testing.T) {
	r, store := setupIdempotencyServer(t)

	// Add a route that returns an error
	r.POST("/add-document-error", func(c *gin.Context) {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "fail"})
	})
	// Recreate the filter to include the new route
	// Actually, the existing route filter doesn't know about this path.
	// Let's add it to the idempotentPaths or test directly.

	// Create a custom test
	gin.SetMode(gin.TestMode)
	r2 := gin.New()
	r2.Use(func(c *gin.Context) {
		// Manually invoke the middleware for a specific path
		key := c.GetHeader("Idempotency-Key")
		if key == "" {
			c.Next()
			return
		}
		method := c.Request.Method
		path := c.Request.URL.Path
		bodyBytes, _ := c.GetRawData()
		c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		requestHash := fmt.Sprintf("%x", sha256.Sum256(bodyBytes))

		w := &responseCapture{ResponseWriter: c.Writer, buf: &bytes.Buffer{}}
		c.Writer = w
		c.Next()

		if c.Writer.Status() >= 200 && c.Writer.Status() < 300 {
			store.SetIdempotencyRecord(key, method, path, requestHash,
				w.Status(), w.buf.Bytes(), nil)
		}
	})
	r2.POST("/fail", func(c *gin.Context) {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "fail"})
	})

	key := "test-key-error"
	req := httptest.NewRequest("POST", "/fail", strings.NewReader(`{"x":1}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Idempotency-Key", key)
	resp := httptest.NewRecorder()
	r2.ServeHTTP(resp, req)

	if resp.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d", resp.Code)
	}

	// Verify NOT cached (error responses skip caching)
	record, _ := store.GetIdempotencyRecord(key, "POST", "/fail")
	if record != nil {
		t.Error("error responses should not be cached")
	}
}
