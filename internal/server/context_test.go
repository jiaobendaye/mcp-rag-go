package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func setupTracingRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(TracingMiddleware())
	r.GET("/test", func(c *gin.Context) {
		rc := GetRequestContext(c)
		if rc == nil {
			c.String(500, "no context")
			return
		}
		c.JSON(200, gin.H{
			"request_id": rc.RequestID,
			"trace_id":   rc.TraceID,
			"transport":  rc.Transport,
		})
	})
	return r
}

func TestXRequestIdPassthrough(t *testing.T) {
	r := setupTracingRouter()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	req.Header.Set("X-Request-Id", "my-custom-id")
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}

	respID := w.Header().Get("X-Request-Id")
	if respID != "my-custom-id" {
		t.Errorf("expected X-Request-Id=my-custom-id, got %s", respID)
	}
}

func TestXRequestIdGenerated(t *testing.T) {
	r := setupTracingRouter()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}

	respID := w.Header().Get("X-Request-Id")
	if respID == "" {
		t.Error("expected X-Request-Id to be generated")
	}
	if len(respID) != 36 {
		t.Errorf("expected UUID length 36, got %d (%s)", len(respID), respID)
	}
}

func TestTraceParentParsing(t *testing.T) {
	r := setupTracingRouter()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	req.Header.Set("traceparent", "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}

	respTrace := w.Header().Get("X-Trace-Id")
	if respTrace != "0af7651916cd43dd8448eb211c80319c" {
		t.Errorf("expected extracted trace_id, got %s", respTrace)
	}
}

func TestTraceParentGenerated(t *testing.T) {
	r := setupTracingRouter()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}

	respTrace := w.Header().Get("X-Trace-Id")
	if respTrace == "" {
		t.Error("expected X-Trace-Id to be present")
	}
	if len(respTrace) != 32 {
		t.Errorf("expected trace_id length 32, got %d (%s)", len(respTrace), respTrace)
	}
}

func TestTransportHTTP(t *testing.T) {
	r := setupTracingRouter()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, `"transport":"http"`) {
		t.Errorf("expected transport=http, got %s", body)
	}
}

func TestTransportMCP(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(TracingMiddleware())
	r.Any("/mcp/*path", func(c *gin.Context) {
		rc := GetRequestContext(c)
		c.JSON(200, gin.H{"transport": rc.Transport})
	})
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/mcp/tools", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, `"transport":"mcp"`) {
		t.Errorf("expected transport=mcp, got %s", body)
	}
}

func TestGetRequestContextNilWithoutMiddleware(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/no-mw", func(c *gin.Context) {
		rc := GetRequestContext(c)
		if rc != nil {
			t.Error("expected nil context without middleware")
		}
		c.Status(200)
	})
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/no-mw", nil)
	r.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestInvalidTraceParent(t *testing.T) {
	tests := []struct {
		name   string
		header string
	}{
		{"empty", ""},
		{"wrong version", "01-abc123-abc123-01"},
		{"too short", "00-short"},
		{"bad trace id", "00-tooshort-b7ad6b7169203331-01"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			r := gin.New()
			r.Use(TracingMiddleware())
			r.GET("/test", func(c *gin.Context) {
				rc := GetRequestContext(c)
				c.JSON(200, gin.H{"trace_id": rc.TraceID})
			})
			w := httptest.NewRecorder()
			req, _ := http.NewRequest("GET", "/test", nil)
			if tt.header != "" {
				req.Header.Set("traceparent", tt.header)
			}
			r.ServeHTTP(w, req)
			respTrace := w.Header().Get("X-Trace-Id")
			if len(respTrace) != 32 {
				t.Errorf("expected generated trace_id length 32, got %d (%s)", len(respTrace), respTrace)
			}
		})
	}
}
