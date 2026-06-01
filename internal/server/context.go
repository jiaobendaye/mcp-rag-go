package server

import (
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// RequestContext holds per-request tracing metadata.
type RequestContext struct {
	RequestID string `json:"request_id"`
	TraceID   string `json:"trace_id"`
	Transport string `json:"transport"` // "http" or "mcp"
	Tenant    string `json:"tenant"`
}

const (
	ctxKey = "request_context"
)

// TracingMiddleware parses (or generates) tracing headers and stores a
// RequestContext in the Gin context for downstream handlers.
func TracingMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		rc := &RequestContext{}

		// --- Request ID ---------------------------------------------------
		rc.RequestID = c.GetHeader("X-Request-Id")
		if rc.RequestID == "" {
			rc.RequestID = uuid.New().String()
		}

		// --- Trace ID (W3C traceparent) -----------------------------------
		tp := c.GetHeader("traceparent")
		if tp != "" {
			rc.TraceID = parseTraceParent(tp)
		}
		if rc.TraceID == "" {
			// Generate UUID without dashes (compatible with W3C format)
			rc.TraceID = strings.ReplaceAll(uuid.New().String(), "-", "")
		}

		// --- Transport ----------------------------------------------------
		path := c.Request.URL.Path
		if strings.HasPrefix(path, "/mcp") {
			rc.Transport = "mcp"
		} else {
			rc.Transport = "http"
		}

		// --- Response headers ---------------------------------------------
		c.Header("X-Request-Id", rc.RequestID)
		c.Header("X-Trace-Id", rc.TraceID)

		// --- Store --------------------------------------------------------
		c.Set(ctxKey, rc)

		c.Next()
	}
}

// GetRequestContext retrieves the RequestContext stored by TracingMiddleware.
// Returns nil when middleware was not applied.
func GetRequestContext(c *gin.Context) *RequestContext {
	val, exists := c.Get(ctxKey)
	if !exists {
		return nil
	}
	rc, ok := val.(*RequestContext)
	if !ok {
		return nil
	}
	return rc
}

// parseTraceParent extracts the trace_id from a W3C traceparent header.
//
// Format: "00-{trace_id}-{span_id}-{flags}"
// We only care about trace_id (32 hex chars).
func parseTraceParent(header string) string {
	parts := strings.SplitN(header, "-", 4)
	if len(parts) < 3 {
		return ""
	}
	// Must start with "00"
	if parts[0] != "00" {
		return ""
	}
	traceID := parts[1]
	// Must be exactly 32 hex chars
	if len(traceID) != 32 {
		return ""
	}
	return traceID
}
