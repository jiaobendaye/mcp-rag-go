package server

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/jiaobendaye/mcp-rag-go/internal/knowledgebase"
)

// IdempotencyMiddleware returns a Gin middleware that handles Idempotency-Key.
//
// Semantics (Stripe-compatible):
//   - No header: pass through.
//   - Cache hit, same hash: return cached response with Idempotency-Replay: true.
//   - Cache hit, different hash: return 422 (key reuse with different payload).
//   - Cache miss: process normally, capture response, cache for 24h.
func IdempotencyMiddleware(store *knowledgebase.Store) gin.HandlerFunc {
	return func(c *gin.Context) {
		key := c.GetHeader("Idempotency-Key")
		if key == "" {
			c.Next()
			return
		}

		method := c.Request.Method
		path := c.Request.URL.Path

		// Read and hash the request body
		bodyBytes, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"detail": "failed to read request body"})
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		requestHash := fmt.Sprintf("%x", sha256.Sum256(bodyBytes))

		// Check cache
		record, err := store.GetIdempotencyRecord(key, method, path)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"detail": "idempotency lookup failed"})
			return
		}

		if record != nil {
			if record.RequestHash == requestHash {
				// Cache hit — replay
				c.Header("Idempotency-Replay", "true")
				c.Data(record.ResponseStatus, "application/json", record.ResponseBody)
				c.Abort()
				return
			}
			// Same key, different payload
			c.AbortWithStatusJSON(http.StatusUnprocessableEntity, gin.H{
				"detail": "idempotency key reused with different request body",
			})
			return
		}

		// Cache miss — capture the response
		w := &responseCapture{ResponseWriter: c.Writer, buf: &bytes.Buffer{}}
		c.Writer = w
		c.Next()

		// Only cache successful responses (2xx)
		if c.Writer.Status() >= 200 && c.Writer.Status() < 300 {
			headers := make(map[string]string)
			for k, v := range c.Writer.Header() {
				if len(v) > 0 {
					headers[k] = v[0]
				}
			}
			// Replay is a response-only header; don't cache it
			delete(headers, "Idempotency-Replay")

			store.SetIdempotencyRecord(key, method, path, requestHash,
				w.Status(), w.buf.Bytes(), headers)
		}
	}
}

// responseCapture is a Gin ResponseWriter that captures the response body.
type responseCapture struct {
	gin.ResponseWriter
	buf    *bytes.Buffer
	status int
}

func (w *responseCapture) Write(b []byte) (int, error) {
	w.buf.Write(b)
	return w.ResponseWriter.Write(b)
}

func (w *responseCapture) WriteString(s string) (int, error) {
	w.buf.WriteString(s)
	return w.ResponseWriter.WriteString(s)
}

// idempotentPaths returns the paths that support idempotency keys.
var idempotentPaths = map[string]bool{
	"/add-document":    true,
	"/upload-files":    true,
	"/knowledge-bases": true,
}

// IdempotencyRouteFilter returns a middleware that only applies idempotency
// to write-API endpoints.
func IdempotencyRouteFilter(store *knowledgebase.Store) gin.HandlerFunc {
	handler := IdempotencyMiddleware(store)
	return func(c *gin.Context) {
		path := c.Request.URL.Path
		// Support both exact and sub-path matches
		match := idempotentPaths[path]
		if !match {
			for p := range idempotentPaths {
				if strings.HasPrefix(path, p) {
					match = true
					break
				}
			}
		}
		if !match {
			c.Next()
			return
		}
		handler(c)
	}
}
