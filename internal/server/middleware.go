package server

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/security"
)

// requestIDHeader is the HTTP header for request correlation.
const requestIDHeader = "X-Request-Id"

// requestIDCtxKey is the Gin context key for the request ID.
const requestIDCtxKey = "request_id"

// RequestIDMiddleware generates or propagates a request ID for every HTTP request.
// It reads X-Request-Id from the incoming request; if missing, generates a new UUID.
// The request ID is set on the response header, stored in the Gin context, and
// attached to the request context's slog logger so all downstream handlers and
// eino graph nodes can access it.
func RequestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		rid := c.GetHeader(requestIDHeader)
		if rid == "" {
			rid = uuid.NewString()
		}
		c.Header(requestIDHeader, rid)
		c.Set(requestIDCtxKey, rid)

		// Attach request_id to slog so handlers can use slog.InfoContext(ctx, ...)
		ctx := c.Request.Context()
		ctx = contextWithSlogAttrs(ctx, slog.String("request_id", rid))
		c.Request = c.Request.WithContext(ctx)

		c.Next()
	}
}

// GetRequestID extracts the request ID from the Gin context.
func GetRequestID(c *gin.Context) string {
	if rid, ok := c.Get(requestIDCtxKey); ok {
		if s, ok := rid.(string); ok {
			return s
		}
	}
	return ""
}

// slogAttrsKey is the context key for storing slog attributes.
type slogAttrsKey struct{}

// contextWithSlogAttrs attaches slog attributes to the context so they are
// automatically included by slog.InfoContext / slog.WarnContext / etc.
func contextWithSlogAttrs(ctx context.Context, attrs ...slog.Attr) context.Context {
	existing, _ := ctx.Value(slogAttrsKey{}).([]slog.Attr)
	return context.WithValue(ctx, slogAttrsKey{}, append(existing, attrs...))
}

// SlogAttrsFromContext retrieves slog attributes previously stored via
// contextWithSlogAttrs.
func SlogAttrsFromContext(ctx context.Context) []slog.Attr {
	if attrs, ok := ctx.Value(slogAttrsKey{}).([]slog.Attr); ok {
		return attrs
	}
	return nil
}

// deriveTenantKey generates a tenant key from request parameters.
// Format: u{user_id}_a{agent_id}_{collection}
// Aligns with Python TenantContext.canonical_key().
func deriveTenantKey(c *gin.Context) string {
	userID := c.Query("user_id")
	agentID := c.Query("agent_id")
	collection := c.DefaultQuery("collection", "default")

	if userID == "" {
		return collection
	}
	if agentID == "" {
		return fmt.Sprintf("u%s_%s", userID, collection)
	}
	return fmt.Sprintf("u%s_a%s_%s", userID, agentID, collection)
}

// SecurityMiddleware creates Gin middleware for auth and rate limiting.
func SecurityMiddleware(cfg *config.Config) gin.HandlerFunc {
	auth := security.NewSecurityPolicy(
		cfg.SecurityEnabled,
		cfg.SecurityAllowAnonymous,
		cfg.SecurityAPIKeys,
		cfg.SecurityTenantAPIKeys,
	)
	limiter := security.NewRateLimiter(
		cfg.RateLimitRequestsPerWindow,
		cfg.RateLimitBurst,
		float64(cfg.RateLimitWindowSeconds),
	)

	return func(c *gin.Context) {
		// Extract API key from headers or query params
		apiKey := extractAPIKey(c)

		// Derive tenant key from request parameters (was clientIP)
		tenantKey := deriveTenantKey(c)

		// Populate RequestContext.Tenant (was always empty)
		if rc := GetRequestContext(c); rc != nil {
			rc.Tenant = tenantKey
		}

		// Auth check
		decision := auth.Validate(apiKey, tenantKey)
		if !decision.Allowed {
			if decision.Reason == "api key required" {
				c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"detail": decision.Reason})
			} else {
				c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"detail": decision.Reason})
			}
			return
		}

		// Rate limit check (use tenant key as subject; fallback to clientIP when disabled)
		rateLimitSubject := tenantKey
		if tenantKey == "default" {
			rateLimitSubject = c.ClientIP()
		}
		rlDecision := limiter.Allow(rateLimitSubject)
		if !rlDecision.Allowed {
			c.Header("Retry-After", fmt.Sprintf("%.0f", rlDecision.RetryAfterSeconds))
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{"detail": "rate limit exceeded"})
			return
		}

		c.Next()
	}
}

func extractAPIKey(c *gin.Context) string {
	// x-api-key header
	if key := c.GetHeader("x-api-key"); key != "" {
		return key
	}
	// X-API-Key header (case insensitive)
	if key := c.GetHeader("X-API-Key"); key != "" {
		return key
	}
	// Authorization: Bearer <token>
	if auth := c.GetHeader("Authorization"); auth != "" {
		if len(auth) > 7 && auth[:7] == "Bearer " {
			return auth[7:]
		}
	}
	// Query param
	if key := c.Query("api_key"); key != "" {
		return key
	}
	return ""
}

func clientIP(c *gin.Context) string {
	return c.ClientIP()
}

// SlogAccessLogger returns a Gin middleware that writes one JSON log line
// per request using slog. It records method, path, status, latency_ms,
// request_id, and client_ip.
func SlogAccessLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		latencyMs := float64(time.Since(start).Microseconds()) / 1000.0

		attrs := []any{
			"method", c.Request.Method,
			"path", c.Request.URL.Path,
			"status", c.Writer.Status(),
			"latency_ms", latencyMs,
			"request_id", GetRequestID(c),
			"client_ip", c.ClientIP(),
		}

		status := c.Writer.Status()
		if status >= 500 {
			slog.Error("request", attrs...)
		} else if status >= 400 {
			slog.Warn("request", attrs...)
		} else {
			slog.Info("request", attrs...)
		}
	}
}
