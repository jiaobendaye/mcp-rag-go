package server

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/security"
)

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

		// Auth check
		decision := auth.Validate(apiKey, c.ClientIP())
		if !decision.Allowed {
			if decision.Reason == "api key required" {
				c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"detail": decision.Reason})
			} else {
				c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"detail": decision.Reason})
			}
			return
		}

		// Rate limit check
		rlDecision := limiter.Allow(c.ClientIP())
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
