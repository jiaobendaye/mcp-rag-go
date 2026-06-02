package server

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/security"
)

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
