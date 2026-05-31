package security

import (
	"math"
	"sync"
	"time"
)

type bucket struct {
	tokens    float64
	updatedAt time.Time
}

// TokenBucketRateLimiter implements in-memory token bucket rate limiting.
type TokenBucketRateLimiter struct {
	mu            sync.Mutex
	limit         int
	windowSeconds float64
	burst         int
	ratePerSecond float64
	capacity      float64
	buckets       map[string]*bucket
}

// RateLimitDecision represents the result of a rate limit check.
type RateLimitDecision struct {
	Subject           string
	Allowed           bool
	Limit             int
	WindowSeconds     float64
	Remaining         float64
	RetryAfterSeconds float64
}

// NewRateLimiter creates a token bucket rate limiter. limit=0 disables limiting.
func NewRateLimiter(limit, burst int, windowSeconds float64) *TokenBucketRateLimiter {
	if limit < 0 {
		limit = 0
	}
	if burst < 0 {
		burst = 0
	}
	if windowSeconds <= 0 {
		windowSeconds = 60
	}
	return &TokenBucketRateLimiter{
		limit:         limit,
		windowSeconds: windowSeconds,
		burst:         burst,
		ratePerSecond: float64(limit) / windowSeconds,
		capacity:      float64(limit + burst),
		buckets:       make(map[string]*bucket),
	}
}

// Allow checks whether a request from the given subject is allowed.
func (l *TokenBucketRateLimiter) Allow(subject string) RateLimitDecision {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.limit <= 0 {
		return RateLimitDecision{Subject: subject, Allowed: true}
	}

	now := time.Now()
	b, ok := l.buckets[subject]
	if !ok {
		b = &bucket{tokens: l.capacity, updatedAt: now}
		l.buckets[subject] = b
	} else {
		elapsed := now.Sub(b.updatedAt).Seconds()
		b.tokens = math.Min(l.capacity, b.tokens+elapsed*l.ratePerSecond)
		b.updatedAt = now
	}

	if b.tokens >= 1 {
		b.tokens--
		return RateLimitDecision{
			Subject:       subject,
			Allowed:       true,
			Limit:         l.limit,
			WindowSeconds: l.windowSeconds,
			Remaining:     b.tokens,
		}
	}

	missing := 1 - b.tokens
	retryAfter := missing / l.ratePerSecond
	if l.ratePerSecond <= 0 {
		retryAfter = l.windowSeconds
	}
	return RateLimitDecision{
		Subject:           subject,
		Allowed:           false,
		Limit:             l.limit,
		WindowSeconds:     l.windowSeconds,
		Remaining:         b.tokens,
		RetryAfterSeconds: retryAfter,
	}
}
