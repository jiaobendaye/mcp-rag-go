package security

// AuthError represents authentication failure (401).
type AuthError struct{ Msg string }

func (e *AuthError) Error() string { return e.Msg }

// ForbiddenError represents authorization failure (403).
type ForbiddenError struct{ Msg string }

func (e *ForbiddenError) Error() string { return e.Msg }

// RateLimitError represents rate limit exceeded (429).
type RateLimitError struct{ Msg string }

func (e *RateLimitError) Error() string { return e.Msg }

// QuotaError represents quota exceeded (413).
type QuotaError struct{ Msg string }

func (e *QuotaError) Error() string { return e.Msg }
