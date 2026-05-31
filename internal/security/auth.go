package security

import "crypto/subtle"

// SecurityPolicy validates API keys against global and tenant-scoped key lists.
type SecurityPolicy struct {
	Enabled        bool
	AllowAnonymous bool
	APIKeys        []string
	TenantAPIKeys  map[string][]string
}

// AuthDecision represents the result of an authentication check.
type AuthDecision struct {
	Allowed       bool
	Reason        string
	TenantKey     string
	APIKeyPresent bool
	MatchedScope  string // "disabled" | "tenant" | "global" | "anonymous" | "none"
}

// NewSecurityPolicy creates a SecurityPolicy with the given settings.
func NewSecurityPolicy(enabled, allowAnon bool, apiKeys []string, tenantKeys map[string][]string) *SecurityPolicy {
	return &SecurityPolicy{
		Enabled:        enabled,
		AllowAnonymous: allowAnon,
		APIKeys:        apiKeys,
		TenantAPIKeys:  tenantKeys,
	}
}

// Validate checks whether an API key is authorized.
func (p *SecurityPolicy) Validate(apiKey, tenantKey string) AuthDecision {
	hasKey := apiKey != ""

	if !p.Enabled {
		return AuthDecision{Allowed: true, Reason: "security disabled", TenantKey: tenantKey, APIKeyPresent: hasKey, MatchedScope: "disabled"}
	}

	// 1. Tenant-scoped keys take priority
	if keys, ok := p.TenantAPIKeys[tenantKey]; ok && tenantKey != "" {
		if hasKey && p.match(apiKey, keys) {
			return AuthDecision{Allowed: true, Reason: "tenant api key matched", TenantKey: tenantKey, APIKeyPresent: true, MatchedScope: "tenant"}
		}
		return AuthDecision{Allowed: false, Reason: "api key not permitted for tenant", TenantKey: tenantKey, APIKeyPresent: hasKey, MatchedScope: "tenant"}
	}

	// 2. Global keys
	if hasKey && p.match(apiKey, p.APIKeys) {
		return AuthDecision{Allowed: true, Reason: "global api key matched", TenantKey: tenantKey, APIKeyPresent: true, MatchedScope: "global"}
	}

	// 3. No key provided
	if !hasKey {
		if p.AllowAnonymous {
			return AuthDecision{Allowed: true, Reason: "anonymous access allowed", MatchedScope: "anonymous"}
		}
		return AuthDecision{Allowed: false, Reason: "api key required", MatchedScope: "none"}
	}

	// 4. Invalid key
	return AuthDecision{Allowed: false, Reason: "invalid api key", MatchedScope: "none"}
}

func (p *SecurityPolicy) match(provided string, candidates []string) bool {
	for _, c := range candidates {
		if subtle.ConstantTimeCompare([]byte(provided), []byte(c)) == 1 {
			return true
		}
	}
	return false
}
