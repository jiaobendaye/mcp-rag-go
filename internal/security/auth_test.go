package security

import "testing"

func TestSecurityPolicy(t *testing.T) {
	policy := NewSecurityPolicy(true, true, []string{"sk-global"}, map[string][]string{
		"tenant1": {"sk-tenant1"},
	})

	t.Run("disabled", func(t *testing.T) {
		p := NewSecurityPolicy(false, false, nil, nil)
		d := p.Validate("anykey", "")
		if !d.Allowed || d.MatchedScope != "disabled" {
			t.Errorf("expected disabled scope, got %s", d.MatchedScope)
		}
	})

	t.Run("valid global key", func(t *testing.T) {
		d := policy.Validate("sk-global", "")
		if !d.Allowed || d.MatchedScope != "global" {
			t.Errorf("expected global scope, got %s", d.MatchedScope)
		}
	})

	t.Run("invalid key", func(t *testing.T) {
		d := policy.Validate("wrong-key", "")
		if d.Allowed {
			t.Error("expected denied for invalid key")
		}
		if d.MatchedScope != "none" {
			t.Errorf("expected none scope, got %s", d.MatchedScope)
		}
	})

	t.Run("anonymous allowed", func(t *testing.T) {
		d := policy.Validate("", "")
		if !d.Allowed || d.MatchedScope != "anonymous" {
			t.Errorf("expected anonymous scope, got %s", d.MatchedScope)
		}
	})

	t.Run("anonymous denied", func(t *testing.T) {
		p := NewSecurityPolicy(true, false, nil, nil)
		d := p.Validate("", "")
		if d.Allowed {
			t.Error("expected denied for anonymous when disallowed")
		}
	})

	t.Run("tenant key valid", func(t *testing.T) {
		d := policy.Validate("sk-tenant1", "tenant1")
		if !d.Allowed || d.MatchedScope != "tenant" {
			t.Errorf("expected tenant scope, got %s", d.MatchedScope)
		}
	})

	t.Run("tenant key wrong", func(t *testing.T) {
		d := policy.Validate("wrong-key", "tenant1")
		if d.Allowed {
			t.Error("expected denied for wrong tenant key")
		}
	})

	t.Run("api key required", func(t *testing.T) {
		p := NewSecurityPolicy(true, false, nil, nil)
		p.TenantAPIKeys = map[string][]string{"t1": {"k1"}}
		d := p.Validate("", "t1")
		if d.Allowed {
			t.Error("expected denied when tenant requires key")
		}
	})
}
