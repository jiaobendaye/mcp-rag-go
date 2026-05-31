package security

import "testing"

func TestRateLimiter(t *testing.T) {
	t.Run("disabled when limit is 0", func(t *testing.T) {
		l := NewRateLimiter(0, 0, 60)
		d := l.Allow("test")
		if !d.Allowed {
			t.Error("expected allowed when limit=0")
		}
	})

	t.Run("allows with burst", func(t *testing.T) {
		l := NewRateLimiter(10, 5, 60) // 10+5=15 capacity
		for i := 0; i < 15; i++ {
			d := l.Allow("test")
			if !d.Allowed {
				t.Errorf("expected allowed at request %d", i+1)
			}
		}
		// 16th should be denied
		d := l.Allow("test")
		if d.Allowed {
			t.Error("expected denied after exceeding capacity")
		}
		if d.RetryAfterSeconds <= 0 {
			t.Error("expected positive retry_after")
		}
	})

	t.Run("different subjects independent", func(t *testing.T) {
		l := NewRateLimiter(1, 0, 60)
		// Exhaust subject A
		l.Allow("a")
		// Subject B should still work
		d := l.Allow("b")
		if !d.Allowed {
			t.Error("expected allowed for different subject")
		}
	})

	t.Run("negative values use defaults", func(t *testing.T) {
		l := NewRateLimiter(-1, -1, -1)
		d := l.Allow("test")
		if !d.Allowed {
			t.Error("expected allowed for negative values (treats limit=0)")
		}
	})
}
