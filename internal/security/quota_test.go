package security

import "testing"

func TestUploadQuota(t *testing.T) {
	p := NewUploadQuotaPolicy(10, 1000, 100)

	t.Run("within quota", func(t *testing.T) {
		d := p.Check([]int64{50, 30})
		if !d.Allowed {
			t.Errorf("expected allowed, got: %s", d.Reason)
		}
	})

	t.Run("exceeds file count", func(t *testing.T) {
		sizes := make([]int64, 11)
		d := p.Check(sizes)
		if d.Allowed {
			t.Error("expected denied for too many files")
		}
	})

	t.Run("exceeds total bytes", func(t *testing.T) {
		d := p.Check([]int64{600, 500})
		if d.Allowed {
			t.Error("expected denied for batch too large")
		}
	})

	t.Run("exceeds file size", func(t *testing.T) {
		d := p.Check([]int64{150})
		if d.Allowed {
			t.Error("expected denied for file too large")
		}
	})
}

func TestIndexQuota(t *testing.T) {
	p := NewIndexQuotaPolicy(100, 500, 10000)

	t.Run("within quota", func(t *testing.T) {
		d := p.Check(50, 200, 5000)
		if !d.Allowed {
			t.Errorf("expected allowed, got: %s", d.Reason)
		}
	})

	t.Run("exceeds documents", func(t *testing.T) {
		d := p.Check(101, 200, 5000)
		if d.Allowed {
			t.Error("expected denied for too many documents")
		}
	})

	t.Run("exceeds chunks", func(t *testing.T) {
		d := p.Check(50, 501, 5000)
		if d.Allowed {
			t.Error("expected denied for too many chunks")
		}
	})

	t.Run("exceeds chars", func(t *testing.T) {
		d := p.Check(50, 200, 99999)
		if d.Allowed {
			t.Error("expected denied for too many characters")
		}
	})
}
