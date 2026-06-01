package observability

import (
	"testing"
)

func TestRecordAndSnapshot(t *testing.T) {
	mc := NewMetricsCollector(DefaultMetricsConfig())

	// Record some operations
	mc.Record("search", 45, false)
	mc.Record("search", 120, false)
	mc.Record("search", 2000, true) // slow + error
	mc.Record("chat", 300, false)

	snap := mc.Snapshot()

	if snap.TotalRequests != 4 {
		t.Errorf("expected 4 total requests, got %d", snap.TotalRequests)
	}
	if snap.ErrorCount != 1 {
		t.Errorf("expected 1 error, got %d", snap.ErrorCount)
	}

	search, ok := snap.Operations["search"]
	if !ok {
		t.Fatal("expected search operation stats")
	}
	if search.Count != 3 {
		t.Errorf("expected search count=3, got %d", search.Count)
	}
	if search.ErrorCount != 1 {
		t.Errorf("expected search error_count=1, got %d", search.ErrorCount)
	}
	if search.MinMs != 45 {
		t.Errorf("expected search min_ms=45, got %f", search.MinMs)
	}
	if search.MaxMs != 2000 {
		t.Errorf("expected search max_ms=2000, got %f", search.MaxMs)
	}
	expectedAvg := float64(45+120+2000) / 3.0
	if search.AvgMs != expectedAvg {
		t.Errorf("expected search avg_ms=%f, got %f", expectedAvg, search.AvgMs)
	}
	if search.SlowCount != 1 {
		t.Errorf("expected search slow_count=1, got %d", search.SlowCount)
	}
}

func TestPercentiles(t *testing.T) {
	mc := NewMetricsCollector(DefaultMetricsConfig())

	// Feed 100 samples with values 1..100
	for i := int64(1); i <= 100; i++ {
		mc.Record("search", i, false)
	}

	snap := mc.Snapshot()
	search := snap.Operations["search"]

	// P50 should be near 50
	if search.P50 < 45 || search.P50 > 55 {
		t.Errorf("expected P50 ~50, got %f", search.P50)
	}
	// P95 should be near 95
	if search.P95 < 90 || search.P95 > 99 {
		t.Errorf("expected P95 ~95, got %f", search.P95)
	}
	// P99 should be near 99
	if search.P99 < 95 {
		t.Errorf("expected P99 ~99, got %f", search.P99)
	}
}

func TestPercentileEmpty(t *testing.T) {
	mc := NewMetricsCollector(DefaultMetricsConfig())
	snap := mc.Snapshot()
	// No operations recorded - should not panic
	if snap.TotalRequests != 0 {
		t.Errorf("expected 0 total requests")
	}
}

func TestPercentileSingle(t *testing.T) {
	mc := NewMetricsCollector(DefaultMetricsConfig())
	mc.Record("test", 42, false)

	snap := mc.Snapshot()
	op := snap.Operations["test"]
	if op.P50 != 42 || op.P95 != 42 || op.P99 != 42 {
		t.Errorf("expected all percentiles=42 for single sample, got P50=%f P95=%f P99=%f", op.P50, op.P95, op.P99)
	}
}

func TestRecordProvider(t *testing.T) {
	mc := NewMetricsCollector(DefaultMetricsConfig())
	mc.RecordProvider("ollama.encode", 50, false)
	mc.RecordProvider("ollama.encode", 60, true)
	mc.RecordProvider("openai.generate", 200, false)

	snap := mc.Snapshot()

	ollama, ok := snap.Providers["ollama.encode"]
	if !ok {
		t.Fatal("expected ollama.encode provider stats")
	}
	if ollama.Count != 2 {
		t.Errorf("expected ollama count=2, got %d", ollama.Count)
	}
	if ollama.ErrorCount != 1 {
		t.Errorf("expected ollama error_count=1, got %d", ollama.ErrorCount)
	}

	openai, ok := snap.Providers["openai.generate"]
	if !ok {
		t.Fatal("expected openai.generate provider stats")
	}
	if openai.Count != 1 {
		t.Errorf("expected openai count=1, got %d", openai.Count)
	}
}

func TestHealthHealthy(t *testing.T) {
	mc := NewMetricsCollector(DefaultMetricsConfig())
	// No errors, no slow requests
	for i := 0; i < 100; i++ {
		mc.Record("search", 45, false)
	}

	status := mc.Health()
	if status != HealthHealthy {
		t.Errorf("expected healthy, got %s", status)
	}
}

func TestHealthDegradedErrors(t *testing.T) {
	mc := NewMetricsCollector(DefaultMetricsConfig())
	// 10% error rate > 5% warning threshold
	for i := 0; i < 90; i++ {
		mc.Record("search", 45, false)
	}
	for i := 0; i < 10; i++ {
		mc.Record("search", 45, true)
	}

	status := mc.Health()
	if status != HealthDegraded {
		t.Errorf("expected degraded, got %s", status)
	}
}

func TestHealthDegradedSlow(t *testing.T) {
	mc := NewMetricsCollector(DefaultMetricsConfig())
	// No errors, but one slow request
	mc.Record("search", 45, false)
	mc.Record("search", 2000, false) // > 1000ms threshold

	status := mc.Health()
	if status != HealthDegraded {
		t.Errorf("expected degraded due to slow requests, got %s", status)
	}
}

func TestHealthUnhealthy(t *testing.T) {
	mc := NewMetricsCollector(DefaultMetricsConfig())
	// 30% error rate > 20% critical threshold
	for i := 0; i < 70; i++ {
		mc.Record("search", 45, false)
	}
	for i := 0; i < 30; i++ {
		mc.Record("search", 45, true)
	}

	status := mc.Health()
	if status != HealthUnhealthy {
		t.Errorf("expected unhealthy, got %s", status)
	}
}

func TestHealthEmpty(t *testing.T) {
	mc := NewMetricsCollector(DefaultMetricsConfig())
	// No operations recorded - should be healthy
	status := mc.Health()
	if status != HealthHealthy {
		t.Errorf("expected healthy for empty collector, got %s", status)
	}
}

func TestHealthDetail(t *testing.T) {
	mc := NewMetricsCollector(DefaultMetricsConfig())
	mc.Record("search", 45, false)

	hd := mc.HealthDetail()
	if hd.Status != HealthHealthy {
		t.Errorf("expected healthy status, got %s", hd.Status)
	}
	if hd.TotalRequests != 1 {
		t.Errorf("expected 1 total request, got %d", hd.TotalRequests)
	}
	if hd.UptimeSeconds <= 0 {
		t.Error("expected positive uptime_seconds")
	}
	if _, ok := hd.Operations["search"]; !ok {
		t.Error("expected search in operations")
	}
}
