// Package observability provides structured metrics collection, health checks, and readiness probing.
package observability

import (
	"sort"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// MetricsConfig holds thresholds for health evaluation.
type MetricsConfig struct {
	WarningErrorRate  float64 // default 0.05
	CriticalErrorRate float64 // default 0.20
	SlowRequestMs     int64   // default 1000
}

// DefaultMetricsConfig returns sensible defaults.
func DefaultMetricsConfig() MetricsConfig {
	return MetricsConfig{
		WarningErrorRate:  0.05,
		CriticalErrorRate: 0.20,
		SlowRequestMs:     1000,
	}
}

// HealthStatus represents the three-tier service health.
type HealthStatus string

const (
	HealthHealthy   HealthStatus = "healthy"
	HealthDegraded  HealthStatus = "degraded"
	HealthUnhealthy HealthStatus = "unhealthy"
)

// ---------------------------------------------------------------------------
// RollingSample – ring buffer for percentile calculation
// ---------------------------------------------------------------------------

const maxSamples = 512

// RollingSample is a fixed-capacity ring buffer used to approximate
// P50/P95/P99 latency percentiles.
type RollingSample struct {
	buffer []int64
	size   int
	pos    int
}

// newRollingSample creates an empty rolling sample buffer.
func newRollingSample() *RollingSample {
	return &RollingSample{
		buffer: make([]int64, maxSamples),
	}
}

// add appends a latency value, overwriting the oldest entry when full.
func (rs *RollingSample) add(v int64) {
	rs.buffer[rs.pos] = v
	rs.pos = (rs.pos + 1) % maxSamples
	if rs.size < maxSamples {
		rs.size++
	}
}

// snapshot returns a sorted copy of the sample buffer.
func (rs *RollingSample) snapshot() []int64 {
	if rs.size == 0 {
		return nil
	}
	out := make([]int64, rs.size)
	copy(out, rs.buffer[:rs.size])
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// percentile returns the approximate p-th percentile from the sample.
func (rs *RollingSample) percentile(p float64) float64 {
	sorted := rs.snapshot()
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)) * p)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return float64(sorted[idx])
}

// ---------------------------------------------------------------------------
// OperationStats – per-operation / per-provider metrics
// ---------------------------------------------------------------------------

// OperationStats tracks aggregated statistics for a named operation or provider.
type OperationStats struct {
	Count          int64          `json:"count"`
	ErrorCount     int64          `json:"error_count"`
	TotalLatencyMs int64          `json:"total_latency_ms"`
	MinMs          int64          `json:"min_ms"`
	MaxMs          int64          `json:"max_ms"`
	SlowCount      int64          `json:"slow_count"`
	samples        *RollingSample
}

// newOperationStats allocates a fresh OperationStats.
func newOperationStats() *OperationStats {
	return &OperationStats{
		MinMs:   -1,
		samples: newRollingSample(),
	}
}

func (os *OperationStats) record(latencyMs int64, isError bool, slowThresholdMs int64) {
	os.Count++
	os.TotalLatencyMs += latencyMs
	if isError {
		os.ErrorCount++
	}
	if latencyMs >= slowThresholdMs {
		os.SlowCount++
	}
	if os.MinMs == -1 || latencyMs < os.MinMs {
		os.MinMs = latencyMs
	}
	if latencyMs > os.MaxMs {
		os.MaxMs = latencyMs
	}
	os.samples.add(latencyMs)
}

// snapshot returns a JSON-serialisable view of this stat.
func (os *OperationStats) snapshot() OperationStatsView {
	avg := float64(0)
	if os.Count > 0 {
		avg = float64(os.TotalLatencyMs) / float64(os.Count)
	}
	min := float64(0)
	max := float64(0)
	if os.Count > 0 {
		min = float64(os.MinMs)
		max = float64(os.MaxMs)
	}
	return OperationStatsView{
		Count:        os.Count,
		ErrorCount:   os.ErrorCount,
		AvgMs:        avg,
		MinMs:        min,
		MaxMs:        max,
		P50:          os.samples.percentile(0.50),
		P95:          os.samples.percentile(0.95),
		P99:          os.samples.percentile(0.99),
		SlowCount:    os.SlowCount,
		TotalLatencyMs: os.TotalLatencyMs,
	}
}

// OperationStatsView is the JSON representation of OperationStats.
type OperationStatsView struct {
	Count          int64   `json:"count"`
	ErrorCount     int64   `json:"error_count"`
	AvgMs          float64 `json:"avg_ms"`
	MinMs          float64 `json:"min_ms"`
	MaxMs          float64 `json:"max_ms"`
	P50            float64 `json:"p50"`
	P95            float64 `json:"p95"`
	P99            float64 `json:"p99"`
	SlowCount      int64   `json:"slow_count"`
	TotalLatencyMs int64   `json:"total_latency_ms"`
}

// ---------------------------------------------------------------------------
// MetricsCollector
// ---------------------------------------------------------------------------

// MetricsCollector aggregates per-operation and per-provider timing data.
type MetricsCollector struct {
	mu        sync.RWMutex
	opStats   map[string]*OperationStats // e.g. "search", "chat", "embed", "index"
	provStats map[string]*OperationStats // e.g. "ollama.encode", "openai.generate"
	startTime time.Time
	config    MetricsConfig
}

// NewMetricsCollector creates a ready-to-use MetricsCollector.
func NewMetricsCollector(cfg MetricsConfig) *MetricsCollector {
	return &MetricsCollector{
		opStats:   make(map[string]*OperationStats),
		provStats: make(map[string]*OperationStats),
		startTime: time.Now(),
		config:    cfg,
	}
}

// Record registers a completed operation with its latency and error flag.
//
//	latencyMs: operation wall-clock time in milliseconds
//	isError:   true when the operation returned an error
func (mc *MetricsCollector) Record(operation string, latencyMs int64, isError bool) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	ops, ok := mc.opStats[operation]
	if !ok {
		ops = newOperationStats()
		mc.opStats[operation] = ops
	}
	ops.record(latencyMs, isError, mc.config.SlowRequestMs)
}

// RecordProvider registers a completed provider call.
func (mc *MetricsCollector) RecordProvider(provider string, latencyMs int64, isError bool) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	ps, ok := mc.provStats[provider]
	if !ok {
		ps = newOperationStats()
		mc.provStats[provider] = ps
	}
	ps.record(latencyMs, isError, mc.config.SlowRequestMs)
}

// MetricsSnapshot is the full metrics payload returned by /metrics.
type MetricsSnapshot struct {
	Operations    map[string]OperationStatsView `json:"operations"`
	Providers     map[string]OperationStatsView `json:"providers"`
	TotalRequests int64                         `json:"total_requests"`
	ErrorCount    int64                         `json:"error_count"`
	UptimeSeconds float64                       `json:"uptime_seconds"`
}

// Snapshot returns an immutable snapshot of all collected metrics.
func (mc *MetricsCollector) Snapshot() MetricsSnapshot {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	snap := MetricsSnapshot{
		Operations:    make(map[string]OperationStatsView, len(mc.opStats)),
		Providers:     make(map[string]OperationStatsView, len(mc.provStats)),
		UptimeSeconds: time.Since(mc.startTime).Seconds(),
	}

	for name, os := range mc.opStats {
		v := os.snapshot()
		snap.Operations[name] = v
		snap.TotalRequests += v.Count
		snap.ErrorCount += v.ErrorCount
	}
	for name, ps := range mc.provStats {
		snap.Providers[name] = ps.snapshot()
	}

	return snap
}

// Health evaluates the current error rate and slow-request ratio against
// configured thresholds and returns a three-tier status.
func (mc *MetricsCollector) Health() HealthStatus {
	snap := mc.Snapshot()

	if snap.TotalRequests == 0 {
		return HealthHealthy
	}

	errRate := float64(snap.ErrorCount) / float64(snap.TotalRequests)

	if errRate > mc.config.CriticalErrorRate {
		return HealthUnhealthy
	}
	if errRate > mc.config.WarningErrorRate {
		return HealthDegraded
	}

	// Check slow requests
	for _, op := range snap.Operations {
		if op.SlowCount > 0 {
			return HealthDegraded
		}
	}

	return HealthHealthy
}

// HealthDetail provides structured health information for the /health endpoint.
type HealthDetail struct {
	Status        HealthStatus                      `json:"status"`
	UptimeSeconds float64                           `json:"uptime_seconds"`
	TotalRequests int64                             `json:"total_requests"`
	ErrorCount    int64                             `json:"error_count"`
	Operations    map[string]OperationStatsView     `json:"operations"`
}

// HealthDetail returns structured health info for the /health endpoint.
func (mc *MetricsCollector) HealthDetail() HealthDetail {
	snap := mc.Snapshot()
	return HealthDetail{
		Status:        mc.Health(),
		UptimeSeconds: snap.UptimeSeconds,
		TotalRequests: snap.TotalRequests,
		ErrorCount:    snap.ErrorCount,
		Operations:    snap.Operations,
	}
}
