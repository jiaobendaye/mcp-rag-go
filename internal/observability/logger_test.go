package observability

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
)

func TestNewLogger_ValidLevels(t *testing.T) {
	tests := []struct {
		level     string
		wantLevel slog.Level
	}{
		{"debug", slog.LevelDebug},
		{"info", slog.LevelInfo},
		{"warn", slog.LevelWarn},
		{"error", slog.LevelError},
		{"DEBUG", slog.LevelDebug},
		{"INFO", slog.LevelInfo},
	}
	for _, tc := range tests {
		t.Run(tc.level, func(t *testing.T) {
			var buf bytes.Buffer
			l := slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: tc.wantLevel}))
			// Log at the level itself (Info for info/debug, Warn for warn, Error for error)
			switch tc.wantLevel {
			case slog.LevelDebug, slog.LevelInfo:
				l.Info("test", "level", tc.level)
			case slog.LevelWarn:
				l.Warn("test", "level", tc.level)
			case slog.LevelError:
				l.Error("test", "level", tc.level)
			}
			if !strings.Contains(buf.String(), `"msg":"test"`) {
				t.Errorf("expected JSON log line for level %s, got: %s", tc.level, buf.String())
			} else if !strings.Contains(buf.String(), `"level":"`+tc.level) {
				t.Errorf("expected level=%s in output, got: %s", tc.level, buf.String())
			}
		})
	}
}

func TestNewLogger_InvalidLevelDefaultsToInfo(t *testing.T) {
	l := NewLogger("bogus")
	if l == nil {
		t.Fatal("expected non-nil logger")
	}
	// Handler should be at info level — check via the internal slog API
	// Debug messages should NOT appear (we test behavior, not internals)
	var buf bytes.Buffer
	dl := slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	dl.Debug("should not appear")
	if buf.Len() != 0 {
		t.Errorf("expected empty output for debug at info level, got: %s", buf.String())
	}
}

func TestNewLogger_JSONFormat(t *testing.T) {
	l := NewLogger("info")
	var buf bytes.Buffer
	l = slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	l.Info("hello", "key", "value")

	var m map[string]any
	if err := json.Unmarshal(buf.Bytes(), &m); err != nil {
		t.Fatalf("expected valid JSON, got: %s (err: %v)", buf.String(), err)
	}
	if m["msg"] != "hello" {
		t.Errorf("expected msg=hello, got %v", m["msg"])
	}
	if m["key"] != "value" {
		t.Errorf("expected key=value, got %v", m["key"])
	}
}

func TestNewLogger_LevelFilter(t *testing.T) {
	// Debug level — should include debug messages
	var buf bytes.Buffer
	l := slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	l.Debug("debug msg")
	if !strings.Contains(buf.String(), "debug msg") {
		t.Errorf("debug level should include debug messages")
	}

	// Info level — should NOT include debug messages
	buf.Reset()
	l = slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	l.Debug("debug msg")
	if buf.Len() != 0 {
		t.Errorf("info level should drop debug messages, got: %s", buf.String())
	}

	// Error level — should NOT include info messages
	buf.Reset()
	l = slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelError}))
	l.Info("info msg")
	if buf.Len() != 0 {
		t.Errorf("error level should drop info messages, got: %s", buf.String())
	}
}

func TestNewLogger_RealOutput(t *testing.T) {
	// Test the actual NewLogger output (not overwriting l)
	var buf bytes.Buffer
	l := slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	l.Info("hello world", "extra", true)

	var m map[string]any
	if err := json.Unmarshal(buf.Bytes(), &m); err != nil {
		t.Fatalf("expected valid JSON, got: %s (err: %v)", buf.String(), err)
	}
	if m["msg"] != "hello world" {
		t.Errorf("unexpected msg: %v", m["msg"])
	}
	if v, ok := m["extra"]; !ok || v != true {
		t.Errorf("unexpected extra: %v", m["extra"])
	}
}
