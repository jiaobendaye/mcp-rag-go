package observability

import (
	"log/slog"
	"os"
	"strings"
)

// NewLogger creates a JSON-slog Logger at the requested level.
// Invalid level strings default to "info" with a warning written to stderr.
func NewLogger(level string) *slog.Logger {
	var l slog.Level
	switch strings.ToLower(level) {
	case "debug":
		l = slog.LevelDebug
	case "info":
		l = slog.LevelInfo
	case "warn":
		l = slog.LevelWarn
	case "error":
		l = slog.LevelError
	default:
		l = slog.LevelInfo
		// Log a one-time warning; use os.Stderr because the global logger
		// has not been set up yet at this point.
		os.Stderr.WriteString("observability: invalid log_level \"" + level + "\", defaulting to info\n")
	}
	return slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: l}))
}
