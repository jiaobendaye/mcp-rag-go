package embedder

import (
	"context"
	"sync/atomic"

	"github.com/cloudwego/eino/components/embedding"
)

// SwappableEmbedder wraps an embedding.Embedder and supports atomic
// hot-swap of the underlying embedder implementation at runtime.
// The hot path (EmbedStrings) uses a single atomic.Load, no locks.
type SwappableEmbedder struct {
	current atomic.Pointer[embedding.Embedder]
}

// NewSwappableEmbedder creates a SwappableEmbedder with the given initial embedder.
func NewSwappableEmbedder(e embedding.Embedder) *SwappableEmbedder {
	s := &SwappableEmbedder{}
	s.current.Store(&e)
	return s
}

// EmbedStrings delegates to the current embedder via atomic load.
func (s *SwappableEmbedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
	e := s.current.Load()
	return (*e).EmbedStrings(ctx, texts, opts...)
}

// Swap atomically replaces the current embedder.
func (s *SwappableEmbedder) Swap(e embedding.Embedder) {
	s.current.Store(&e)
}
