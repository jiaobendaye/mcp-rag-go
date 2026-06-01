package rag

import (
	"context"
	"testing"

	"github.com/cloudwego/eino/components/indexer"
	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
)

// makeKBIndexerForTest builds a KBIndexer without going through
// NewKBIndexer, so we don't need a live ES client. The conf.Index
// field is set explicitly; the `base` is nil because the tests
// short-circuit before reaching it.
func makeKBIndexerForTest(boundIndex string) *KBIndexer {
	return &KBIndexer{
		base: nil,
		conf: &elastic_indexer.IndexerConfig{Index: boundIndex},
	}
}

func TestKBIndexer_StoreWithoutWithIndex_Errors(t *testing.T) {
	k := makeKBIndexerForTest(PlaceholderIndex)
	// No WithIndex, bound index is placeholder → must error.
	_, err := k.Store(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error when WithIndex is missing and config.Index is placeholder")
	}
}

func TestKBIndexer_StoreWithWithIndex_RoutesToIndex(t *testing.T) {
	k := makeKBIndexerForTest(PlaceholderIndex)
	// WithIndex overrides placeholder. We can't actually call .base.Store
	// (base is nil in this test), but resolveIndex will return the
	// WithIndex value, proving the routing logic. The actual ES call
	// would happen on the second branch (slow path) which would rebuild
	// the indexer — covered by integration tests.
	idx := k.resolveIndex([]indexer.Option{WithIndex("kb_42")})
	if idx != "kb_42" {
		t.Errorf("expected resolveIndex to return kb_42, got %q", idx)
	}
}

func TestKBIndexer_ResolveIndex_DefaultsToConfigIndex(t *testing.T) {
	k := makeKBIndexerForTest("kb_3")
	if got := k.resolveIndex(nil); got != "kb_3" {
		t.Errorf("expected fallback to config.Index (kb_3), got %q", got)
	}
}

func TestKBIndexer_ResolveIndex_PerCallOverridesConfig(t *testing.T) {
	k := makeKBIndexerForTest("kb_3")
	got := k.resolveIndex([]indexer.Option{WithIndex("kb_99")})
	if got != "kb_99" {
		t.Errorf("expected override kb_99, got %q", got)
	}
}

func TestKBIndexer_ResolveIndex_EmptyPerCallFallsBack(t *testing.T) {
	k := makeKBIndexerForTest("kb_5")
	// WithIndex("") is a no-op and falls back to the bound config.
	got := k.resolveIndex([]indexer.Option{WithIndex("")})
	if got != "kb_5" {
		t.Errorf("expected fallback to bound config (kb_5) when WithIndex(\"\"), got %q", got)
	}
}
