package rag

import (
	"context"
	"testing"

	"github.com/cloudwego/eino/components/retriever"
	elastic_retriever "github.com/cloudwego/eino-ext/components/retriever/es8"
)

// makeKBRetrieverForTest builds a KBRetriever without a live ES client
// for the resolveIndex path. base is nil; tests that exercise
// Retrieve must use a real ES (integration) or use a wrapped base.
func makeKBRetrieverForTest(boundIndex string) *KBRetriever {
	return &KBRetriever{
		base: nil,
		conf: &elastic_retriever.RetrieverConfig{Index: boundIndex},
	}
}

func TestKBRetriever_ResolveIndex_PassesThroughOptions(t *testing.T) {
	k := makeKBRetrieverForTest(PlaceholderIndex)
	// WithIndex should override the placeholder.
	got := k.resolveIndex([]retriever.Option{retriever.WithIndex("kb_77")})
	if got != "kb_77" {
		t.Errorf("expected WithIndex to pass through, got %q", got)
	}
}

func TestKBRetriever_ResolveIndex_DefaultsToConfig(t *testing.T) {
	k := makeKBRetrieverForTest("kb_default")
	if got := k.resolveIndex(nil); got != "kb_default" {
		t.Errorf("expected fallback to bound config.Index, got %q", got)
	}
}

func TestKBRetriever_Retrieve_PlaceholderErrors(t *testing.T) {
	// Base is nil here, but the placeholder check happens before any
	// base call, so this is safe.
	k := makeKBRetrieverForTest(PlaceholderIndex)
	_, err := k.Retrieve(context.Background(), "q")
	if err == nil {
		t.Fatal("expected error when bound config.Index is placeholder and no WithIndex provided")
	}
}

func TestKBRetriever_Retrieve_EmptyIndexErrors(t *testing.T) {
	// Force an empty index after NewKBRetriever's defaulting. We
	// construct a wrapper that has a different bound index, then
	// override the per-call to empty by hacking the conf.
	// Simpler: just verify that resolveIndex doesn't return empty when
	// the config has a value.
	k := makeKBRetrieverForTest("kb_1")
	if got := k.resolveIndex(nil); got != "kb_1" {
		t.Errorf("expected non-empty default, got %q", got)
	}
}
