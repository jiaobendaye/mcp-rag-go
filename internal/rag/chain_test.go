package rag

import (
	"context"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/indexer"
	"github.com/cloudwego/eino-ext/components/document/transformer/splitter/recursive"

	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
)

func TestBuildIndexChainAt_RejectsEmptyIndexName(t *testing.T) {
	splitter, err := recursive.NewSplitter(context.Background(), &recursive.Config{
		ChunkSize: 100, OverlapSize: 10,
	})
	if err != nil {
		t.Fatalf("create splitter: %v", err)
	}
	k := &KBIndexer{base: nil, conf: &elastic_indexer.IndexerConfig{Index: "x"}}

	_, err = BuildIndexChainAt(context.Background(), splitter, k, "")
	if err == nil {
		t.Fatal("expected error for empty indexName")
	}
	if !strings.Contains(err.Error(), "empty indexName") {
		t.Errorf("expected error to mention empty indexName, got %v", err)
	}
}

func TestBuildIndexChainAt_RejectsNilSplitter(t *testing.T) {
	k := &KBIndexer{base: nil, conf: &elastic_indexer.IndexerConfig{Index: "x"}}
	_, err := BuildIndexChainAt(context.Background(), nil, k, "kb_1")
	if err == nil || !strings.Contains(err.Error(), "nil splitter") {
		t.Errorf("expected nil-splitter error, got %v", err)
	}
}

func TestBuildIndexChainAt_RejectsNilIndexer(t *testing.T) {
	splitter, err := recursive.NewSplitter(context.Background(), &recursive.Config{
		ChunkSize: 100, OverlapSize: 10,
	})
	if err != nil {
		t.Fatalf("create splitter: %v", err)
	}
	_, err = BuildIndexChainAt(context.Background(), splitter, nil, "kb_1")
	if err == nil || !strings.Contains(err.Error(), "nil kbIndexer") {
		t.Errorf("expected nil-kbIndexer error, got %v", err)
	}
}

// TestBuildIndexChainAt_ClosureCapturesIndex verifies that the chain
// compiled by BuildIndexChainAt, when invoked, calls the underlying
// indexer's Store with WithIndex(indexName) for the index name passed
// to the build function. We test this by inspecting the resolveIndex
// behavior, which is the function the lambda calls into.
//
// We can't easily intercept the lambda's actual call without running
// the chain end-to-end (which requires a real file + embedder), so
// the strongest verifiable claim is: BuildIndexChainAt with indexName=X
// doesn't return an error (compile succeeds), AND the bound config
// is preserved unchanged so that the per-call WithIndex in the lambda
// will route to X.
func TestBuildIndexChainAt_ClosureCapturesIndex(t *testing.T) {
	splitter, err := recursive.NewSplitter(context.Background(), &recursive.Config{
		ChunkSize: 100, OverlapSize: 10,
	})
	if err != nil {
		t.Fatalf("create splitter: %v", err)
	}
	// Construct a KBIndexer with bound config.Index = "_placeholder"
	// (the typical startup config). The slow path (rebuild) inside
	// Store will only fire if WithIndex routes to a different index
	// and would need a real ES client. So we just verify compile
	// succeeds and the bound config is not mutated.
	k := &KBIndexer{
		base: nil,
		conf: &elastic_indexer.IndexerConfig{Index: PlaceholderIndex},
	}
	runnable, err := BuildIndexChainAt(context.Background(), splitter, k, "kb_2")
	if err != nil {
		t.Fatalf("BuildIndexChainAt: %v", err)
	}
	if runnable == nil {
		t.Fatal("expected non-nil runnable")
	}
	if k.conf.Index != PlaceholderIndex {
		t.Errorf("bound config.Index was mutated: %q", k.conf.Index)
	}
	// Verify resolveIndex with the WithIndex the lambda will pass.
	idx := k.resolveIndex([]indexer.Option{WithIndex("kb_2")})
	if idx != "kb_2" {
		t.Errorf("expected resolveIndex to return kb_2, got %q", idx)
	}
}
