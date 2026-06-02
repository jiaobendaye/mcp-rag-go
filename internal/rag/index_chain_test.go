package rag

import (
	"context"
	"strings"
	"testing"

	"github.com/cloudwego/eino-ext/components/document/transformer/splitter/recursive"

	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
)

func TestBuildIndexChain_RejectsNilSplitter(t *testing.T) {
	conf := &elastic_indexer.IndexerConfig{Index: "x"}
	_, err := BuildIndexChain(context.Background(), nil, conf, "kb_test")
	if err == nil || !strings.Contains(err.Error(), "nil splitter") {
		t.Errorf("expected nil-splitter error, got %v", err)
	}
}

func TestBuildIndexChain_RejectsNilIndexerConf(t *testing.T) {
	splitter, err := recursive.NewSplitter(context.Background(), &recursive.Config{
		ChunkSize: 100, OverlapSize: 10,
	})
	if err != nil {
		t.Fatalf("create splitter: %v", err)
	}
	_, err = BuildIndexChain(context.Background(), splitter, nil, "kb_test")
	if err == nil || !strings.Contains(err.Error(), "nil indexerConf") {
		t.Errorf("expected nil-indexerConf error, got %v", err)
	}
}

// TestBuildIndexChain_ConfigNotMutated verifies that BuildIndexChain copies
// the config rather than mutating it.
func TestBuildIndexChain_ConfigNotMutated(t *testing.T) {
	splitter, err := recursive.NewSplitter(context.Background(), &recursive.Config{
		ChunkSize: 100, OverlapSize: 10,
	})
	if err != nil {
		t.Fatalf("create splitter: %v", err)
	}
	conf := &elastic_indexer.IndexerConfig{Index: "test_placeholder"}
	_, err = BuildIndexChain(context.Background(), splitter, conf, "kb_test")
	// Without a real ES client, NewIndexer will fail — that's expected.
	if conf.Index != "test_placeholder" {
		t.Errorf("config.Index was mutated: %q", conf.Index)
	}
	_ = err
}
