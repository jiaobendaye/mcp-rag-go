//go:build integration
// +build integration

package rag

import (
	"context"
	"os"
	"testing"

	"github.com/elastic/go-elasticsearch/v8"

	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
	elastic_retriever "github.com/cloudwego/eino-ext/components/retriever/es8"
	elastic_search_mode "github.com/cloudwego/eino-ext/components/retriever/es8/search_mode"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
)

const testDims = 4

func setupIntegrationTest(t *testing.T) (*KBIndexer, *KBRetriever, func()) {
	t.Helper()

	cfg := config.DefaultConfig()
	if url := os.Getenv("MCP_RAG_ES_URL"); url != "" {
		cfg.ESUrl = url
	}

	esClient, err := elasticsearch.NewClient(elasticsearch.Config{
		Addresses: []string{cfg.ESUrl},
	})
	if err != nil {
		t.Fatalf("create es client: %v", err)
	}

	indexName := "test_kb_integration"

	indexer, err := NewKBIndexer(context.Background(), &elastic_indexer.IndexerConfig{
		Client:           esClient,
		Index:            indexName,
		IndexSpec:        indexSpecForTestDims(testDims),
		DocumentToFields: ProjectDocumentToFields(),
	})
	if err != nil {
		t.Fatalf("create indexer: %v", err)
	}
	if err := indexer.EnsureIndexForKB(context.Background(), indexName); err != nil {
		t.Fatalf("ensure index: %v", err)
	}

	retriever, err := NewKBRetriever(context.Background(), &elastic_retriever.RetrieverConfig{
		Client:       esClient,
		Index:        indexName,
		TopK:         5,
		SearchMode:   elastic_search_mode.SearchModeRawStringRequest(),
		ResultParser: ProjectResultParser(),
	})
	if err != nil {
		t.Fatalf("create retriever: %v", err)
	}

	cleanup := func() {
		esClient.Indices.Delete([]string{indexName})
	}

	return indexer, retriever, cleanup
}

func indexSpecForTestDims(dims int) *elastic_indexer.IndexSpec {
	return &elastic_indexer.IndexSpec{
		Settings: map[string]any{
			"number_of_shards":   1,
			"number_of_replicas": 0,
		},
		Mappings: map[string]any{
			"properties": map[string]any{
				"content":        map[string]any{"type": "text"},
				"content_vector": map[string]any{"type": "dense_vector", "dims": dims, "similarity": "cosine"},
				"document_id":    map[string]any{"type": "keyword"},
				"chunk_id":       map[string]any{"type": "keyword"},
				"source":         map[string]any{"type": "keyword"},
				"filename":       map[string]any{"type": "keyword"},
				"file_type":      map[string]any{"type": "keyword"},
				"chunk_index":    map[string]any{"type": "integer"},
				"total_chunks":   map[string]any{"type": "integer"},
				"processed_at":   map[string]any{"type": "date"},
			},
		},
	}
}

// TestIntegrationIndexAndSearch verifies the KBIndexer and KBRetriever
// can be wired against a real Elasticsearch and built without error.
// It does not assert exact hit counts because the test index is empty
// in this minimal smoke test; full round-trip tests live in
// internal/server/integration_test.go.
func TestIntegrationIndexAndSearch(t *testing.T) {
	_, _, cleanup := setupIntegrationTest(t)
	defer cleanup()
	t.Log("integration setup OK; indexer/retriever built without error")
}

func TestIntegrationSearchEmptyIndex(t *testing.T) {
	cfg := config.DefaultConfig()
	if url := os.Getenv("MCP_RAG_ES_URL"); url != "" {
		cfg.ESUrl = url
	}

	esClient, err := elasticsearch.NewClient(elasticsearch.Config{
		Addresses: []string{cfg.ESUrl},
	})
	if err != nil {
		t.Fatalf("create es client: %v", err)
	}

	indexName := "test_empty_integration"
	ctx := context.Background()

	indexer, err := NewKBIndexer(ctx, &elastic_indexer.IndexerConfig{
		Client:    esClient,
		Index:     indexName,
		IndexSpec: indexSpecForTestDims(testDims),
	})
	if err != nil {
		t.Fatalf("create indexer: %v", err)
	}
	if err := indexer.EnsureIndexForKB(ctx, indexName); err != nil {
		t.Fatalf("ensure index: %v", err)
	}
	defer esClient.Indices.Delete([]string{indexName})

	// Empty index: confirm we can build a retriever without error
	_, err = NewKBRetriever(ctx, &elastic_retriever.RetrieverConfig{
		Client:     esClient,
		Index:      indexName,
		TopK:       5,
		SearchMode: elastic_search_mode.SearchModeRawStringRequest(),
	})
	if err != nil {
		t.Fatalf("create retriever on empty index: %v", err)
	}
}
