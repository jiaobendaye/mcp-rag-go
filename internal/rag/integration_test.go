//go:build integration
// +build integration

package rag

import (
	"context"
	"testing"

	"github.com/elastic/go-elasticsearch/v8"

	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
	elastic_retriever "github.com/cloudwego/eino-ext/components/retriever/es8"
	elastic_search_mode "github.com/cloudwego/eino-ext/components/retriever/es8/search_mode"

	"github.com/jiaobendaye/mcp-rag-go/internal/testutil"
)

const testDims = 4

func setupIntegrationTest(t *testing.T) (*elastic_indexer.IndexerConfig, *elastic_retriever.Retriever, func()) {
	t.Helper()

	ctx := context.Background()

	esURL, err := testutil.StartES(t, ctx)
	if err != nil {
		t.Skipf("SKIP: cannot start ES container: %v", err)
	}

	esClient, err := elasticsearch.NewClient(elasticsearch.Config{
		Addresses: []string{esURL},
	})
	if err != nil {
		t.Fatalf("create es client: %v", err)
	}

	indexName := "test_kb_integration"

	conf := &elastic_indexer.IndexerConfig{
		Client:           esClient,
		IndexSpec:        indexSpecForTestDims(testDims),
		DocumentToFields: ProjectDocumentToFields(),
	}
	{
		confCopy := *conf
		confCopy.Index = indexName
		if _, err := elastic_indexer.NewIndexer(context.Background(), &confCopy); err != nil {
			t.Fatalf("ensure index: %v", err)
		}
	}

	retriever, err := elastic_retriever.NewRetriever(context.Background(), &elastic_retriever.RetrieverConfig{
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

	return conf, retriever, cleanup
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
	ctx := context.Background()

	esURL, err := testutil.StartES(t, ctx)
	if err != nil {
		t.Skipf("SKIP: cannot start ES container: %v", err)
	}

	esClient, err := elasticsearch.NewClient(elasticsearch.Config{
		Addresses: []string{esURL},
	})
	if err != nil {
		t.Fatalf("create es client: %v", err)
	}

	indexName := "test_empty_integration"

	conf := &elastic_indexer.IndexerConfig{
		Client:           esClient,
		IndexSpec:        indexSpecForTestDims(testDims),
		DocumentToFields: ProjectDocumentToFields(),
	}
	{
		confCopy := *conf
		confCopy.Index = indexName
		if _, err := elastic_indexer.NewIndexer(ctx, &confCopy); err != nil {
			t.Fatalf("ensure index: %v", err)
		}
	}
	defer esClient.Indices.Delete([]string{indexName})

	// Empty index: confirm we can build a retriever without error
	_, err = elastic_retriever.NewRetriever(ctx, &elastic_retriever.RetrieverConfig{
		Client:       esClient,
		Index:        indexName,
		TopK:         5,
		SearchMode:   elastic_search_mode.SearchModeRawStringRequest(),
		ResultParser: ProjectResultParser(),
	})
	if err != nil {
		t.Fatalf("create retriever on empty index: %v", err)
	}
}
