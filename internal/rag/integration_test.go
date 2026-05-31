//go:build integration
// +build integration

package rag

import (
	"context"
	"os"
	"testing"

	"github.com/elastic/go-elasticsearch/v8"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
)

const testDims = 1536

// integEmbedder returns testDims-dimensional dummy vectors for integration tests.
type integEmbedder struct{}

func (e *integEmbedder) EmbedStrings(ctx context.Context, texts []string) ([][]float64, error) {
	vecs := make([][]float64, len(texts))
	for i := range vecs {
		vec := make([]float64, testDims)
		for j := range vec {
			vec[j] = 0.01 * float64(j%100+1)
		}
		vec[0] = float64(i+1) * 0.1
		vecs[i] = vec
	}
	return vecs, nil
}

func makeQueryVector() []float32 {
	vec := make([]float32, testDims)
	for i := range vec {
		vec[i] = 0.01
	}
	vec[0] = 0.15
	return vec
}

func makeDummyEmbeddings(n int) [][]float32 {
	vectors := make([][]float32, n)
	for i := range vectors {
		vec := make([]float32, testDims)
		for j := range vec {
			vec[j] = 0.01 * float32(j%100+1)
		}
		vec[0] = float32(i+1) * 0.1
		vectors[i] = vec
	}
	return vectors
}

func setupIntegrationTest(t *testing.T) (*ES8Indexer, func()) {
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
	indexer := NewES8Indexer(esClient, indexName)

	if err := indexer.EnsureIndex(context.Background(), testDims); err != nil {
		t.Fatalf("ensure index: %v", err)
	}

	cleanup := func() {
		esClient.Indices.Delete([]string{indexName})
	}

	return indexer, cleanup
}

func TestIntegrationIndexAndSearch(t *testing.T) {
	indexer, cleanup := setupIntegrationTest(t)
	defer cleanup()

	ctx := context.Background()

	chunks := []Chunk{
		{
			ID:          "test_chunk_0001",
			DocumentID:  "test_doc_001",
			ChunkIndex:  0,
			TotalChunks: 2,
			Source:      "test.md",
			Filename:    "test.md",
			FileType:    "md",
			Content:     "RAG是检索增强生成技术，结合了信息检索和文本生成的能力。",
		},
		{
			ID:          "test_chunk_0002",
			DocumentID:  "test_doc_001",
			ChunkIndex:  1,
			TotalChunks: 2,
			Source:      "test.md",
			Filename:    "test.md",
			FileType:    "md",
			Content:     "Elasticsearch是一个分布式搜索和分析引擎。",
		},
	}

	vectors := makeDummyEmbeddings(len(chunks))

	if err := indexer.IndexChunks(ctx, chunks, vectors); err != nil {
		t.Fatalf("index chunks: %v", err)
	}

	indexer.client.Indices.Refresh(
		indexer.client.Indices.Refresh.WithIndex(indexer.indexName),
	)

	hits, err := indexer.Search(ctx, makeQueryVector(), 5, 0.1)
	if err != nil {
		t.Fatalf("search: %v", err)
	}

	if len(hits) == 0 {
		t.Error("expected at least 1 search result")
	}
	if len(hits) > 0 && hits[0].Content == "" {
		t.Error("expected non-empty content in search hit")
	}
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
	indexer := NewES8Indexer(esClient, indexName)

	ctx := context.Background()
	if err := indexer.EnsureIndex(ctx, testDims); err != nil {
		t.Fatalf("ensure index: %v", err)
	}
	defer esClient.Indices.Delete([]string{indexName})

	hits, err := indexer.Search(ctx, makeQueryVector(), 5, 0.7)
	if err != nil {
		t.Fatalf("search on empty index: %v", err)
	}

	if len(hits) != 0 {
		t.Errorf("expected 0 hits on empty index, got %d", len(hits))
	}
}

func TestIntegrationFileUpload(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := tmpDir + "/test_document.md"
	content := "# Test Document\n\nThis is a test RAG document about information retrieval."
	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	indexer, cleanup := setupIntegrationTest(t)
	defer cleanup()

	emb := &integEmbedder{}
	pipeline := NewIndexPipeline(emb, indexer, 4000, 200)

	result, err := pipeline.IndexFile(context.Background(), filePath)
	if err != nil {
		t.Fatalf("index file: %v", err)
	}

	if result.ChunkCount == 0 {
		t.Error("expected at least 1 chunk")
	}
	if result.DocumentID == "" {
		t.Error("expected non-empty document ID")
	}

	// Verify we can search the indexed content
	indexer.client.Indices.Refresh(
		indexer.client.Indices.Refresh.WithIndex(indexer.indexName),
	)
	hits, err := indexer.Search(context.Background(), makeQueryVector(), 3, 0.1)
	if err != nil {
		t.Fatalf("search after index: %v", err)
	}
		if len(hits) == 0 {
		t.Error("expected search hits after file indexing")
	}
}

func TestIntegrationHybridSearch(t *testing.T) {
	indexer, cleanup := setupIntegrationTest(t)
	defer cleanup()

	ctx := context.Background()

	// Index two documents with different content styles
	chunks := []Chunk{
		{ID: "hyb_c1", DocumentID: "hyb_d1", ChunkIndex: 0, TotalChunks: 1,
			Source: "vector.md", Filename: "vector.md", FileType: "md",
			Content: "向量搜索使用余弦相似度计算文档与查询之间的语义相关性"},
		{ID: "hyb_c2", DocumentID: "hyb_d2", ChunkIndex: 0, TotalChunks: 1,
			Source: "keyword.md", Filename: "keyword.md", FileType: "md",
			Content: "全文检索基于倒排索引，使用BM25算法进行关键词匹配和排序"},
	}
	vectors := makeDummyEmbeddings(len(chunks))

	if err := indexer.IndexChunks(ctx, chunks, vectors); err != nil {
		t.Fatalf("index chunks: %v", err)
	}
	indexer.client.Indices.Refresh(indexer.client.Indices.Refresh.WithIndex(indexer.indexName))

	// Hybrid search: vector matching should find vector doc, BM25 should find keyword doc
	hits, err := indexer.SearchHybrid(ctx, "关键词匹配", makeQueryVector(), 2, 0.0)
	if err != nil {
		t.Fatalf("hybrid search: %v", err)
	}

	if len(hits) == 0 {
		t.Error("expected hybrid search results")
	}
	t.Logf("Hybrid search found %d results", len(hits))
	for _, h := range hits {
		t.Logf("  score=%.4f content=%s", h.Score, h.Content[:min(50, len(h.Content))])
	}
}
