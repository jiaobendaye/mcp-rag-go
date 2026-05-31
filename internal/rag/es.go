package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"sort"

	"github.com/elastic/go-elasticsearch/v8"
)

// ES8Indexer implements Indexer and Searcher using Elasticsearch 8.x.
type ES8Indexer struct {
	client    *elasticsearch.Client
	indexName string
}

// NewES8Indexer creates a new ES8Indexer.
func NewES8Indexer(client *elasticsearch.Client, indexName string) *ES8Indexer {
	return &ES8Indexer{
		client:    client,
		indexName: indexName,
	}
}

// HealthCheck verifies the ES cluster is reachable.
func (e *ES8Indexer) HealthCheck(ctx context.Context) error {
	res, err := e.client.Ping()
	if err != nil {
		return fmt.Errorf("es ping: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		body, _ := io.ReadAll(res.Body)
		return fmt.Errorf("es unhealthy: status=%d body=%s", res.StatusCode, string(body))
	}
	return nil
}

// EnsureIndex creates the ES index with proper mapping if it doesn't exist.
func (e *ES8Indexer) EnsureIndex(ctx context.Context, dims int) error {
	// Check if index exists
	res, err := e.client.Indices.Exists([]string{e.indexName})
	if err != nil {
		return fmt.Errorf("check index exists: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode == 200 {
		log.Printf("Index %s already exists", e.indexName)
		return nil
	}

	// Create index with mapping
	log.Printf("Creating index %s with dims=%d", e.indexName, dims)

	mapping := map[string]any{
		"mappings": map[string]any{
			"dynamic": "strict",
			"properties": map[string]any{
				"content":        map[string]any{"type": "text"},
				"content_vector": map[string]any{"type": "dense_vector", "dims": dims, "similarity": "cosine"},
				"document_id":    map[string]any{"type": "keyword"},
				"chunk_index":    map[string]any{"type": "integer"},
				"total_chunks":   map[string]any{"type": "integer"},
				"source":         map[string]any{"type": "keyword"},
				"filename":       map[string]any{"type": "keyword"},
				"file_type":      map[string]any{"type": "keyword"},
				"processed_at":   map[string]any{"type": "date"},
			},
		},
	}

	body, err := json.Marshal(mapping)
	if err != nil {
		return fmt.Errorf("marshal mapping: %w", err)
	}

	res, err = e.client.Indices.Create(e.indexName, e.client.Indices.Create.WithBody(bytes.NewReader(body)))
	if err != nil {
		return fmt.Errorf("create index: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		respBody, _ := io.ReadAll(res.Body)
		return fmt.Errorf("create index failed: status=%d body=%s", res.StatusCode, string(respBody))
	}

	log.Printf("Index %s created successfully", e.indexName)
	return nil
}

// IndexChunks bulk-indexes chunks with their embedding vectors into ES.
func (e *ES8Indexer) IndexChunks(ctx context.Context, chunks []Chunk, vectors [][]float32) error {
	if len(chunks) == 0 {
		return nil
	}
	if len(chunks) != len(vectors) {
		return fmt.Errorf("chunks and vectors length mismatch: %d != %d", len(chunks), len(vectors))
	}

	var buf bytes.Buffer

	for i, chunk := range chunks {
		// Action line
		action := map[string]any{
			"index": map[string]any{
				"_index": e.indexName,
				"_id":    chunk.ID,
			},
		}
		actionJSON, _ := json.Marshal(action)
		buf.Write(actionJSON)
		buf.WriteByte('\n')

		// Document line
		doc := map[string]any{
			"content":        chunk.Content,
			"content_vector": vectors[i],
			"document_id":    chunk.DocumentID,
			"chunk_index":    chunk.ChunkIndex,
			"total_chunks":   chunk.TotalChunks,
			"source":         chunk.Source,
			"filename":       chunk.Filename,
			"file_type":      chunk.FileType,
			"processed_at":   Now().UTC().Format(timeFormat),
		}
		docJSON, _ := json.Marshal(doc)
		buf.Write(docJSON)
		buf.WriteByte('\n')
	}

	res, err := e.client.Bulk(bytes.NewReader(buf.Bytes()), e.client.Bulk.WithContext(ctx))
	if err != nil {
		return fmt.Errorf("bulk index: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		respBody, _ := io.ReadAll(res.Body)
		return fmt.Errorf("bulk index failed: status=%d body=%s", res.StatusCode, string(respBody))
	}

	// Parse response for errors
	var bulkResp struct {
		Errors bool `json:"errors"`
	}
	if err := json.NewDecoder(res.Body).Decode(&bulkResp); err != nil {
		// If we can't parse, assume success since status was 2xx
		return nil
	}

	if bulkResp.Errors {
		return fmt.Errorf("bulk index had errors")
	}

	return nil
}

// Search performs KNN vector search on the content_vector field (backward compat).
func (e *ES8Indexer) Search(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
	return e.searchKNN(ctx, queryVector, topK, minScore)
}

// SearchHybrid performs hybrid search based on mode:
//
//	"hybrid": KNN + BM25 manual fusion (default, works on free license)
//	"rrf":    ES native RRF fusion (requires paid license)
//	"knn":    pure KNN vector search
func (e *ES8Indexer) SearchHybrid(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
	return e.searchWithMode(ctx, query, queryVector, topK, minScore, "hybrid")
}

// SearchWithMode performs search with explicit mode selection.
func (e *ES8Indexer) SearchWithMode(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error) {
	return e.searchWithMode(ctx, query, queryVector, topK, minScore, mode)
}

func (e *ES8Indexer) searchWithMode(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error) {
	switch mode {
	case "knn":
		return e.searchKNN(ctx, queryVector, topK, minScore)
	case "rrf":
		return e.searchRRF(ctx, query, queryVector, topK, minScore)
	default: // "hybrid"
		return e.searchManualFusion(ctx, query, queryVector, topK, minScore)
	}
}

// searchKNN performs pure KNN vector search.
func (e *ES8Indexer) searchKNN(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
	numCandidates := topK * 10
	if numCandidates < 50 {
		numCandidates = 50
	}
	req := map[string]any{
		"knn": []map[string]any{{
			"field":          "content_vector",
			"query_vector":   queryVector,
			"k":              topK,
			"num_candidates": numCandidates,
		}},
		"size": topK,
	}
	if minScore > 0 {
		req["min_score"] = minScore
	}
	return e.doSearch(ctx, req)
}

// searchRRF performs ES native KNN + BM25 + RRF fusion (requires paid license).
func (e *ES8Indexer) searchRRF(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
	numCandidates := topK * 10
	if numCandidates < 50 {
		numCandidates = 50
	}
	req := map[string]any{
		"knn": []map[string]any{{
			"field":          "content_vector",
			"query_vector":   queryVector,
			"k":              topK,
			"num_candidates": numCandidates,
		}},
		"query": map[string]any{
			"bool": map[string]any{
				"must": []map[string]any{
					{"match": map[string]any{"content": query}},
				},
			},
		},
		"rank": map[string]any{
			"rrf": map[string]any{
				"rank_constant":   60,
				"rank_window_size": 100,
			},
		},
		"size": topK,
	}
	if minScore > 0 {
		req["min_score"] = minScore
	}
	return e.doSearch(ctx, req)
}

// searchBM25 performs pure BM25 full-text search.
func (e *ES8Indexer) searchBM25(ctx context.Context, query string, topK int, minScore float64) ([]SearchHit, error) {
	req := map[string]any{
		"query": map[string]any{
			"match": map[string]any{
				"content": query,
			},
		},
		"size": topK,
	}
	if minScore > 0 {
		req["min_score"] = minScore
	}
	return e.doSearch(ctx, req)
}

// searchManualFusion performs KNN + BM25 with client-side RRF-like fusion.
func (e *ES8Indexer) searchManualFusion(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
	candidateK := topK * 2
	if candidateK < 10 {
		candidateK = 10
	}

	// Run KNN and BM25 in parallel
	type result struct {
		hits []SearchHit
		err  error
	}
	knnCh := make(chan result, 1)
	bm25Ch := make(chan result, 1)

	go func() {
		h, err := e.searchKNN(ctx, queryVector, candidateK, 0)
		knnCh <- result{h, err}
	}()
	go func() {
		h, err := e.searchBM25(ctx, query, candidateK, 0)
		bm25Ch <- result{h, err}
	}()

	knnRes := <-knnCh
	bm25Res := <-bm25Ch

	if knnRes.err != nil {
		return nil, fmt.Errorf("knn search: %w", knnRes.err)
	}
	if bm25Res.err != nil {
		return nil, fmt.Errorf("bm25 search: %w", bm25Res.err)
	}

	return fuseResults(knnRes.hits, bm25Res.hits, topK, minScore), nil
}

// fuseResults merges KNN and BM25 results using RRF-like reciprocal rank fusion.
func fuseResults(knnHits, bm25Hits []SearchHit, topK int, minScore float64) []SearchHit {
	const (
		vectorWeight  = 0.7
		keywordWeight = 0.3
		fusionK       = 60.0
	)

	// Track best score per chunk_id
	type candidate struct {
		hit          *SearchHit
		vectorScore  float64
		keywordScore float64
		vectorRank   int
		keywordRank  int
	}

	candidates := make(map[string]*candidate)

	// Process KNN results
	for i, h := range knnHits {
		c := &candidate{hit: &h, vectorScore: h.Score, vectorRank: i + 1}
		candidates[h.ChunkID] = c
	}

	// Process BM25 results
	for i, h := range bm25Hits {
		c, exists := candidates[h.ChunkID]
		if !exists {
			c = &candidate{hit: &h}
			candidates[h.ChunkID] = c
		}
		c.keywordScore = h.Score
		c.keywordRank = i + 1
	}

	// Calculate fused scores
	type scored struct {
		hit   *SearchHit
		score float64
	}
	var ranked []scored
	for _, c := range candidates {
		score := 0.0
		if c.vectorRank > 0 {
			score += vectorWeight * (1.0 / (fusionK + float64(c.vectorRank)))
			score += vectorWeight * 0.05 * c.vectorScore
		}
		if c.keywordRank > 0 {
			score += keywordWeight * (1.0 / (fusionK + float64(c.keywordRank)))
			score += keywordWeight * 0.05 * c.keywordScore
		}

		if score < minScore {
			continue
		}
		ranked = append(ranked, scored{hit: c.hit, score: score})
	}

	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].score > ranked[j].score
	})

	if len(ranked) > topK {
		ranked = ranked[:topK]
	}

	result := make([]SearchHit, len(ranked))
	for i, r := range ranked {
		result[i] = *r.hit
		result[i].Score = r.score
	}
	return result
}

func (e *ES8Indexer) doSearch(ctx context.Context, req map[string]any) ([]SearchHit, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal search request: %w", err)
	}

	res, err := e.client.Search(
		e.client.Search.WithContext(ctx),
		e.client.Search.WithIndex(e.indexName),
		e.client.Search.WithBody(bytes.NewReader(body)),
	)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		respBody, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("search failed: status=%d body=%s", res.StatusCode, string(respBody))
	}

	var searchResp struct {
		Hits struct {
			Hits []struct {
				ID     string  `json:"_id"`
				Score  float64 `json:"_score"`
				Source struct {
					Content    string `json:"content"`
					DocumentID string `json:"document_id"`
					ChunkIndex int    `json:"chunk_index"`
					Source     string `json:"source"`
					Filename   string `json:"filename"`
				} `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}

	if err := json.NewDecoder(res.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("decode search response: %w", err)
	}

	hits := make([]SearchHit, 0, len(searchResp.Hits.Hits))
	for _, h := range searchResp.Hits.Hits {
		hits = append(hits, SearchHit{
			ChunkID:    h.ID,
			DocumentID: h.Source.DocumentID,
			Score:      h.Score,
			Source:     h.Source.Source,
			Filename:   h.Source.Filename,
			Content:    h.Source.Content,
			ChunkIndex: h.Source.ChunkIndex,
		})
	}

	return hits, nil
}

const timeFormat = "2006-01-02T15:04:05Z07:00"
