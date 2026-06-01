package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"sort"

	"github.com/cloudwego/eino/components/indexer"
	"github.com/cloudwego/eino/schema"
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
	return e.ensureIndex(ctx, e.indexName, dims)
}

// IndexName returns the ES index name this indexer writes to.
func (e *ES8Indexer) IndexName() string {
	return e.indexName
}

// EnsureIndexForKB creates an ES index with the given name if it doesn't exist.
func (e *ES8Indexer) EnsureIndexForKB(ctx context.Context, indexName string, dims int) error {
	return e.ensureIndex(ctx, indexName, dims)
}

func (e *ES8Indexer) ensureIndex(ctx context.Context, indexName string, dims int) error {
	// Check if index exists
	res, err := e.client.Indices.Exists([]string{indexName})
	if err != nil {
		return fmt.Errorf("check index exists: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode == 200 {
		log.Printf("Index %s already exists", indexName)
		return nil
	}

	// Create index with mapping
	log.Printf("Creating index %s with dims=%d", indexName, dims)

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

	res, err = e.client.Indices.Create(indexName, e.client.Indices.Create.WithBody(bytes.NewReader(body)))
	if err != nil {
		return fmt.Errorf("create index: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		respBody, _ := io.ReadAll(res.Body)
		return fmt.Errorf("create index failed: status=%d body=%s", res.StatusCode, string(respBody))
	}

	log.Printf("Index %s created successfully", indexName)
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

// Store implements eino indexer.Indexer. It extracts chunks and vectors from eino Documents.
func (e *ES8Indexer) Store(ctx context.Context, docs []*schema.Document, opts ...indexer.Option) ([]string, error) {
	chunks := make([]Chunk, 0, len(docs))
	vectors := make([][]float32, 0, len(docs))
	ids := make([]string, 0, len(docs))

	for i, doc := range docs {
		// Extract vector from metadata (set by Embedder node)
		var vec []float32
		if v, ok := doc.MetaData["_embedding"]; ok {
			switch vv := v.(type) {
			case []float64:
				vec = make([]float32, len(vv))
				for j, f := range vv {
					vec[j] = float32(f)
				}
			case []float32:
				vec = vv
			}
		}

		chunk := Chunk{
			ID:          fmt.Sprintf("%s_chunk_%04d", doc.ID, i),
			DocumentID:  doc.ID,
			ChunkIndex:  i,
			TotalChunks: len(docs),
			Source:      stringMeta(doc.MetaData, "source", doc.ID),
			Filename:    stringMeta(doc.MetaData, "filename", "unknown"),
			FileType:    stringMeta(doc.MetaData, "file_type", "text"),
			Content:     doc.Content,
		}
		chunks = append(chunks, chunk)
		vectors = append(vectors, vec)
		ids = append(ids, chunk.ID)
	}

	if err := e.IndexChunks(ctx, chunks, vectors); err != nil {
		return nil, err
	}
	return ids, nil
}

func stringMeta(m map[string]any, key, defaultVal string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return defaultVal
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
	return e.SearchWithWeights(ctx, query, queryVector, topK, minScore, mode, DefaultWeights)
}

// SearchWithWeights performs search with explicit mode and custom hybrid fusion weights.
func (e *ES8Indexer) SearchWithWeights(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string, weights SearchWeights) ([]SearchHit, error) {
	switch mode {
	case "knn":
		return e.searchKNN(ctx, queryVector, topK, minScore)
	case "rrf":
		return e.searchRRF(ctx, query, queryVector, topK, minScore)
	default: // "hybrid"
		return e.searchManualFusionWithWeights(ctx, query, queryVector, topK, minScore, weights)
	}
}

func (e *ES8Indexer) searchWithMode(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error) {
	return e.SearchWithMode(ctx, query, queryVector, topK, minScore, mode)
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

// searchManualFusionWithWeights is like searchManualFusion but with configurable weights.
func (e *ES8Indexer) searchManualFusionWithWeights(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, weights SearchWeights) ([]SearchHit, error) {
	candidateK := topK * 2
	if candidateK < 10 {
		candidateK = 10
	}

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

	return fuseResultsWithWeights(knnRes.hits, bm25Res.hits, topK, minScore, weights), nil
}

// fuseResults merges KNN and BM25 results using RRF-like reciprocal rank fusion.
func fuseResults(knnHits, bm25Hits []SearchHit, topK int, minScore float64) []SearchHit {
	return fuseResultsWithWeights(knnHits, bm25Hits, topK, minScore, SearchWeights{Vector: 0.7, Keyword: 0.3})
}

// fuseResultsWithWeights merges KNN and BM25 results with configurable weights.
func fuseResultsWithWeights(knnHits, bm25Hits []SearchHit, topK int, minScore float64, weights SearchWeights) []SearchHit {
	fusionK := 60.0

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
			score += weights.Vector * (1.0 / (fusionK + float64(c.vectorRank)))
			score += weights.Vector * 0.05 * c.vectorScore
		}
		if c.keywordRank > 0 {
			score += weights.Keyword * (1.0 / (fusionK + float64(c.keywordRank)))
			score += weights.Keyword * 0.05 * c.keywordScore
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

// ListDocuments returns paginated documents from an index using ES aggregation.
func (e *ES8Indexer) ListDocuments(indexName string, limit, offset int) (*DocumentList, error) {
	return e.listDocuments(indexName, limit, offset, nil)
}

func (e *ES8Indexer) listDocuments(indexName string, limit, offset int, filename *string) (*DocumentList, error) {
	ctx := context.Background()
	req := map[string]any{"size": 0, "aggs": map[string]any{
		"by_doc": map[string]any{
			"terms": map[string]any{"field": "document_id", "size": limit + offset},
			"aggs": map[string]any{
				"sample":      map[string]any{"top_hits": map[string]any{"size": 1, "_source": []string{"content", "document_id", "source", "filename", "file_type", "chunk_index", "processed_at", "chunk_char_count"}}},
				"chunk_count": map[string]any{"value_count": map[string]string{"field": "chunk_index"}},
			},
		},
	}}
	body, _ := json.Marshal(req)
	res, err := e.client.Search(e.client.Search.WithContext(ctx), e.client.Search.WithIndex(indexName), e.client.Search.WithBody(bytes.NewReader(body)))
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode == 404 {
		return &DocumentList{Total: 0, Documents: []DocInfo{}, Limit: limit, Offset: offset}, nil
	}
	if res.StatusCode >= 400 {
		b, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("list documents: %s", string(b))
	}
	return parseDocList(res.Body, limit, offset)
}

func parseDocList(r io.Reader, limit, offset int) (*DocumentList, error) {
	var aggResp struct {
		Aggregations struct {
			ByDoc struct {
				Buckets []struct {
					Key string `json:"key"`
					Sample struct {
						Hits struct {
							Hits []struct {
								Source struct {
									Content        string `json:"content"`
									DocumentID     string `json:"document_id"`
									Source         string `json:"source"`
									Filename       string `json:"filename"`
									FileType       string `json:"file_type"`
									ChunkIndex     int    `json:"chunk_index"`
									ProcessedAt    string `json:"processed_at"`
									ChunkCharCount int    `json:"chunk_char_count"`
								} `json:"_source"`
							} `json:"hits"`
						} `json:"hits"`
					} `json:"sample"`
					ChunkCount struct{ Value int } `json:"chunk_count"`
				} `json:"buckets"`
			} `json:"by_doc"`
		} `json:"aggregations"`
	}
	json.NewDecoder(r).Decode(&aggResp)
	buckets := aggResp.Aggregations.ByDoc.Buckets
	total := len(buckets)
	var docs []DocInfo
	for i := offset; i < total && i < offset+limit; i++ {
		b := buckets[i]
		if len(b.Sample.Hits.Hits) > 0 {
			src := b.Sample.Hits.Hits[0].Source
			docs = append(docs, DocInfo{
				ID: src.DocumentID, Content: truncate(src.Content, 200),
				Source: src.Source, Filename: src.Filename, FileType: src.FileType,
				ChunkCount: b.ChunkCount.Value, ProcessedAt: src.ProcessedAt,
			})
		}
	}
	if docs == nil {
		docs = []DocInfo{}
	}
	return &DocumentList{Total: total, Documents: docs, Limit: limit, Offset: offset}, nil
}

// DeleteDocument removes all chunks with the given document_id.
func (e *ES8Indexer) DeleteDocument(indexName, documentID string) error {
	req := map[string]any{"query": map[string]any{"term": map[string]any{"document_id": documentID}}}
	body, _ := json.Marshal(req)
	res, err := e.client.DeleteByQuery([]string{indexName}, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("delete document: %w", err)
	}
	res.Body.Close()
	return nil
}

// ListFiles returns aggregated file information.
func (e *ES8Indexer) ListFiles(indexName string) ([]FileInfo, error) {
	ctx := context.Background()
	req := map[string]any{"size": 0, "aggs": map[string]any{
		"by_file": map[string]any{
			"terms": map[string]any{"field": "filename", "size": 200},
			"aggs": map[string]any{
				"sample":      map[string]any{"top_hits": map[string]any{"size": 1, "_source": []string{"document_id", "source", "file_type", "processed_at"}}},
				"chunk_count": map[string]any{"value_count": map[string]string{"field": "chunk_index"}},
				"total_chars": map[string]any{"sum": map[string]string{"field": "chunk_char_count"}},
			},
		},
	}}
	body, _ := json.Marshal(req)
	res, err := e.client.Search(e.client.Search.WithContext(ctx), e.client.Search.WithIndex(indexName), e.client.Search.WithBody(bytes.NewReader(body)))
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode == 404 {
		return []FileInfo{}, nil
	}
	if res.StatusCode >= 400 {
		b, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("list files: %s", string(b))
	}
	var aggResp struct {
		Aggregations struct {
			ByFile struct {
				Buckets []struct {
					Key        string  `json:"key"`
					ChunkCount struct{ Value int } `json:"chunk_count"`
					TotalChars struct{ Value float64 } `json:"total_chars"`
					Sample     struct {
						Hits struct {
							Hits []struct {
								Source struct {
									DocumentID  string `json:"document_id"`
									Source      string `json:"source"`
									FileType    string `json:"file_type"`
									ProcessedAt string `json:"processed_at"`
								} `json:"_source"`
							} `json:"hits"`
						} `json:"hits"`
					} `json:"sample"`
				} `json:"buckets"`
			} `json:"by_file"`
		} `json:"aggregations"`
	}
	json.NewDecoder(res.Body).Decode(&aggResp)
	var files []FileInfo
	for _, b := range aggResp.Aggregations.ByFile.Buckets {
		fi := FileInfo{Filename: b.Key, ChunkCount: b.ChunkCount.Value, TotalChars: int(b.TotalChars.Value)}
		if len(b.Sample.Hits.Hits) > 0 {
			src := b.Sample.Hits.Hits[0].Source
			fi.DocumentID, fi.Source, fi.FileType, fi.ProcessedAt = src.DocumentID, src.Source, src.FileType, src.ProcessedAt
		}
		files = append(files, fi)
	}
	if files == nil {
		files = []FileInfo{}
	}
	return files, nil
}

// DeleteFile removes all chunks with the given filename.
func (e *ES8Indexer) DeleteFile(indexName, filename string) error {
	req := map[string]any{"query": map[string]any{"term": map[string]any{"filename": filename}}}
	body, _ := json.Marshal(req)
	res, err := e.client.DeleteByQuery([]string{indexName}, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("delete file: %w", err)
	}
	res.Body.Close()
	return nil
}

// DocInfo is a summary of a document for listing.
type DocInfo struct {
	ID          string `json:"id"`
	Content     string `json:"content"`
	Source      string `json:"source"`
	Filename    string `json:"filename"`
	FileType    string `json:"file_type"`
	ChunkCount  int    `json:"chunk_count"`
	ProcessedAt string `json:"processed_at"`
}

// DocumentList is the response for list-documents.
type DocumentList struct {
	Total     int       `json:"total"`
	Documents []DocInfo `json:"documents"`
	Limit     int       `json:"limit"`
	Offset    int       `json:"offset"`
}

// FileInfo is the response for list-files.
type FileInfo struct {
	Filename    string `json:"filename"`
	Source      string `json:"source"`
	FileType    string `json:"file_type"`
	ChunkCount  int    `json:"chunk_count"`
	TotalChars  int    `json:"total_chars"`
	DocumentID  string `json:"document_id"`
	ProcessedAt string `json:"processed_at"`
}

func truncate(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen]) + "..."
}
