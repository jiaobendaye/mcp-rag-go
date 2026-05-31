package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"

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

// Search performs KNN vector search on the content_vector field.
func (e *ES8Indexer) Search(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error) {
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
