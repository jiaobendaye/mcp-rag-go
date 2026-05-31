// Package rag provides RAG pipeline components: indexing, retrieval, and chat.
package rag

import (
	"context"
	"crypto/md5"
	"fmt"
	"time"
)

// Chunk represents a document chunk to be indexed.
type Chunk struct {
	ID          string  // unique chunk ID
	DocumentID  string  // parent document ID
	ChunkIndex  int     // position in document
	TotalChunks int     // total chunks in document
	Source      string  // origin filename or URL
	Filename    string  // display filename
	FileType    string  // file extension
	Content     string  // chunk text content
}

// SearchHit represents a search result from ES.
type SearchHit struct {
	ChunkID    string  `json:"chunk_id"`
	DocumentID string  `json:"document_id"`
	Score      float64 `json:"score"`
	Source     string  `json:"source"`
	Filename   string  `json:"filename"`
	Content    string  `json:"content"`
	ChunkIndex int     `json:"chunk_index"`
}

// GenerateChunkID creates a deterministic chunk ID from document ID and index.
func GenerateChunkID(docID string, index int) string {
	return fmt.Sprintf("%s_chunk_%04d", docID, index)
}

// GenerateDocID creates a deterministic document ID from content hash.
func GenerateDocID(content string) string {
	hash := md5.Sum([]byte(content))
	return fmt.Sprintf("%x", hash)
}

// Indexer defines the interface for indexing chunks.
type Indexer interface {
	// EnsureIndex creates the ES index with proper mapping if it doesn't exist.
	EnsureIndex(ctx context.Context, dims int) error

	// IndexChunks bulk-indexes chunks with their embedding vectors.
	IndexChunks(ctx context.Context, chunks []Chunk, vectors [][]float32) error
}

// Searcher defines the interface for searching indexed documents.
type Searcher interface {
	// Search performs KNN vector search (backward compat).
	Search(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error)

	// SearchHybrid performs hybrid search based on mode: "hybrid" | "rrf" | "knn".
	SearchHybrid(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error)

	// SearchWithMode performs search with explicit mode selection.
	SearchWithMode(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error)
}

// HealthChecker wraps a health check for external services.
type HealthChecker interface {
	HealthCheck(ctx context.Context) error
}

// Now is a testable clock function.
var Now = time.Now
