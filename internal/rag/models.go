// Package rag provides RAG pipeline components: indexing, retrieval, and chat.
package rag

import (
	"context"
	"crypto/md5"
	"fmt"
	"time"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
)

// Embedder is eino's embedding.Embedder (re-exported for convenience).
type Embedder = embedding.Embedder

// LLMGenerator is eino's model.BaseChatModel (re-exported for convenience).
type LLMGenerator = model.BaseChatModel

// Chunk represents a document chunk to be indexed.
type Chunk struct {
	ID          string
	DocumentID  string
	ChunkIndex  int
	TotalChunks int
	Source      string
	Filename    string
	FileType    string
	Content     string
}

// SearchHit represents a search result from ES.
type SearchHit struct {
	ChunkID         string  `json:"chunk_id"`
	DocumentID      string  `json:"document_id"`
	Score           float64 `json:"score"`
	VectorScore     float64 `json:"vector_score,omitempty"`
	KeywordScore    float64 `json:"keyword_score,omitempty"`
	RetrievalMethod string  `json:"retrieval_method,omitempty"`
	Source          string  `json:"source"`
	Filename        string  `json:"filename"`
	Content         string  `json:"content"`
	ChunkIndex      int     `json:"chunk_index"`
}

// GenerateChunkID creates a deterministic chunk ID from document ID and index.
func GenerateChunkID(docID string, index int) string {
	return fmt.Sprintf("%s_chunk_%04d", docID, index)
}

// GenerateDocID creates a deterministic document ID from content hash.
func GenerateDocID(content string) string {
	return fmt.Sprintf("%x", md5.Sum([]byte(content)))
}

// Searcher defines the interface for searching indexed documents.
type Searcher interface {
	Search(ctx context.Context, queryVector []float32, topK int, minScore float64) ([]SearchHit, error)
	SearchHybrid(ctx context.Context, query string, queryVector []float32, topK int, minScore float64) ([]SearchHit, error)
	SearchWithMode(ctx context.Context, query string, queryVector []float32, topK int, minScore float64, mode string) ([]SearchHit, error)
}

// HealthChecker wraps a health check for external services.
type HealthChecker interface {
	HealthCheck(ctx context.Context) error
}

var Now = time.Now
