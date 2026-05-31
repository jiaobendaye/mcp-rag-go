package rag

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Embedder is the interface for generating vector embeddings.
// This mirrors the eino embedding.Embedder interface.
type Embedder interface {
	EmbedStrings(ctx context.Context, texts []string) ([][]float64, error)
}

// IndexPipeline orchestrates the document indexing flow:
//
//	parse → split → embed → index
type IndexPipeline struct {
	embedder  Embedder
	indexer   Indexer
	chunkSize int
	overlap   int
}

// NewIndexPipeline creates a new IndexPipeline.
func NewIndexPipeline(embedder Embedder, indexer Indexer, chunkSize, overlap int) *IndexPipeline {
	if chunkSize <= 0 {
		chunkSize = 4000
	}
	if overlap < 0 {
		overlap = 200
	}
	return &IndexPipeline{
		embedder:  embedder,
		indexer:   indexer,
		chunkSize: chunkSize,
		overlap:   overlap,
	}
}

// IndexText indexes raw text content.
func (p *IndexPipeline) IndexText(ctx context.Context, content, source string) (*IndexResult, error) {
	if strings.TrimSpace(content) == "" {
		return nil, fmt.Errorf("content is empty")
	}

	// 1. Generate document ID
	docID := GenerateDocID(content)
	filename := source
	if filename == "" {
		filename = "manual_input"
	}
	fileType := "text"

	// 2. Split into chunks
	chunkTexts := splitText(content, p.chunkSize, p.overlap)
	if len(chunkTexts) == 0 {
		return nil, fmt.Errorf("no content extracted")
	}

	// 3. Create chunk records
	chunks := make([]Chunk, len(chunkTexts))
	for i, text := range chunkTexts {
		chunks[i] = Chunk{
			ID:          GenerateChunkID(docID, i),
			DocumentID:  docID,
			ChunkIndex:  i,
			TotalChunks: len(chunkTexts),
			Source:      source,
			Filename:    filename,
			FileType:    fileType,
			Content:     text,
		}
	}

	// 4. Generate embeddings
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Content
	}

	vectors64, err := p.embedder.EmbedStrings(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("embed chunks: %w", err)
	}

	// Convert float64 to float32
	vectors := make([][]float32, len(vectors64))
	for i, v := range vectors64 {
		vectors[i] = make([]float32, len(v))
		for j, f := range v {
			vectors[i][j] = float32(f)
		}
	}

	// 5. Index into ES
	if err := p.indexer.IndexChunks(ctx, chunks, vectors); err != nil {
		return nil, fmt.Errorf("index chunks: %w", err)
	}

	return &IndexResult{
		DocumentID: docID,
		ChunkCount: len(chunks),
	}, nil
}

// IndexFile indexes a file from disk.
// Supported formats: .txt, .md (plain text). PDF/DOCX require eino-ext parsers.
func (p *IndexPipeline) IndexFile(ctx context.Context, filePath string) (*IndexResult, error) {
	// Read file content
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("read file %s: %w", filePath, err)
	}

	content := string(data)
	filename := filepath.Base(filePath)

	return p.IndexText(ctx, content, filename)
}

// IndexResult contains the result of an indexing operation.
type IndexResult struct {
	DocumentID string `json:"document_id"`
	ChunkCount int    `json:"chunk_count"`
}

// splitText splits text into overlapping chunks using recursive character splitting.
func splitText(text string, chunkSize, overlap int) []string {
	if len(text) <= chunkSize {
		return []string{text}
	}

	// Try to split on paragraph boundaries first, then on sentences, then on characters
	separators := []string{"\n\n", "\n", "。", ". ", " ", ""}
	return splitRecursive(text, chunkSize, overlap, separators)
}

func splitRecursive(text string, chunkSize, overlap int, separators []string) []string {
	if len(separators) == 0 {
		// Last resort: character split
		return splitByChars(text, chunkSize, overlap)
	}

	sep := separators[0]
	if sep == "" || !strings.Contains(text, sep) {
		return splitRecursive(text, chunkSize, overlap, separators[1:])
	}

	// Split on separator
	var chunks []string
	current := ""
	parts := splitWithSep(text, sep)

	for _, part := range parts {
		partWithSep := part
		if current != "" {
			partWithSep = sep + part
		}

		if len(current)+len(partWithSep) <= chunkSize || current == "" {
			current += partWithSep
		} else {
			if strings.TrimSpace(current) != "" {
				chunks = append(chunks, current)
			}
			// Start new chunk with overlap
			if overlap > 0 && len(current) > overlap {
				current = current[len(current)-overlap:]
				// Don't break in the middle of a separator
				if idx := strings.Index(current, sep); idx >= 0 {
					current = current[idx+len(sep):]
				}
			} else {
				current = ""
			}
			current += partWithSep
		}
	}

	if strings.TrimSpace(current) != "" {
		chunks = append(chunks, current)
	}

	// If any chunk is still too large, recurse with next separator
	var result []string
	for _, chunk := range chunks {
		if len(chunk) > chunkSize {
			result = append(result, splitRecursive(chunk, chunkSize, overlap, separators[1:])...)
		} else {
			result = append(result, chunk)
		}
	}

	return result
}

func splitWithSep(text, sep string) []string {
	parts := strings.Split(text, sep)
	var result []string
	for i, p := range parts {
		if i == 0 {
			if p != "" {
				result = append(result, p)
			}
		} else {
			result = append(result, p)
		}
	}
	return result
}

func splitByChars(text string, chunkSize, overlap int) []string {
	if len(text) <= chunkSize {
		return []string{text}
	}

	runes := []rune(text)
	var chunks []string
	step := chunkSize - overlap
	if step <= 0 {
		step = chunkSize
	}

	for i := 0; i < len(runes); i += step {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
	}

	return chunks
}
