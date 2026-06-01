package rag

import (
	"context"
	"fmt"
	"time"

	"github.com/cloudwego/eino/schema"

	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
)

// contentVectorField is the dense_vector field name used for the chunk
// embedding. It must match the mapping created by EnsureIndex and the
// field referenced by BuildHybridQueryJSON.
const contentVectorField = "content_vector"

// ProjectDocumentToFields returns the DocumentToFields callback used by
// eino-ext's indexer/es8 to project an eino *schema.Document into the
// ES field set. The EmbedKey on the content field tells eino-ext to
// call Embedding.EmbedStrings on the content and store the result under
// the dense_vector field.
func ProjectDocumentToFields() func(ctx context.Context, doc *schema.Document) (map[string]elastic_indexer.FieldValue, error) {
	return func(ctx context.Context, doc *schema.Document) (map[string]elastic_indexer.FieldValue, error) {
		if doc == nil {
			return nil, fmt.Errorf("ProjectDocumentToFields: nil doc")
		}
		meta := doc.MetaData
		if meta == nil {
			meta = map[string]any{}
		}
		processedAt := stringMeta(meta, "processed_at", "")
		if processedAt == "" {
			// Stamp with current time in RFC3339 form. The mapping
			// declares `processed_at` as `date`; an empty string
			// would be rejected by ES on insert.
			processedAt = time.Now().UTC().Format(time.RFC3339)
		}
		return map[string]elastic_indexer.FieldValue{
			"content":      {Value: doc.Content, EmbedKey: contentVectorField},
			"document_id":  {Value: stringMeta(meta, "document_id", doc.ID)},
			"chunk_index":  {Value: intMeta(meta, "chunk_index", 0)},
			"total_chunks": {Value: intMeta(meta, "total_chunks", 1)},
			"source":       {Value: stringMeta(meta, "source", doc.ID)},
			"filename":     {Value: stringMeta(meta, "filename", "unknown")},
			"file_type":    {Value: stringMeta(meta, "file_type", "text")},
			"processed_at": {Value: processedAt},
		}, nil
	}
}

func stringMeta(m map[string]any, key, defaultVal string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return defaultVal
}

func intMeta(m map[string]any, key string, defaultVal int) int {
	if v, ok := m[key]; ok {
		switch n := v.(type) {
		case int:
			return n
		case int32:
			return int(n)
		case int64:
			return int(n)
		case float32:
			return int(n)
		case float64:
			return int(n)
		}
	}
	return defaultVal
}
