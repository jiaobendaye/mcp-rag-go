package rag

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/cloudwego/eino/schema"
	"github.com/elastic/go-elasticsearch/v8/typedapi/types"
)

// ProjectResultParser returns a ResultParser callback for
// eino-ext/components/retriever/es8. It wraps the eino-ext default
// parser (which extracts `content` as the Document.Content and stores
// the remaining source fields plus `score` in MetaData) and additionally
// extracts `chunk_id` from hit.Source_ into doc.MetaData["chunk_id"] so
// downstream consumers can correlate hits to the original ChunkID
// (matches the legacy ESRetriever behavior).
func ProjectResultParser() func(ctx context.Context, hit types.Hit) (*schema.Document, error) {
	return func(ctx context.Context, hit types.Hit) (*schema.Document, error) {
		doc, err := defaultResultParser(ctx, hit)
		if err != nil {
			return nil, err
		}
		if hit.Source_ != nil {
			var src map[string]any
			if err := json.Unmarshal(hit.Source_, &src); err == nil {
				if cid, ok := src["chunk_id"].(string); ok && cid != "" {
					if doc.MetaData == nil {
						doc.MetaData = make(map[string]any)
					}
					doc.MetaData["chunk_id"] = cid
				}
			}
		}
		return doc, nil
	}
}

// defaultResultParser is a local re-implementation of the eino-ext
// defaultResultParser. We don't import the unexported symbol from
// eino-ext, so we mirror its behavior: read content as string, store
// the rest of source plus score in MetaData. If the id or content
// fields are missing, return an error.
func defaultResultParser(ctx context.Context, hit types.Hit) (*schema.Document, error) {
	if hit.Id_ == nil {
		return nil, fmt.Errorf("defaultResultParser: field '_id' not found in hit")
	}
	id := *hit.Id_

	score := 0.0
	if hit.Score_ != nil {
		score = float64(*hit.Score_)
	}

	if hit.Source_ == nil {
		return nil, fmt.Errorf("defaultResultParser: field '_source' not found in document %s", id)
	}

	var source map[string]any
	if err := json.Unmarshal(hit.Source_, &source); err != nil {
		return nil, fmt.Errorf("defaultResultParser: unmarshal document content failed: %v", err)
	}

	val, ok := source["content"]
	if !ok {
		return nil, fmt.Errorf("defaultResultParser: field 'content' not found in document %s", id)
	}
	content, ok := val.(string)
	if !ok {
		return nil, fmt.Errorf("defaultResultParser: field 'content' in document %s is not a string", id)
	}

	meta := make(map[string]any, len(source)+1)
	for k, v := range source {
		if k != "content" {
			meta[k] = v
		}
	}
	meta["score"] = score

	doc := &schema.Document{
		ID:       id,
		Content:  content,
		MetaData: meta,
	}
	return doc.WithScore(score), nil
}
