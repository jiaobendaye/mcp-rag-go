package rag

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
)

// ESRetriever implements retriever.Retriever by wrapping an embedder and a Searcher.
// It encapsulates the embedding + hybrid search flow so callers only need to pass a query string.
type ESRetriever struct {
	embedder   embedding.Embedder
	searcher   Searcher
	indexName  string
	searchMode string
	topK       int
	minScore   float64
}

// NewESRetriever creates a new ESRetriever with default retrieval parameters.
func NewESRetriever(embedder embedding.Embedder, searcher Searcher, indexName, searchMode string, topK int, minScore float64) *ESRetriever {
	return &ESRetriever{
		embedder:   embedder,
		searcher:   searcher,
		indexName:  indexName,
		searchMode: searchMode,
		topK:       topK,
		minScore:   minScore,
	}
}

// Retrieve embeds the query, performs hybrid search, and converts results to []*schema.Document.
func (r *ESRetriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	// 1. Parse options (override defaults)
	topK := r.topK
	minScore := r.minScore
	indexName := r.indexName

	base := &retriever.Options{}
	common := retriever.GetCommonOptions(base, opts...)
	if common.TopK != nil {
		topK = *common.TopK
	}
	if common.ScoreThreshold != nil {
		minScore = *common.ScoreThreshold
	}
	if common.Index != nil {
		indexName = *common.Index
	}
	_ = indexName // reserved for future multi-index support

	// 2. Embed the query
	vecs, err := r.embedder.EmbedStrings(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("embedding returned empty result for query")
	}

	// 3. Perform hybrid search
	hits, err := r.searcher.SearchWithMode(ctx, query, toFloat32(vecs[0]), topK, minScore, r.searchMode)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	// 4. Convert SearchHit to schema.Document
	docs := make([]*schema.Document, 0, len(hits))
	for _, hit := range hits {
		doc := &schema.Document{
			Content: hit.Content,
		}
		doc.MetaData = map[string]any{
			"filename":     hit.Filename,
			"source":       hit.Source,
			"score":        hit.Score,
			"chunk_index":  hit.ChunkIndex,
			"document_id":  hit.DocumentID,
			"chunk_id":     hit.ChunkID,
		}
		docs = append(docs, doc)
	}

	return docs, nil
}
