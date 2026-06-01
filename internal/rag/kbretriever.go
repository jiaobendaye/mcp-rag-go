package rag

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"

	elastic_retriever "github.com/cloudwego/eino-ext/components/retriever/es8"
)

// KBRetriever wraps eino-ext's *elastic_retriever.Retriever to support
// per-call index routing. eino-ext's underlying retriever uses
// r.config.Index in the search call rather than honoring the WithIndex
// option at the search call site, so the wrapper rebuilds a fresh
// underlying retriever per call when a different index is requested.
//
// Like KBIndexer, KBRetriever is constructed once at startup with a
// placeholder Index; callers pass retriever.WithIndex("kb_2") on each
// Retrieve call.
type KBRetriever struct {
	base *elastic_retriever.Retriever
	conf *elastic_retriever.RetrieverConfig
}

// NewKBRetriever creates a new KBRetriever from the given RetrieverConfig.
// The caller is responsible for setting Index (typically to
// PlaceholderIndex) and providing Client, SearchMode, ResultParser, and
// Embedding.
func NewKBRetriever(ctx context.Context, conf *elastic_retriever.RetrieverConfig) (*KBRetriever, error) {
	if conf == nil {
		return nil, fmt.Errorf("NewKBRetriever: nil config")
	}
	if conf.Index == "" {
		conf.Index = PlaceholderIndex
	}
	base, err := elastic_retriever.NewRetriever(ctx, conf)
	if err != nil {
		return nil, fmt.Errorf("NewKBRetriever: %w", err)
	}
	return &KBRetriever{base: base, conf: conf}, nil
}

// Retrieve performs a search, routing to the per-call index if
// retriever.WithIndex was provided. If no WithIndex is provided AND the
// bound config index is the placeholder, an error is returned.
func (k *KBRetriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	targetIdx := k.resolveIndex(opts)

	if targetIdx == "" {
		return nil, fmt.Errorf("KBRetriever.Retrieve: empty target index")
	}
	if targetIdx == PlaceholderIndex {
		return nil, fmt.Errorf("KBRetriever.Retrieve: retriever.WithIndex is required when bound config.Index is the placeholder")
	}

	if targetIdx == k.conf.Index {
		return k.base.Retrieve(ctx, query, opts...)
	}

	// Slow path: rebuild a fresh retriever bound to the per-call index.
	confCopy := *k.conf
	confCopy.Index = targetIdx
	r, err := elastic_retriever.NewRetriever(ctx, &confCopy)
	if err != nil {
		return nil, fmt.Errorf("KBRetriever.Retrieve: build retriever for %q: %w", targetIdx, err)
	}
	return r.Retrieve(ctx, query, opts...)
}

func (k *KBRetriever) resolveIndex(opts []retriever.Option) string {
	base := &retriever.Options{Index: &k.conf.Index}
	common := retriever.GetCommonOptions(base, opts...)
	if common.Index != nil && *common.Index != "" {
		return *common.Index
	}
	return k.conf.Index
}
