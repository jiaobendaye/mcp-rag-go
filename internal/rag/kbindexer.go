package rag

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/components/indexer"
	"github.com/cloudwego/eino/schema"

	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
)

// KBIndexOptions is the project-specific option bag that the KBIndexer
// uses to extract per-call routing state (target ES index name).
type KBIndexOptions struct {
	Index string
}

// WithIndex returns an eino indexer.Option that sets the target ES index
// for the current Store call. The KBIndexer.Store method reads this
// option via indexer.GetImplSpecificOptions and routes the call to the
// named index.
func WithIndex(name string) indexer.Option {
	return indexer.WrapImplSpecificOptFn(func(o *KBIndexOptions) {
		o.Index = name
	})
}

// PlaceholderIndex is a sentinel name that signals "no real index bound
// at construction time". NewKBIndexer rewrites this to a benign-but-real
// name before constructing the underlying eino-ext indexer (which
// performs an IndicesExists check on the bound name and rejects names
// starting with "_" or empty strings).
const PlaceholderIndex = "__rag_placeholder__"

// KBIndexer wraps eino-ext's *elastic_indexer.Indexer to support per-call
// index routing. The wrapper is constructed once at startup with a
// placeholder Index; callers pass WithIndex("kb_2") on each Store call.
//
// The wrapper also exposes an EnsureIndexForKB helper used at startup to
// pre-create the default KB's index and from the create-knowledge-base
// handler to create new KBs on demand.
type KBIndexer struct {
	base *elastic_indexer.Indexer
	conf *elastic_indexer.IndexerConfig
}

// NewKBIndexer creates a new KBIndexer from the given IndexerConfig. The
// caller is responsible for setting Index (typically to PlaceholderIndex)
// and providing Embedding, DocumentToFields, IndexSpec, and Client.
func NewKBIndexer(ctx context.Context, conf *elastic_indexer.IndexerConfig) (*KBIndexer, error) {
	if conf == nil {
		return nil, fmt.Errorf("NewKBIndexer: nil config")
	}
	if conf.Index == "" {
		conf.Index = PlaceholderIndex
	}
	// eino-ext's NewIndexer requires a real ES-indexable name for the
	// bound Index (it does an IndicesExists HEAD on it). If the caller
	// passed PlaceholderIndex, swap in a benign-but-legal name so the
	// existence check passes; the placeholder is never used to write
	// because Store refuses to route to it.
	if conf.Index == PlaceholderIndex {
		conf.Index = "rag_placeholder_index"
	}
	base, err := elastic_indexer.NewIndexer(ctx, conf)
	if err != nil {
		return nil, fmt.Errorf("NewKBIndexer: %w", err)
	}
	return &KBIndexer{base: base, conf: conf}, nil
}

// Store writes documents to ES, routing to the per-call index if WithIndex
// was provided. If no WithIndex is provided AND the bound config index is
// the placeholder, an error is returned.
func (k *KBIndexer) Store(ctx context.Context, docs []*schema.Document, opts ...indexer.Option) ([]string, error) {
	targetIdx := k.resolveIndex(opts)
	if targetIdx == "" {
		return nil, fmt.Errorf("KBIndexer.Store: empty target index (no WithIndex provided and config.Index is empty)")
	}
	if targetIdx == PlaceholderIndex {
		return nil, fmt.Errorf("KBIndexer.Store: WithIndex is required when bound config.Index is the placeholder")
	}

	// Fast path: target matches the bound config — no rebuild needed.
	if targetIdx == k.conf.Index {
		return k.base.Store(ctx, docs, opts...)
	}

	// Slow path: copy the config, override the index, build a fresh indexer.
	confCopy := *k.conf
	confCopy.Index = targetIdx
	idx, err := elastic_indexer.NewIndexer(ctx, &confCopy)
	if err != nil {
		return nil, fmt.Errorf("KBIndexer.Store: build indexer for %q: %w", targetIdx, err)
	}
	return idx.Store(ctx, docs, opts...)
}

// EnsureIndexForKB creates the named index if it doesn't exist. It uses
// the wrapper's bound IndexerConfig.IndexSpec and Embedding (only Embedding
// dims are not needed; eino-ext's NewIndexer with IndexSpec creates the
// index using the spec verbatim).
func (k *KBIndexer) EnsureIndexForKB(ctx context.Context, indexName string) error {
	if k.conf.IndexSpec == nil {
		return fmt.Errorf("KBIndexer.EnsureIndexForKB: nil IndexSpec in bound config")
	}
	confCopy := *k.conf
	confCopy.Index = indexName
	if _, err := elastic_indexer.NewIndexer(ctx, &confCopy); err != nil {
		return fmt.Errorf("KBIndexer.EnsureIndexForKB: %w", err)
	}
	return nil
}

// resolveIndex returns the target index for a Store call, honoring the
// per-call WithIndex option if provided, otherwise falling back to the
// bound config.Index. Exposed for unit testing.
func (k *KBIndexer) resolveIndex(opts []indexer.Option) string {
	extra := indexer.GetImplSpecificOptions(&KBIndexOptions{}, opts...)
	if extra != nil && extra.Index != "" {
		return extra.Index
	}
	return k.conf.Index
}
