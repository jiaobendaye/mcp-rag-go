package rag

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino/compose"

	"github.com/cloudwego/eino-ext/components/document/loader/file"
	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
)

// BuildIndexChain returns a compose.Runnable that loads a file, splits it
// into chunks, and indexes them into the named ES index. The chain is built
// per-request because indexing is infrequent; the compile overhead (~12µs)
// is negligible compared to file I/O, splitting, embedding, and ES writes.
//
// The chain topology is:
//
//	Source → FileLoader → RecursiveSplitter → AppendIndexer(idx)
//
// The eino-ext Indexer handles embedding internally — DocumentToFields
// returns EmbedKey: "content_vector" on the content field, so eino-ext's
// bulkAdd calls Embedding.EmbedStrings for us.
func BuildIndexChain(
	ctx context.Context,
	splitter document.Transformer,
	indexerConf *elastic_indexer.IndexerConfig,
	indexName string,
) (compose.Runnable[document.Source, []string], error) {
	if indexerConf == nil {
		return nil, fmt.Errorf("BuildIndexChain: nil indexerConf")
	}
	if splitter == nil {
		return nil, fmt.Errorf("BuildIndexChain: nil splitter")
	}

	// Create a fresh eino-ext indexer for the target index.
	confCopy := *indexerConf
	confCopy.Index = indexName
	idx, err := elastic_indexer.NewIndexer(ctx, &confCopy)
	if err != nil {
		return nil, fmt.Errorf("BuildIndexChain: create indexer for %q: %w", indexName, err)
	}

	loader, err := file.NewFileLoader(ctx, &file.FileLoaderConfig{})
	if err != nil {
		return nil, fmt.Errorf("BuildIndexChain: create loader: %w", err)
	}

	chain := compose.NewChain[document.Source, []string]()
	chain.
		AppendLoader(loader).
		AppendDocumentTransformer(splitter).
		AppendIndexer(idx)

	runnable, err := chain.Compile(ctx)
	if err != nil {
		return nil, fmt.Errorf("BuildIndexChain: compile: %w", err)
	}
	return runnable, nil
}
