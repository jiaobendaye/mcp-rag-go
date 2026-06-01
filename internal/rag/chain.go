package rag

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"

	"github.com/cloudwego/eino-ext/components/document/loader/file"
)

// BuildIndexChainAt returns a compose.Runnable that writes the indexed
// chunks to the given ES index. Per-request compile: each call rebuilds
// a fresh chain with a `writeAt` lambda that closes over indexName, so
// the KBIndexer.Store call inside the lambda receives WithIndex(indexName).
//
// The chain topology is:
//
//	Source → FileLoader → RecursiveSplitter → Lambda(writeAt)
//
// The eino-ext KBIndexer handles embedding internally — DocumentToFields
// returns EmbedKey: "content_vector" on the content field, so eino-ext's
// bulkAdd calls Embedding.EmbedStrings for us. We don't need an explicit
// embed lambda in the chain.
//
// Per-request Compile cost is sub-ms for a 3-node linear graph.
func BuildIndexChainAt(
	ctx context.Context,
	splitter document.Transformer,
	kbIndexer *KBIndexer,
	indexName string,
) (compose.Runnable[document.Source, []string], error) {
	if indexName == "" {
		return nil, fmt.Errorf("BuildIndexChainAt: empty indexName")
	}
	if kbIndexer == nil {
		return nil, fmt.Errorf("BuildIndexChainAt: nil kbIndexer")
	}
	if splitter == nil {
		return nil, fmt.Errorf("BuildIndexChainAt: nil splitter")
	}

	loader, err := file.NewFileLoader(ctx, &file.FileLoaderConfig{})
	if err != nil {
		return nil, fmt.Errorf("BuildIndexChainAt: create loader: %w", err)
	}

	// Lambda closes over indexName — the per-request state.
	writeAt := compose.InvokableLambda(func(ctx context.Context, docs []*schema.Document) ([]string, error) {
		return kbIndexer.Store(ctx, docs, WithIndex(indexName))
	})

	chain := compose.NewChain[document.Source, []string]()
	chain.
		AppendLoader(loader).
		AppendDocumentTransformer(splitter).
		AppendLambda(writeAt)

	runnable, err := chain.Compile(ctx)
	if err != nil {
		return nil, fmt.Errorf("BuildIndexChainAt: compile: %w", err)
	}
	return runnable, nil
}
