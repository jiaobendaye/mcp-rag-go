package rag

import (
	"context"

	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/indexer"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"

	"github.com/cloudwego/eino-ext/components/document/loader/file"
	"github.com/cloudwego/eino-ext/components/document/transformer/splitter/recursive"
)

// BuildIndexChain creates an eino compose.Chain for document indexing.
//
//	Source → Loader → Splitter → [extract text + embed] → Indexer
//
// The Lambda bridges the gap between Splitter ([]*schema.Document) and
// Indexer ([]*schema.Document), injecting the embedding vectors.
func BuildIndexChain(ctx context.Context, emb embedding.Embedder, idx indexer.Indexer, chunkSize, overlap int) (compose.Runnable[document.Source, []string], error) {
	loader, err := file.NewFileLoader(ctx, &file.FileLoaderConfig{})
	if err != nil {
		return nil, err
	}
	splitter, err := recursive.NewSplitter(ctx, &recursive.Config{
		ChunkSize:   chunkSize,
		OverlapSize: overlap,
	})
	if err != nil {
		return nil, err
	}

	// Lambda: embed text and attach vectors to documents
	embedDocs := compose.InvokableLambda(func(ctx context.Context, docs []*schema.Document) ([]*schema.Document, error) {
		texts := make([]string, len(docs))
		for i, d := range docs {
			texts[i] = d.Content
		}
		vecs, err := emb.EmbedStrings(ctx, texts)
		if err != nil {
			return nil, err
		}
		for i, v := range vecs {
			if docs[i].MetaData == nil {
				docs[i].MetaData = make(map[string]any)
			}
			docs[i].MetaData["_embedding"] = v
		}
		return docs, nil
	})

	chain := compose.NewChain[document.Source, []string]()
	chain.
		AppendLoader(loader).
		AppendDocumentTransformer(splitter).
		AppendLambda(embedDocs).
		AppendIndexer(idx)

	return chain.Compile(ctx)
}
