package rag

import (
	"context"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
	"github.com/cloudwego/eino-ext/components/document/transformer/splitter/recursive"
	elastic_retriever "github.com/cloudwego/eino-ext/components/retriever/es8"
)

func BenchmarkBuildIndexChainAt(b *testing.B) {
	ctx := context.Background()
	splitter, _ := recursive.NewSplitter(ctx, &recursive.Config{
		ChunkSize: 100, OverlapSize: 10,
	})
	k := &KBIndexer{
		base: nil,
		conf: &elastic_indexer.IndexerConfig{Index: PlaceholderIndex},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := BuildIndexChainAt(ctx, splitter, k, "kb_2")
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBuildRetrievalGraphAt(b *testing.B) {
	ctx := context.Background()
	llm := &benchLLM{}
	emb := &benchEmbedder{}
	k := &KBRetriever{
		base: nil,
		conf: &elastic_retriever.RetrieverConfig{Index: PlaceholderIndex},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := BuildRetrievalGraphAt(ctx, k, llm, emb, "kb_2", "hybrid", 5, 0.7)
		if err != nil {
			b.Fatal(err)
		}
	}
}

type benchLLM struct{}

func (b *benchLLM) Generate(ctx context.Context, in []*schema.Message, _ ...model.Option) (*schema.Message, error) {
	return &schema.Message{Role: schema.Assistant, Content: "ok"}, nil
}
func (b *benchLLM) Stream(ctx context.Context, in []*schema.Message, _ ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	return nil, nil
}

type benchEmbedder struct{}

func (b *benchEmbedder) EmbedStrings(ctx context.Context, texts []string, _ ...embedding.Option) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i := range texts {
		out[i] = []float64{0.1, 0.2, 0.3, 0.4}
	}
	return out, nil
}
