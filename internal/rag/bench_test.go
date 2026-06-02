package rag

import (
	"context"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"

	elasticsearch "github.com/elastic/go-elasticsearch/v8"
)

func BenchmarkBuildIndexChain(b *testing.B) {
	// Per-request BuildIndexChain calls elastic_indexer.NewIndexer which
	// requires a real ES client. Benchmark only when ES is available.
	b.Skip("requires ES client for NewIndexer inside BuildIndexChain")
}

func BenchmarkBuildRetrievalGraph(b *testing.B) {
	ctx := context.Background()
	llm := &benchLLM{}
	emb := &benchEmbedder{}
	esClient := &elasticsearch.Client{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := BuildRetrievalGraph(ctx, esClient, llm, emb, []string{"kb_2"}, 7, 0.6, "hybrid")
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
