package rag

import (
	"context"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"

	elasticsearch "github.com/elastic/go-elasticsearch/v8"
)

func TestBuildRetrievalGraph_RejectsNilESClient(t *testing.T) {
	llm := &stubLLM{}
	emb := &stubEmbedder{}
	_, err := BuildRetrievalGraph(context.Background(), nil, llm, emb, []string{"kb_1"}, 5, 0.5, "hybrid")
	if err == nil || !strings.Contains(err.Error(), "nil esClient") {
		t.Errorf("expected nil-esClient error, got %v", err)
	}
}

func TestBuildRetrievalGraph_RejectsNilLLM(t *testing.T) {
	emb := &stubEmbedder{}
	esClient := &elasticsearch.Client{}
	_, err := BuildRetrievalGraph(context.Background(), esClient, nil, emb, []string{"kb_1"}, 5, 0.5, "hybrid")
	if err == nil || !strings.Contains(err.Error(), "nil llm") {
		t.Errorf("expected nil-llm error, got %v", err)
	}
}

func TestBuildRetrievalGraph_RejectsNilEmbedder(t *testing.T) {
	llm := &stubLLM{}
	esClient := &elasticsearch.Client{}
	_, err := BuildRetrievalGraph(context.Background(), esClient, llm, nil, []string{"kb_1"}, 5, 0.5, "hybrid")
	if err == nil || !strings.Contains(err.Error(), "nil embedder") {
		t.Errorf("expected nil-embedder error, got %v", err)
	}
}

func TestBuildRetrievalGraph_RejectsEmptyIndexNames(t *testing.T) {
	llm := &stubLLM{}
	emb := &stubEmbedder{}
	esClient := &elasticsearch.Client{}
	_, err := BuildRetrievalGraph(context.Background(), esClient, llm, emb, nil, 5, 0.5, "hybrid")
	if err == nil || !strings.Contains(err.Error(), "empty indexNames") {
		t.Errorf("expected empty-indexNames error, got %v", err)
	}
}

// TestBuildRetrievalGraph_CompilesPerRequest verifies that the graph compiles
// per-request with index names and parameters baked in directly.
func TestBuildRetrievalGraph_CompilesPerRequest(t *testing.T) {
	llm := &stubLLM{}
	emb := &stubEmbedder{}
	esClient := &elasticsearch.Client{}

	// Single index
	r1, err := BuildRetrievalGraph(context.Background(), esClient, llm, emb, []string{"kb_1"}, 7, 0.6, "hybrid")
	if err != nil {
		t.Fatalf("BuildRetrievalGraph (single): %v", err)
	}
	if r1 == nil {
		t.Fatal("expected non-nil runnable (single)")
	}

	// Multi index
	r2, err := BuildRetrievalGraph(context.Background(), esClient, llm, emb, []string{"kb_a", "kb_b"}, 7, 0.6, "hybrid")
	if err != nil {
		t.Fatalf("BuildRetrievalGraph (multi): %v", err)
	}
	if r2 == nil {
		t.Fatal("expected non-nil runnable (multi)")
	}
}

// stubLLM is a minimal eino model.BaseChatModel for graph compile.
type stubLLM struct{}

func (s *stubLLM) Generate(ctx context.Context, in []*schema.Message, _ ...model.Option) (*schema.Message, error) {
	return &schema.Message{Role: schema.Assistant, Content: "ok"}, nil
}
func (s *stubLLM) Stream(ctx context.Context, in []*schema.Message, _ ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	return nil, nil
}

// stubEmbedder is a minimal eino embedding.Embedder for graph compile. It
// returns a 4-dim zero vector — enough to satisfy the build_query step.
type stubEmbedder struct{}

func (s *stubEmbedder) EmbedStrings(ctx context.Context, texts []string, _ ...embedding.Option) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i := range texts {
		out[i] = []float64{0.1, 0.2, 0.3, 0.4}
	}
	return out, nil
}
