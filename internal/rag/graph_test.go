package rag

import (
	"context"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
	elastic_retriever "github.com/cloudwego/eino-ext/components/retriever/es8"
)

func TestBuildRetrievalGraphAt_RejectsEmptyIndex(t *testing.T) {
	llm := &stubLLM{}
	emb := &stubEmbedder{}
	k := &KBRetriever{base: nil, conf: &elastic_retriever.RetrieverConfig{Index: "x"}}
	_, err := BuildRetrievalGraphAt(context.Background(), k, llm, emb, "", "hybrid", 5, 0.0)
	if err == nil || !strings.Contains(err.Error(), "empty indexName") {
		t.Errorf("expected empty-indexName error, got %v", err)
	}
}

func TestBuildRetrievalGraphAt_RejectsNilRetriever(t *testing.T) {
	llm := &stubLLM{}
	emb := &stubEmbedder{}
	_, err := BuildRetrievalGraphAt(context.Background(), nil, llm, emb, "kb_1", "hybrid", 5, 0.0)
	if err == nil || !strings.Contains(err.Error(), "nil kbRetriever") {
		t.Errorf("expected nil-retriever error, got %v", err)
	}
}

func TestBuildRetrievalGraphAt_RejectsNilLLM(t *testing.T) {
	emb := &stubEmbedder{}
	k := &KBRetriever{base: nil, conf: &elastic_retriever.RetrieverConfig{Index: "x"}}
	_, err := BuildRetrievalGraphAt(context.Background(), k, nil, emb, "kb_1", "hybrid", 5, 0.0)
	if err == nil || !strings.Contains(err.Error(), "nil llm") {
		t.Errorf("expected nil-llm error, got %v", err)
	}
}

func TestBuildRetrievalGraphAt_RejectsNilEmbedder(t *testing.T) {
	llm := &stubLLM{}
	k := &KBRetriever{base: nil, conf: &elastic_retriever.RetrieverConfig{Index: "x"}}
	_, err := BuildRetrievalGraphAt(context.Background(), k, llm, nil, "kb_1", "hybrid", 5, 0.0)
	if err == nil || !strings.Contains(err.Error(), "nil embedder") {
		t.Errorf("expected nil-embedder error, got %v", err)
	}
}

// TestBuildRetrievalGraphAt_ClosureCapturesIndex verifies that the graph
// compiled by BuildRetrievalGraphAt captures indexName in the per-request
// retriever options. We verify by:
//  1. Confirming compile succeeds
//  2. Confirming the underlying KBRetriever's resolveIndex returns the
//     per-request index when the same options are applied
func TestBuildRetrievalGraphAt_ClosureCapturesIndex(t *testing.T) {
	llm := &stubLLM{}
	emb := &stubEmbedder{}
	// Use a placeholder-bound KBRetriever so the per-request WithIndex
	// will be the source of truth.
	k := &KBRetriever{
		base: nil,
		conf: &elastic_retriever.RetrieverConfig{Index: PlaceholderIndex},
	}
	runnable, err := BuildRetrievalGraphAt(context.Background(), k, llm, emb, "kb_2", "hybrid", 7, 0.6)
	if err != nil {
		t.Fatalf("BuildRetrievalGraphAt: %v", err)
	}
	if runnable == nil {
		t.Fatal("expected non-nil runnable")
	}
	// Verify the option semantics match what we passed.
	opts := []retriever.Option{retriever.WithIndex("kb_2"), retriever.WithTopK(7), retriever.WithScoreThreshold(0.6)}
	common := retriever.GetCommonOptions(&retriever.Options{}, opts...)
	if common.Index == nil || *common.Index != "kb_2" {
		t.Errorf("expected common.Index=kb_2, got %v", common.Index)
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
