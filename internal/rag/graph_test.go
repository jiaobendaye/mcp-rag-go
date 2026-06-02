package rag

import (
	"context"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	elastic_retriever "github.com/cloudwego/eino-ext/components/retriever/es8"
)

func TestBuildRetrievalGraph_RejectsNilRetriever(t *testing.T) {
	llm := &stubLLM{}
	emb := &stubEmbedder{}
	_, err := BuildRetrievalGraph(context.Background(), nil, llm, emb)
	if err == nil || !strings.Contains(err.Error(), "nil kbRetriever") {
		t.Errorf("expected nil-retriever error, got %v", err)
	}
}

func TestBuildRetrievalGraph_RejectsNilLLM(t *testing.T) {
	emb := &stubEmbedder{}
	k := &KBRetriever{base: nil, conf: &elastic_retriever.RetrieverConfig{Index: "x"}}
	_, err := BuildRetrievalGraph(context.Background(), k, nil, emb)
	if err == nil || !strings.Contains(err.Error(), "nil llm") {
		t.Errorf("expected nil-llm error, got %v", err)
	}
}

func TestBuildRetrievalGraph_RejectsNilEmbedder(t *testing.T) {
	llm := &stubLLM{}
	k := &KBRetriever{base: nil, conf: &elastic_retriever.RetrieverConfig{Index: "x"}}
	_, err := BuildRetrievalGraph(context.Background(), k, llm, nil)
	if err == nil || !strings.Contains(err.Error(), "nil embedder") {
		t.Errorf("expected nil-embedder error, got %v", err)
	}
}

// TestBuildRetrievalGraph_CompilesOnce verifies that the graph compiles
// successfully without per-request KB params. The graph should be
// reusable across requests with different KB parameters passed via
// WithKBParams context.
func TestBuildRetrievalGraph_CompilesOnce(t *testing.T) {
	llm := &stubLLM{}
	emb := &stubEmbedder{}
	k := &KBRetriever{
		base: nil,
		conf: &elastic_retriever.RetrieverConfig{Index: PlaceholderIndex},
	}
	runnable, err := BuildRetrievalGraph(context.Background(), k, llm, emb)
	if err != nil {
		t.Fatalf("BuildRetrievalGraph: %v", err)
	}
	if runnable == nil {
		t.Fatal("expected non-nil runnable")
	}

	// Verify the graph is reusable: invoke with KBParams via context.
	ctx := WithKBParams(context.Background(), KBParams{
		IndexNames: []string{"kb_2"},
		TopK:       7,
		MinScore:   0.6,
		SearchMode: "hybrid",
	})
	// We can't actually invoke the graph (the retriever is nil-backed
	// so it would panic), but we can verify the params flow through
	// by checking GetKBParams round-trips.
	kb, ok := GetKBParams(ctx)
	if !ok {
		t.Fatal("GetKBParams returned false")
	}
	if len(kb.IndexNames) != 1 || kb.IndexNames[0] != "kb_2" || kb.TopK != 7 || kb.MinScore != 0.6 || kb.SearchMode != "hybrid" {
		t.Errorf("KBParams round-trip mismatch: %+v", kb)
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
