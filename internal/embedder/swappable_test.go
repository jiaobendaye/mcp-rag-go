package embedder

import (
	"context"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
)

type testEmbedder struct {
	vectors [][]float64
	err     error
}

func (t *testEmbedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
	return t.vectors, t.err
}

func TestSwappableEmbedder_PassThrough(t *testing.T) {
	e := &testEmbedder{vectors: [][]float64{{0.1, 0.2, 0.3}}}
	s := NewSwappableEmbedder(e)

	vecs, err := s.EmbedStrings(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != 3 {
		t.Fatalf("unexpected vectors: %v", vecs)
	}
}

func TestSwappableEmbedder_Swap(t *testing.T) {
	e1 := &testEmbedder{vectors: [][]float64{{1.0, 1.0}}}
	e2 := &testEmbedder{vectors: [][]float64{{2.0, 2.0}}}
	s := NewSwappableEmbedder(e1)

	// Before swap
	vecs, _ := s.EmbedStrings(context.Background(), []string{"x"})
	if vecs[0][0] != 1.0 {
		t.Fatalf("expected 1.0 from e1, got %f", vecs[0][0])
	}

	// After swap
	s.Swap(e2)
	vecs, _ = s.EmbedStrings(context.Background(), []string{"x"})
	if vecs[0][0] != 2.0 {
		t.Fatalf("expected 2.0 from e2, got %f", vecs[0][0])
	}
}
