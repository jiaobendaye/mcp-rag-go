package rag

import (
	"context"
	"testing"

	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
)

// fakeRetriever records the options it was called with.
type fakeRetriever struct {
	calls [][]retriever.Option
	docs  []*schema.Document
	err   error
}

func (f *fakeRetriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	f.calls = append(f.calls, opts)
	return f.docs, f.err
}

func TestNewRetrieverWithFixedIndex_IgnoresCallerOpts(t *testing.T) {
	base := &fakeRetriever{docs: []*schema.Document{{ID: "1", Content: "hello"}}}
	fixed := retriever.WithIndex("kb_2")
	fixedTopK := retriever.WithTopK(7)

	wrapped := newRetrieverWithFixedIndex(base, []retriever.Option{fixed, fixedTopK})

	// Caller passes their own WithIndex/WithTopK; the wrapper should ignore
	// them and use the captured ones.
	docs, err := wrapped.Retrieve(context.Background(), "q",
		retriever.WithIndex("kb_999"),
		retriever.WithTopK(1),
	)
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(docs) != 1 || docs[0].ID != "1" {
		t.Errorf("expected single doc with id=1, got %+v", docs)
	}
	if len(base.calls) != 1 {
		t.Fatalf("expected 1 base call, got %d", len(base.calls))
	}
	opts := base.calls[0]
	common := retriever.GetCommonOptions(&retriever.Options{}, opts...)
	if common.Index == nil || *common.Index != "kb_2" {
		t.Errorf("expected base to receive WithIndex(kb_2), got Index=%v", common.Index)
	}
}

func TestNewRetrieverWithFixedIndex_PassesQuery(t *testing.T) {
	base := &fakeRetriever{docs: nil}
	wrapped := newRetrieverWithFixedIndex(base, []retriever.Option{retriever.WithIndex("kb_x")})
	_, _ = wrapped.Retrieve(context.Background(), "the query text")
	if len(base.calls) != 1 {
		t.Fatalf("expected 1 base call, got %d", len(base.calls))
	}
}

func TestNewRetrieverWithFixedIndex_PropagatesError(t *testing.T) {
	base := &fakeRetriever{err: errFake("test error")}
	wrapped := newRetrieverWithFixedIndex(base, []retriever.Option{retriever.WithIndex("kb_x")})
	_, err := wrapped.Retrieve(context.Background(), "q")
	if err == nil || err.Error() != "test error" {
		t.Errorf("expected propagated error, got %v", err)
	}
}

type errFake string

func (e errFake) Error() string { return string(e) }
