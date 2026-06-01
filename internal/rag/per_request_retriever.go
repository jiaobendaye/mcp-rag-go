package rag

import (
	"context"

	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
)

// fixedIndexRetriever is a small adapter that wraps a KBRetriever and
// freezes a set of retriever.Option values, ignoring any options the
// caller passes at Retrieve time. It exists so that
// compose.AddRetrieverNode (which takes a retriever.Retriever value,
// not a closure) can run a per-request retriever with closure-captured
// options.
type fixedIndexRetriever struct {
	base retriever.Retriever
	opts []retriever.Option
}

// Retrieve runs the wrapped retriever with the captured options,
// ignoring any caller-supplied options.
func (f *fixedIndexRetriever) Retrieve(ctx context.Context, query string, _ ...retriever.Option) ([]*schema.Document, error) {
	return f.base.Retrieve(ctx, query, f.opts...)
}

// newRetrieverWithFixedIndex returns a retriever.Retriever that ignores
// caller options and uses the captured opts instead. This is the adapter
// used by BuildRetrievalGraphAt to bind per-request retriever.WithIndex
// (and friends) at graph-construction time.
func newRetrieverWithFixedIndex(kbRetriever retriever.Retriever, opts []retriever.Option) retriever.Retriever {
	return &fixedIndexRetriever{base: kbRetriever, opts: opts}
}
