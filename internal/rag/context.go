package rag

import "context"

type kbParamsKey struct{}

// KBParams holds per-request knowledge base parameters injected into
// context via WithKBParams and read by the pre-compiled graph/chain.
type KBParams struct {
	IndexNames []string // per-request KB index names (single KB: []string{indexName})
	TopK       int
	MinScore   float64
	SearchMode string
}

// WithKBParams injects KBParams into the context for consumption by
// pre-compiled retrieval graphs and index chains.
func WithKBParams(ctx context.Context, p KBParams) context.Context {
	return context.WithValue(ctx, kbParamsKey{}, p)
}

// GetKBParams extracts KBParams from context if present.
func GetKBParams(ctx context.Context) (KBParams, bool) {
	p, ok := ctx.Value(kbParamsKey{}).(KBParams)
	return p, ok
}
