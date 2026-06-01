package rag

import (
	"encoding/json"
	"fmt"
)

// BuildHybridQueryJSON constructs the ES request body JSON used with
// SearchModeRawStringRequest.
//
//   - mode "hybrid": bool.should with both match (BM25) and knn clauses,
//     weighted by weights.Keyword and weights.Vector respectively.
//   - mode "knn"   : only a knn clause (vector weights ignored).
//   - mode "rrf"   : falls back to bool.should with both clauses (eino-ext
//     does not expose a native RRF mode; the design drops RRF).
//   - mode "keyword" (or anything else): only a match clause.
//
// topK is the number of hits to return, minScore filters by score, and
// vector/dims is the dense query vector (must be `dims` long).
func BuildHybridQueryJSON(query string, vector []float64, topK int, minScore float64, weights SearchWeights, mode string) (string, error) {
	if topK <= 0 {
		topK = 5
	}
	if minScore < 0 {
		minScore = 0
	}

	var body map[string]any

	switch mode {
	case "knn":
		knnClause := buildKNNClause(vector, topK, minScore, weights.Vector)
		body = map[string]any{
			"size": topK,
			"knn":  []any{knnClause},
		}
		if minScore > 0 {
			body["min_score"] = minScore
		}
	case "keyword", "bm25":
		body = map[string]any{
			"size": topK,
			"query": map[string]any{
				"match": map[string]any{
					"content": map[string]any{
						"query": query,
					},
				},
			},
		}
		if minScore > 0 {
			body["min_score"] = minScore
		}
	default: // "hybrid" and "rrf" both fall through to bool.should hybrid
		matchClause := map[string]any{
			"match": map[string]any{
				"content": map[string]any{
					"query": query,
					"boost": weights.Keyword,
				},
			},
		}
		knnLeaf := buildKNNClause(vector, topK, minScore, weights.Vector)
		// ES bool.should requires a list of complete query clauses; the knn
		// clause is wrapped in {"knn": ...} to be a valid leaf query.
		knnClause := map[string]any{"knn": knnLeaf}
		body = map[string]any{
			"size": topK,
			"query": map[string]any{
				"bool": map[string]any{
					"should":               []any{matchClause, knnClause},
					"minimum_should_match": 1,
				},
			},
		}
		if minScore > 0 {
			body["min_score"] = minScore
		}
	}

	b, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("BuildHybridQueryJSON: marshal: %w", err)
	}
	return string(b), nil
}

func buildKNNClause(vector []float64, topK int, minScore, boost float64) map[string]any {
	numCandidates := topK * 10
	if numCandidates < 50 {
		numCandidates = 50
	}
	clause := map[string]any{
		"field":          "content_vector",
		"query_vector":   vector,
		"k":              topK,
		"num_candidates": numCandidates,
		"boost":          boost,
	}
	if minScore > 0 {
		clause["similarity"] = minScore
	}
	return clause
}
