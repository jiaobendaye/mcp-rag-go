package rag

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildHybridQueryJSON_HybridMode(t *testing.T) {
	vec := []float64{0.1, 0.2, 0.3}
	out, err := BuildHybridQueryJSON("hello", vec, 5, 0.7, SearchWeights{Vector: 0.6, Keyword: 0.4}, "hybrid")
	if err != nil {
		t.Fatalf("BuildHybridQueryJSON: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal([]byte(out), &body); err != nil {
		t.Fatalf("not valid JSON: %v\nbody=%s", err, out)
	}
	if size, _ := body["size"].(float64); int(size) != 5 {
		t.Errorf("expected size=5, got %v", body["size"])
	}
	if body["min_score"] == nil {
		t.Errorf("expected min_score in body, got %v", body["min_score"])
	}
	q, ok := body["query"].(map[string]any)
	if !ok {
		t.Fatalf("expected query object, got %T", body["query"])
	}
	boolQ, ok := q["bool"].(map[string]any)
	if !ok {
		t.Fatalf("expected bool query, got %T", q["bool"])
	}
	should, ok := boolQ["should"].([]any)
	if !ok || len(should) != 2 {
		t.Fatalf("expected 2 should clauses, got %v", should)
	}
	// First clause: match (keyword)
	matchClause, ok := should[0].(map[string]any)["match"].(map[string]any)
	if !ok {
		t.Fatalf("expected match clause first, got %v", should[0])
	}
	contentMatch, ok := matchClause["content"].(map[string]any)
	if !ok {
		t.Fatalf("expected content match, got %v", matchClause)
	}
	if boost, _ := contentMatch["boost"].(float64); boost != 0.4 {
		t.Errorf("expected keyword boost=0.4, got %v", contentMatch["boost"])
	}
	// Second clause: knn (vector) — wrapped as {"knn": <clause>} so the
	// item in bool.should is a valid query DSL leaf. (Earlier versions
	// emitted the knn map directly, but ES bool.should requires wrapped
	// query clauses, not raw leaf maps.)
	knnWrapped, ok := should[1].(map[string]any)["knn"].(map[string]any)
	if !ok {
		t.Fatalf("expected knn clause second (wrapped as {\"knn\": <...>}), got %v", should[1])
	}
	if field, _ := knnWrapped["field"].(string); field != "content_vector" {
		t.Errorf("expected field=content_vector, got %v", knnWrapped["field"])
	}
	if boost, _ := knnWrapped["boost"].(float64); boost != 0.6 {
		t.Errorf("expected vector boost=0.6, got %v", knnWrapped["boost"])
	}
}

func TestBuildHybridQueryJSON_PureVector(t *testing.T) {
	vec := []float64{0.1, 0.2, 0.3}
	out, err := BuildHybridQueryJSON("hello", vec, 5, 0.5, SearchWeights{}, "knn")
	if err != nil {
		t.Fatalf("BuildHybridQueryJSON: %v", err)
	}
	var body map[string]any
	json.Unmarshal([]byte(out), &body)
	knn, ok := body["knn"].([]any)
	if !ok || len(knn) != 1 {
		t.Fatalf("expected 1 knn clause, got %v", body["knn"])
	}
	clause, ok := knn[0].(map[string]any)
	if !ok {
		t.Fatalf("expected knn clause as object, got %T", knn[0])
	}
	if clause["field"] != "content_vector" {
		t.Errorf("expected field=content_vector, got %v", clause["field"])
	}
	if _, ok := body["query"]; ok {
		t.Errorf("knn mode should not have a 'query' key, got %v", body)
	}
}

func TestBuildHybridQueryJSON_PureKeyword(t *testing.T) {
	out, err := BuildHybridQueryJSON("hello", []float64{0.1, 0.2}, 5, 0.5, SearchWeights{}, "keyword")
	if err != nil {
		t.Fatalf("BuildHybridQueryJSON: %v", err)
	}
	var body map[string]any
	json.Unmarshal([]byte(out), &body)
	q, ok := body["query"].(map[string]any)
	if !ok {
		t.Fatalf("expected query object, got %T", body["query"])
	}
	match, ok := q["match"].(map[string]any)
	if !ok {
		t.Fatalf("expected match query, got %T", q["match"])
	}
	content, ok := match["content"].(map[string]any)
	if !ok {
		t.Fatalf("expected content match, got %T", match["content"])
	}
	if content["query"] != "hello" {
		t.Errorf("expected query=hello, got %v", content["query"])
	}
	if _, ok := body["knn"]; ok {
		t.Errorf("keyword mode should not have a 'knn' key, got %v", body)
	}
}

func TestBuildHybridQueryJSON_RRFFallsBackToHybrid(t *testing.T) {
	vec := []float64{0.1}
	out, err := BuildHybridQueryJSON("hello", vec, 3, 0.0, SearchWeights{Vector: 0.5, Keyword: 0.5}, "rrf")
	if err != nil {
		t.Fatalf("BuildHybridQueryJSON: %v", err)
	}
	if !strings.Contains(out, `"should"`) {
		t.Errorf("rrf mode should fall back to bool.should hybrid, got %s", out)
	}
}

func TestBuildHybridQueryJSON_DefaultsApplied(t *testing.T) {
	// topK=0 should default to 5; minScore<0 should default to 0
	out, err := BuildHybridQueryJSON("q", []float64{0.1}, 0, -1, SearchWeights{}, "knn")
	if err != nil {
		t.Fatalf("BuildHybridQueryJSON: %v", err)
	}
	var body map[string]any
	json.Unmarshal([]byte(out), &body)
	if size, _ := body["size"].(float64); int(size) != 5 {
		t.Errorf("expected default size=5, got %v", body["size"])
	}
	if _, ok := body["min_score"]; ok {
		t.Errorf("min_score should be omitted when set to 0, got %v", body["min_score"])
	}
}
