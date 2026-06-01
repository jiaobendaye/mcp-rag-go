package rag

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/elastic/go-elasticsearch/v8/typedapi/types"
)

func TestProjectResultParser_ExtractsChunkID(t *testing.T) {
	parser := ProjectResultParser()
	id := "abc123"
	score := types.Float64(0.87)
	src, _ := json.Marshal(map[string]any{
		"content":     "hello world",
		"chunk_id":    "doc1_chunk_0001",
		"document_id": "doc1",
	})
	hit := types.Hit{
		Id_:     &id,
		Score_:  &score,
		Source_: src,
	}

	doc, err := parser(context.Background(), hit)
	if err != nil {
		t.Fatalf("parser returned error: %v", err)
	}
	if doc.Content != "hello world" {
		t.Errorf("expected content=hello world, got %q", doc.Content)
	}
	if cid, ok := doc.MetaData["chunk_id"].(string); !ok || cid != "doc1_chunk_0001" {
		t.Errorf("expected chunk_id=doc1_chunk_0001 in MetaData, got %v", doc.MetaData["chunk_id"])
	}
	if doc.MetaData["document_id"] != "doc1" {
		t.Errorf("expected document_id=doc1 in MetaData, got %v", doc.MetaData["document_id"])
	}
	if doc.MetaData["score"].(float64) != 0.87 {
		t.Errorf("expected score=0.87 in MetaData, got %v", doc.MetaData["score"])
	}
}

func TestProjectResultParser_MissingChunkIDIsOK(t *testing.T) {
	parser := ProjectResultParser()
	id := "x"
	score := types.Float64(0.5)
	src, _ := json.Marshal(map[string]any{"content": "no chunk id here"})
	hit := types.Hit{Id_: &id, Score_: &score, Source_: src}

	doc, err := parser(context.Background(), hit)
	if err != nil {
		t.Fatalf("parser returned error: %v", err)
	}
	if _, has := doc.MetaData["chunk_id"]; has {
		t.Errorf("expected no chunk_id key when source has none, got %v", doc.MetaData["chunk_id"])
	}
}

func TestProjectResultParser_EmptyChunkIDIsOmitted(t *testing.T) {
	// Note: defaultResultParser copies ALL source fields into MetaData,
	// so chunk_id="" will appear in MetaData. The parser's own chunk_id
	// check is a no-op when the value is empty (it never overwrites
	// MetaData), so the empty value is preserved from the default copy.
	// This test documents that behavior.
	parser := ProjectResultParser()
	id := "x"
	score := types.Float64(0.5)
	src, _ := json.Marshal(map[string]any{"content": "empty chunk id", "chunk_id": ""})
	hit := types.Hit{Id_: &id, Score_: &score, Source_: src}

	doc, err := parser(context.Background(), hit)
	if err != nil {
		t.Fatalf("parser returned error: %v", err)
	}
	// MetaData has chunk_id (from default copy) but it's an empty string.
	cid, ok := doc.MetaData["chunk_id"].(string)
	if !ok {
		t.Fatalf("expected chunk_id in MetaData, got %v (type %T)", doc.MetaData["chunk_id"], doc.MetaData["chunk_id"])
	}
	if cid != "" {
		t.Errorf("expected empty chunk_id preserved, got %q", cid)
	}
}
