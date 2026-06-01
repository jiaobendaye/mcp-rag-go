package rag

import "testing"

func TestAggregateResults_Empty(t *testing.T) {
	got := AggregateResults(nil, 5)
	if got != nil && len(got) != 0 {
		t.Errorf("empty input should yield empty/nil, got %v", got)
	}
}

func TestAggregateResults_SortByScoreDesc(t *testing.T) {
	ownerUID := int64(7)
	ownerAID := int64(3)
	kbResults := []KBSearchResult{
		{
			KBID: 1, KBName: "alpha", KBScope: "public",
			OwnerUserID: &ownerUID, OwnerAgentID: &ownerAID,
			Hits: []SearchHit{
				{ChunkID: "a1", Content: "low", Score: 0.30},
				{ChunkID: "a2", Content: "high", Score: 0.95},
			},
		},
		{
			KBID: 2, KBName: "beta", KBScope: "agent_private",
			Hits: []SearchHit{
				{ChunkID: "b1", Content: "mid", Score: 0.60},
			},
		},
	}

	got := AggregateResults(kbResults, 10)
	if len(got) != 3 {
		t.Fatalf("expected 3 merged results, got %d", len(got))
	}

	// Verify sort order (descending by Score)
	wantOrder := []string{"a2", "b1", "a1"}
	for i, want := range wantOrder {
		if got[i].KBID == 0 {
			// reconstruct chunk id from content + KB name to avoid adding fields
		}
		if got[i].Score < got[i+1-i].Score && i+1 < len(got) {
			// already covered below
		}
		_ = want
	}
	if got[0].Score != 0.95 || got[1].Score != 0.60 || got[2].Score != 0.30 {
		t.Errorf("expected scores [0.95, 0.60, 0.30], got [%.2f, %.2f, %.2f]",
			got[0].Score, got[1].Score, got[2].Score)
	}
}

func TestAggregateResults_TruncateToLimit(t *testing.T) {
	kbResults := []KBSearchResult{
		{KBID: 1, KBName: "k1", Hits: []SearchHit{
			{ChunkID: "c1", Score: 0.9},
			{ChunkID: "c2", Score: 0.8},
			{ChunkID: "c3", Score: 0.7},
		}},
	}
	got := AggregateResults(kbResults, 2)
	if len(got) != 2 {
		t.Fatalf("expected 2 results after truncation, got %d", len(got))
	}
	if got[0].Score != 0.9 || got[1].Score != 0.8 {
		t.Errorf("expected top-2 scores [0.9, 0.8], got [%.2f, %.2f]", got[0].Score, got[1].Score)
	}
}

func TestAggregateResults_MetadataInjection(t *testing.T) {
	ownerUID := int64(42)
	kbResults := []KBSearchResult{
		{
			KBID: 99, KBName: "docs", KBScope: "public",
			OwnerUserID: &ownerUID,
			Hits: []SearchHit{
				{ChunkID: "c1", Content: "x", Source: "src", Filename: "f.txt",
					VectorScore: 0.8, KeywordScore: 0.2, RetrievalMethod: "hybrid", Score: 0.75},
			},
		},
	}
	got := AggregateResults(kbResults, 10)
	if len(got) != 1 {
		t.Fatalf("expected 1 result, got %d", len(got))
	}
	r := got[0]
	if r.KBID != 99 {
		t.Errorf("expected KBID=99, got %d", r.KBID)
	}
	if r.KBName != "docs" {
		t.Errorf("expected KBName=docs, got %s", r.KBName)
	}
	if r.KBScope != "public" {
		t.Errorf("expected KBScope=public, got %s", r.KBScope)
	}
	if r.OwnerUserID == nil || *r.OwnerUserID != 42 {
		t.Errorf("expected OwnerUserID=42, got %v", r.OwnerUserID)
	}
	if r.Content != "x" || r.Source != "src" || r.Filename != "f.txt" {
		t.Errorf("content/source/filename not copied: %+v", r)
	}
	if r.VectorScore != 0.8 || r.KeywordScore != 0.2 || r.RetrievalMethod != "hybrid" {
		t.Errorf("vector/keyword/method not copied: %+v", r)
	}
}

func TestAggregateResults_DedupWithinKB(t *testing.T) {
	// Aggregate does not dedup across KBs but must not lose hits from a single KB.
	kbResults := []KBSearchResult{
		{KBID: 1, KBName: "k1", Hits: []SearchHit{
			{ChunkID: "c1", Score: 0.5},
			{ChunkID: "c1", Score: 0.7}, // same chunk id, higher score
			{ChunkID: "c2", Score: 0.6},
		}},
	}
	got := AggregateResults(kbResults, 10)
	if len(got) != 3 {
		t.Fatalf("expected 3 results (no dedup at this layer), got %d", len(got))
	}
	// Sort should put 0.7 first
	if got[0].Score != 0.7 {
		t.Errorf("expected first score 0.7, got %.2f", got[0].Score)
	}
}

func TestAggregateResults_MultiKBMerge(t *testing.T) {
	// 3 KBs × 3 hits → 9 total, sorted desc
	kbResults := []KBSearchResult{
		{KBID: 1, KBName: "a", Hits: []SearchHit{
			{ChunkID: "a1", Score: 0.91},
			{ChunkID: "a2", Score: 0.55},
			{ChunkID: "a3", Score: 0.10},
		}},
		{KBID: 2, KBName: "b", Hits: []SearchHit{
			{ChunkID: "b1", Score: 0.88},
			{ChunkID: "b2", Score: 0.42},
			{ChunkID: "b3", Score: 0.30},
		}},
		{KBID: 3, KBName: "c", Hits: []SearchHit{
			{ChunkID: "c1", Score: 0.77},
			{ChunkID: "c2", Score: 0.50},
			{ChunkID: "c3", Score: 0.20},
		}},
	}
	got := AggregateResults(kbResults, 5)
	if len(got) != 5 {
		t.Fatalf("expected 5 results, got %d", len(got))
	}
	// Expected top-5: 0.91, 0.88, 0.77, 0.55, 0.50
	wantScores := []float64{0.91, 0.88, 0.77, 0.55, 0.50}
	for i, w := range wantScores {
		if got[i].Score != w {
			t.Errorf("position %d: expected score %.2f, got %.2f", i, w, got[i].Score)
		}
	}
}

func TestAggregateResults_LimitZeroKeepsAll(t *testing.T) {
	kbResults := []KBSearchResult{
		{KBID: 1, KBName: "a", Hits: []SearchHit{
			{ChunkID: "a1", Score: 0.5},
			{ChunkID: "a2", Score: 0.4},
		}},
	}
	got := AggregateResults(kbResults, 0)
	if len(got) != 2 {
		t.Errorf("expected 2 results with limit=0, got %d", len(got))
	}
}
