package rag

import "sort"

// KBSearchResult bundles a set of search hits with their source KB metadata.
type KBSearchResult struct {
	KBID         int64
	KBName       string
	KBScope      string
	OwnerUserID  *int64
	OwnerAgentID *int64
	Hits         []SearchHit
}

// AggregateResult represents a single merged search result with KB metadata.
type AggregateResult struct {
	Content         string         `json:"content"`
	Score           float64        `json:"score"`
	Source          string         `json:"source"`
	Filename        string         `json:"filename"`
	VectorScore     float64        `json:"vector_score"`
	KeywordScore    float64        `json:"keyword_score"`
	RetrievalMethod string         `json:"retrieval_method"`
	KBID            int64          `json:"-"`
	KBName          string         `json:"-"`
	KBScope         string         `json:"-"`
	OwnerUserID     *int64         `json:"-"`
	OwnerAgentID    *int64         `json:"-"`
}

// AggregateResults merges results from multiple KBs, sorts by score descending,
// truncates to limit, and enriches each hit with source KB metadata.
func AggregateResults(kbResults []KBSearchResult, limit int) []AggregateResult {
	var all []AggregateResult
	for _, kb := range kbResults {
		for _, h := range kb.Hits {
			all = append(all, AggregateResult{
				Content:         h.Content,
				Score:           h.Score,
				Source:          h.Source,
				Filename:        h.Filename,
				VectorScore:     h.VectorScore,
				KeywordScore:    h.KeywordScore,
				RetrievalMethod: h.RetrievalMethod,
				KBID:            kb.KBID,
				KBName:          kb.KBName,
				KBScope:         kb.KBScope,
				OwnerUserID:     kb.OwnerUserID,
				OwnerAgentID:    kb.OwnerAgentID,
			})
		}
	}

	sort.Slice(all, func(i, j int) bool {
		return all[i].Score > all[j].Score
	})

	if limit > 0 && len(all) > limit {
		all = all[:limit]
	}

	return all
}
