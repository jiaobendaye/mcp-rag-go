package rag

import (
	"testing"
)

func TestCacheHitMiss(t *testing.T) {
	cache := NewRetrievalCache()

	key := RetrievalCacheKey{
		Collection: "kb_1",
		Query:      "hello world",
		Mode:       "hybrid",
		Limit:      5,
		Threshold:  0.7,
	}

	// Miss on empty cache
	val, ok := cache.Get(key)
	if ok || val != nil {
		t.Error("expected miss on empty cache")
	}

	// Set
	resp := &SearchResponse{Query: "hello world", Collection: "kb_1"}
	cache.Set(key, resp)

	// Hit
	val, ok = cache.Get(key)
	if !ok {
		t.Error("expected hit after Set")
	}
	if val.Query != "hello world" {
		t.Errorf("expected query='hello world', got %q", val.Query)
	}

	// Stats
	stats := cache.Stats()
	if stats.Hits != 1 {
		t.Errorf("expected hits=1, got %d", stats.Hits)
	}
	if stats.Misses != 1 {
		t.Errorf("expected misses=1, got %d", stats.Misses)
	}
	if stats.Writes != 1 {
		t.Errorf("expected writes=1, got %d", stats.Writes)
	}
	if stats.EntriesCount != 1 {
		t.Errorf("expected entries_count=1, got %d", stats.EntriesCount)
	}
	if stats.ScopesCount != 1 {
		t.Errorf("expected scopes_count=1, got %d", stats.ScopesCount)
	}
}

func TestCacheTTL(t *testing.T) {
	cache := NewRetrievalCache()
	// Override TTL to very short for testing
	cache.cfg.TTL = 0 // expired immediately

	key := RetrievalCacheKey{Collection: "kb_1", Query: "test", Mode: "hybrid", Limit: 5, Threshold: 0.7}
	cache.Set(key, &SearchResponse{Query: "test"})

	// Should miss due to TTL
	_, ok := cache.Get(key)
	if ok {
		t.Error("expected miss due to expired TTL")
	}
}

func TestCacheDifferentKeys(t *testing.T) {
	cache := NewRetrievalCache()

	k1 := RetrievalCacheKey{Collection: "kb_1", Query: "q1", Mode: "hybrid", Limit: 5, Threshold: 0.7}
	k2 := RetrievalCacheKey{Collection: "kb_1", Query: "q2", Mode: "hybrid", Limit: 5, Threshold: 0.7}

	cache.Set(k1, &SearchResponse{Query: "q1"})
	cache.Set(k2, &SearchResponse{Query: "q2"})

	v1, _ := cache.Get(k1)
	v2, _ := cache.Get(k2)

	if v1.Query != "q1" {
		t.Errorf("expected q1, got %q", v1.Query)
	}
	if v2.Query != "q2" {
		t.Errorf("expected q2, got %q", v2.Query)
	}
}

func TestScopeInvalidation(t *testing.T) {
	cache := NewRetrievalCache()

	k1 := RetrievalCacheKey{Collection: "kb_1", Query: "q1", Mode: "hybrid", Limit: 5, Threshold: 0.7}
	k2 := RetrievalCacheKey{Collection: "kb_1", Query: "q2", Mode: "hybrid", Limit: 5, Threshold: 0.7}
	k3 := RetrievalCacheKey{Collection: "kb_2", Query: "q3", Mode: "hybrid", Limit: 5, Threshold: 0.7}

	cache.Set(k1, &SearchResponse{Query: "q1"})
	cache.Set(k2, &SearchResponse{Query: "q2"})
	cache.Set(k3, &SearchResponse{Query: "q3"})

	// Invalidate kb_1
	cache.InvalidateScope("kb_1")

	// kb_1 entries should be gone
	_, ok1 := cache.Get(k1)
	_, ok2 := cache.Get(k2)
	if ok1 || ok2 {
		t.Error("expected kb_1 entries to be invalidated")
	}

	// kb_2 should still be valid
	v3, ok3 := cache.Get(k3)
	if !ok3 {
		t.Error("expected kb_2 entry to survive")
	}
	if v3.Query != "q3" {
		t.Errorf("expected q3, got %q", v3.Query)
	}

	stats := cache.Stats()
	if stats.EntriesCount != 1 {
		t.Errorf("expected 1 entry remaining, got %d", stats.EntriesCount)
	}
	if stats.ScopesCount != 2 {
		t.Errorf("expected 2 scopes remaining, got %d", stats.ScopesCount)
	}
}

func TestInvalidateAll(t *testing.T) {
	cache := NewRetrievalCache()

	k1 := RetrievalCacheKey{Collection: "kb_1", Query: "q1", Mode: "hybrid", Limit: 5, Threshold: 0.7}
	k2 := RetrievalCacheKey{Collection: "kb_2", Query: "q2", Mode: "hybrid", Limit: 5, Threshold: 0.7}

	cache.Set(k1, &SearchResponse{Query: "q1"})
	cache.Set(k2, &SearchResponse{Query: "q2"})

	cache.InvalidateAll()

	_, ok1 := cache.Get(k1)
	_, ok2 := cache.Get(k2)
	if ok1 || ok2 {
		t.Error("expected all entries invalidated")
	}

	stats := cache.Stats()
	if stats.EntriesCount != 0 {
		t.Errorf("expected 0 entries, got %d", stats.EntriesCount)
	}
	if stats.ScopesCount != 0 {
		t.Errorf("expected 0 scopes, got %d", stats.ScopesCount)
	}
}

func TestLRUEviction(t *testing.T) {
	cache := NewRetrievalCache()
	cache.cfg.MaxEntries = 3

	keys := make([]RetrievalCacheKey, 5)
	for i := 0; i < 5; i++ {
		keys[i] = RetrievalCacheKey{
			Collection: "kb_1",
			Query:      string(rune('a' + i)),
			Mode:       "hybrid",
			Limit:      5,
			Threshold:  0.7,
		}
	}

	// Insert 3 entries
	for i := 0; i < 3; i++ {
		cache.Set(keys[i], &SearchResponse{Query: keys[i].Query})
	}

	// Access key 0 (becomes most recently used)
	cache.Get(keys[0])

	// Insert 2 more entries, should evict key 1 (LRU) then key 2
	cache.Set(keys[3], &SearchResponse{Query: keys[3].Query})
	cache.Set(keys[4], &SearchResponse{Query: keys[4].Query})

	// key 0 should survive (recently accessed)
	_, ok0 := cache.Get(keys[0])
	if !ok0 {
		t.Error("key 0 should survive (recently used)")
	}

	// key 1 should be evicted (oldest LRU)
	_, ok1 := cache.Get(keys[1])
	if ok1 {
		t.Error("key 1 should be evicted")
	}

	stats := cache.Stats()
	if stats.Evictions < 2 {
		t.Errorf("expected at least 2 evictions, got %d", stats.Evictions)
	}
	if stats.EntriesCount != 3 {
		t.Errorf("expected 3 entries, got %d", stats.EntriesCount)
	}
}

func TestUpdateExistingEntry(t *testing.T) {
	cache := NewRetrievalCache()
	key := RetrievalCacheKey{Collection: "kb_1", Query: "test", Mode: "hybrid", Limit: 5, Threshold: 0.7}

	cache.Set(key, &SearchResponse{Query: "old"})
	cache.Set(key, &SearchResponse{Query: "new"})

	val, ok := cache.Get(key)
	if !ok {
		t.Error("expected hit after update")
	}
	if val.Query != "new" {
		t.Errorf("expected 'new', got %q", val.Query)
	}

	stats := cache.Stats()
	if stats.Writes != 2 {
		t.Errorf("expected writes=2, got %d", stats.Writes)
	}
	if stats.EntriesCount != 1 {
		t.Errorf("expected 1 entry, got %d", stats.EntriesCount)
	}
}
