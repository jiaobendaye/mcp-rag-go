package rag

import (
	"container/list"
	"fmt"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// RetrievalCacheKey uniquely identifies a cached search result.
type RetrievalCacheKey struct {
	Collection string
	Query      string
	Mode       string
	Limit      int
	Threshold  float64
}

// Hash returns a stable string key for map lookup.
func (k RetrievalCacheKey) Hash() string {
	return fmt.Sprintf("%s|%s|%s|%d|%.2f", k.Collection, k.Query, k.Mode, k.Limit, k.Threshold)
}

// cacheEntry holds a single cached search result.
type cacheEntry struct {
	key       RetrievalCacheKey
	value     *SearchResponse
	createdAt time.Time
	scopeGen  int64 // generation counter at insert time
	lruElement *list.Element
}

// CacheStats exposes hit-rate counters.
type CacheStats struct {
	Hits        int64 `json:"hits"`
	Misses      int64 `json:"misses"`
	Writes      int64 `json:"writes"`
	Evictions   int64 `json:"evictions"`
	EntriesCount int  `json:"entries_count"`
	ScopesCount  int  `json:"scopes_count"`
}

// retrievalCacheConfig holds tunables.
type retrievalCacheConfig struct {
	TTL        time.Duration
	MaxEntries int
}

// defaultCacheConfig returns the standard cache settings.
func defaultCacheConfig() retrievalCacheConfig {
	return retrievalCacheConfig{
		TTL:        300 * time.Second,
		MaxEntries: 256,
	}
}

// ---------------------------------------------------------------------------
// RetrievalCache
// ---------------------------------------------------------------------------

// RetrievalCache is a TTL + LRU + scope-generational cache for search results.
type RetrievalCache struct {
	mu       sync.RWMutex
	entries  map[string]*cacheEntry // key = hash(cacheKey)
	scopes   map[string]int64       // scope → generation counter
	lru      *list.List
	cfg      retrievalCacheConfig
	stats    CacheStats
}

// NewRetrievalCache creates an initialized RetrievalCache.
func NewRetrievalCache() *RetrievalCache {
	return &RetrievalCache{
		entries: make(map[string]*cacheEntry),
		scopes:  make(map[string]int64),
		lru:     list.New(),
		cfg:     defaultCacheConfig(),
	}
}

// Get returns a cached SearchResponse if a non-expired entry exists.
func (rc *RetrievalCache) Get(key RetrievalCacheKey) (*SearchResponse, bool) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	h := key.Hash()
	entry, ok := rc.entries[h]
	if !ok {
		rc.stats.Misses++
		return nil, false
	}

	// TTL check
	if time.Since(entry.createdAt) > rc.cfg.TTL {
		rc.evictEntry(h, entry)
		rc.stats.Misses++
		return nil, false
	}

	// Scope generation check
	currentGen := rc.scopes[key.Collection]
	if entry.scopeGen != currentGen {
		rc.evictEntry(h, entry)
		rc.stats.Misses++
		return nil, false
	}

	// LRU: move to front
	rc.lru.MoveToFront(entry.lruElement)

	rc.stats.Hits++
	return entry.value, true
}

// Set stores a search result in the cache.
func (rc *RetrievalCache) Set(key RetrievalCacheKey, value *SearchResponse) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	h := key.Hash()

	// Update existing entry if present
	if entry, ok := rc.entries[h]; ok {
		entry.value = value
		entry.createdAt = time.Now()
		entry.scopeGen = rc.scopes[key.Collection]
		rc.lru.MoveToFront(entry.lruElement)
		rc.stats.Writes++
		return
	}

	// Evict if at capacity
	for len(rc.entries) >= rc.cfg.MaxEntries {
		oldest := rc.lru.Back()
		if oldest == nil {
			break
		}
		oldEntry := oldest.Value.(*cacheEntry)
		delete(rc.entries, oldEntry.key.Hash())
		rc.lru.Remove(oldest)
		rc.stats.Evictions++
	}

	// Insert new entry
	gen := rc.getOrCreateScopeGen(key.Collection)
	elem := rc.lru.PushFront(nil) // placeholder; we'll set Value below
	entry := &cacheEntry{
		key:        key,
		value:      value,
		createdAt:  time.Now(),
		scopeGen:   gen,
		lruElement: elem,
	}
	elem.Value = entry
	rc.entries[h] = entry
	rc.stats.Writes++
}

// InvalidateScope bumps the generation counter for a collection, immediately
// invalidating all cached entries belonging to it.
func (rc *RetrievalCache) InvalidateScope(collection string) {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	rc.scopes[collection]++

	// Clean up stale entries eagerly
	for h, entry := range rc.entries {
		if entry.key.Collection == collection {
			rc.lru.Remove(entry.lruElement)
			delete(rc.entries, h)
		}
	}
}

// InvalidateAll clears every cached entry.
func (rc *RetrievalCache) InvalidateAll() {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	rc.entries = make(map[string]*cacheEntry)
	rc.scopes = make(map[string]int64)
	rc.lru.Init()
}

// Stats returns a snapshot of cache counters.
func (rc *RetrievalCache) Stats() CacheStats {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	s := rc.stats
	s.EntriesCount = len(rc.entries)
	s.ScopesCount = len(rc.scopes)
	return s
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func (rc *RetrievalCache) evictEntry(hash string, entry *cacheEntry) {
	rc.lru.Remove(entry.lruElement)
	delete(rc.entries, hash)
}

func (rc *RetrievalCache) getOrCreateScopeGen(collection string) int64 {
	gen, ok := rc.scopes[collection]
	if !ok {
		gen = 0
		rc.scopes[collection] = gen
	}
	return gen
}
