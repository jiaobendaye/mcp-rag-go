package knowledgebase

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"
)

// IdempotencyRecord is a cached idempotent response.
type IdempotencyRecord struct {
	Key             string            `json:"key"`
	Method          string            `json:"method"`
	Path            string            `json:"path"`
	RequestHash     string            `json:"request_hash"`
	ResponseStatus  int               `json:"response_status"`
	ResponseBody    []byte            `json:"response_body"`
	ResponseHeaders map[string]string `json:"response_headers"`
	CreatedAt       string            `json:"created_at"`
	ExpiresAt       string            `json:"expires_at"`
}

// GetIdempotencyRecord looks up a cached idempotent response by (key, method, path).
// Returns nil if not found or expired.
func (s *Store) GetIdempotencyRecord(key, method, path string) (*IdempotencyRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	row := s.db.QueryRow(
		`SELECT key, method, path, request_hash, response_status, response_body, response_headers, created_at, expires_at
		 FROM idempotency_keys WHERE key=? AND method=? AND path=? AND expires_at > ?`,
		key, method, path, time.Now().UTC().Format(time.RFC3339),
	)

	var r IdempotencyRecord
	var headersJSON []byte
	err := row.Scan(&r.Key, &r.Method, &r.Path, &r.RequestHash, &r.ResponseStatus,
		&r.ResponseBody, &headersJSON, &r.CreatedAt, &r.ExpiresAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get idempotency record: %w", err)
	}
	if len(headersJSON) > 0 {
		json.Unmarshal(headersJSON, &r.ResponseHeaders)
	}
	return &r, nil
}

// SetIdempotencyRecord stores a new idempotent response with 24h TTL.
func (s *Store) SetIdempotencyRecord(key, method, path, requestHash string, status int, body []byte, headers map[string]string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	headersJSON, _ := json.Marshal(headers)
	now := time.Now().UTC().Format(time.RFC3339)
	expires := time.Now().UTC().Add(24 * time.Hour).Format(time.RFC3339)

	_, err := s.db.Exec(
		`INSERT OR REPLACE INTO idempotency_keys(key, method, path, request_hash, response_status, response_body, response_headers, created_at, expires_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		key, method, path, requestHash, status, body, headersJSON, now, expires,
	)
	if err != nil {
		return fmt.Errorf("set idempotency record: %w", err)
	}
	return nil
}

// CleanupExpiredIdempotency deletes expired idempotency records.
// Returns the number of deleted rows.
func (s *Store) CleanupExpiredIdempotency() (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	result, err := s.db.Exec(
		"DELETE FROM idempotency_keys WHERE expires_at <= ?",
		time.Now().UTC().Format(time.RFC3339),
	)
	if err != nil {
		return 0, fmt.Errorf("cleanup idempotency: %w", err)
	}
	return result.RowsAffected()
}
