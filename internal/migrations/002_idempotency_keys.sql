-- 002: Idempotency keys table for write-API dedup.
-- Cache TTL is 24h; cleanup deletes rows older than expires_at.

CREATE TABLE IF NOT EXISTS idempotency_keys (
    key              TEXT NOT NULL,
    method           TEXT NOT NULL,
    path             TEXT NOT NULL,
    request_hash     TEXT NOT NULL,
    response_status  INTEGER NOT NULL,
    response_body    BLOB NOT NULL,
    response_headers BLOB NOT NULL,
    created_at       TEXT NOT NULL,
    expires_at       TEXT NOT NULL,
    PRIMARY KEY (key, method, path)
);

CREATE INDEX IF NOT EXISTS idempotency_expires ON idempotency_keys(expires_at);
