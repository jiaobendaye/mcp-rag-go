-- 001: Initial schema — knowledge_bases table with embedding model binding.
-- Applied idempotently via CREATE TABLE IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS knowledge_bases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    scope TEXT NOT NULL,
    owner_user_id INTEGER,
    owner_agent_id INTEGER,
    collection_name TEXT NOT NULL UNIQUE,
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_dims  INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Schema version tracking table.
CREATE TABLE IF NOT EXISTS schema_version (
    migration_id TEXT PRIMARY KEY,
    applied_at   TEXT NOT NULL,
    checksum     TEXT NOT NULL
);
