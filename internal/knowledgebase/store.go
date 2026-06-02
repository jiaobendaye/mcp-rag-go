package knowledgebase

import (
	"database/sql"
	"fmt"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// Store provides SQLite-backed knowledge base CRUD.
type Store struct {
	db *sql.DB
	mu sync.RWMutex
}

// NewStore opens or creates the knowledge base database.
func NewStore(dbPath string) (*Store, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	s := &Store{db: db}
	if err := s.initialize(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Store) initialize() error {
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS knowledge_bases (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			name TEXT NOT NULL,
			scope TEXT NOT NULL,
			owner_user_id INTEGER,
			owner_agent_id INTEGER,
			collection_name TEXT NOT NULL UNIQUE,
			status TEXT NOT NULL DEFAULT 'active',
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL
		)
	`)
	return err
}

// Get retrieves a knowledge base by ID.
func (s *Store) Get(kbID int64) (*KnowledgeBase, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.get(kbID)
}

func (s *Store) get(kbID int64) (*KnowledgeBase, error) {
	row := s.db.QueryRow(
		"SELECT id, name, scope, owner_user_id, owner_agent_id, collection_name, status, created_at, updated_at FROM knowledge_bases WHERE id=?",
		kbID,
	)
	return scanKB(row)
}

// GetPublicDefault returns the default public knowledge base.
func (s *Store) GetPublicDefault() (*KnowledgeBase, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	row := s.db.QueryRow(
		"SELECT id, name, scope, owner_user_id, owner_agent_id, collection_name, status, created_at, updated_at FROM knowledge_bases WHERE scope='public' AND owner_user_id IS NULL AND owner_agent_id IS NULL AND status='active' ORDER BY id ASC LIMIT 1",
	)
	return scanKB(row)
}

// GetAgentPrivateDefault returns the default agent-private knowledge base.
func (s *Store) GetAgentPrivateDefault(userID, agentID int64) (*KnowledgeBase, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	row := s.db.QueryRow(
		"SELECT id, name, scope, owner_user_id, owner_agent_id, collection_name, status, created_at, updated_at FROM knowledge_bases WHERE scope='agent_private' AND owner_user_id=? AND owner_agent_id=? AND status='active' ORDER BY id ASC LIMIT 1",
		userID, agentID,
	)
	return scanKB(row)
}

// ListAccessible returns knowledge bases accessible to the given user.
func (s *Store) ListAccessible(userID *int64) ([]*KnowledgeBase, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	rows, err := s.db.Query(
		"SELECT id, name, scope, owner_user_id, owner_agent_id, collection_name, status, created_at, updated_at FROM knowledge_bases WHERE status='active' AND (scope='public' OR (? IS NOT NULL AND owner_user_id=?)) ORDER BY CASE WHEN scope='public' THEN 0 ELSE 1 END, name COLLATE NOCASE ASC, id ASC",
		userID, userID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanKBs(rows)
}

// Create inserts a new knowledge base and returns it with the assigned ID.
func (s *Store) Create(name, scope string, ownerUserID, ownerAgentID *int64) (*KnowledgeBase, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now().UTC().Format(time.RFC3339)
	tmpColl := fmt.Sprintf("pending_%s", now[:10]) // placeholder

	result, err := s.db.Exec(
		"INSERT INTO knowledge_bases(name, scope, owner_user_id, owner_agent_id, collection_name, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
		name, scope, ownerUserID, ownerAgentID, tmpColl, "active", now, now,
	)
	if err != nil {
		return nil, fmt.Errorf("insert kb: %w", err)
	}

	id, _ := result.LastInsertId()
	collName := fmt.Sprintf("kb_%d", id)
	_, err = s.db.Exec("UPDATE knowledge_bases SET collection_name=?, updated_at=? WHERE id=?", collName, now, id)
	if err != nil {
		return nil, fmt.Errorf("update collection_name: %w", err)
	}

	return s.get(id)
}

func scanKB(row *sql.Row) (*KnowledgeBase, error) {
	var kb KnowledgeBase
	err := row.Scan(&kb.ID, &kb.Name, &kb.Scope, &kb.OwnerUserID, &kb.OwnerAgentID, &kb.CollectionName, &kb.Status, &kb.CreatedAt, &kb.UpdatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &kb, nil
}

func scanKBs(rows *sql.Rows) ([]*KnowledgeBase, error) {
	var kbs []*KnowledgeBase
	for rows.Next() {
		var kb KnowledgeBase
		if err := rows.Scan(&kb.ID, &kb.Name, &kb.Scope, &kb.OwnerUserID, &kb.OwnerAgentID, &kb.CollectionName, &kb.Status, &kb.CreatedAt, &kb.UpdatedAt); err != nil {
			return nil, err
		}
		kbs = append(kbs, &kb)
	}
	return kbs, nil
}
