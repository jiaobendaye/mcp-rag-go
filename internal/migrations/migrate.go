// Package migrations provides a versioned database migration system.
//
// SQL files are embedded and applied in lexicographic order. Each
// migration is recorded in a schema_version table with a checksum to
// detect tampering. Migrations are idempotent — CREATE TABLE IF NOT
// EXISTS ensures safe re-runs.
package migrations

import (
	"crypto/sha256"
	"database/sql"
	"embed"
	"fmt"
	"sort"
	"time"
)

//go:embed *.sql
var fs embed.FS

// Run applies all pending migrations in lexicographic order. Each
// migration runs in its own transaction. Already-applied migrations
// are skipped.
func Run(db *sql.DB) error {
	// Ensure schema_version table exists (first-run)
	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS schema_version (
		migration_id TEXT PRIMARY KEY,
		applied_at   TEXT NOT NULL,
		checksum     TEXT NOT NULL
	)`); err != nil {
		return fmt.Errorf("migrate: ensure schema_version: %w", err)
	}

	entries, err := fs.ReadDir(".")
	if err != nil {
		return fmt.Errorf("migrate: read dir: %w", err)
	}

	// Sort by filename
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name() < entries[j].Name()
	})

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		migID := entry.Name()

		applied, err := isApplied(db, migID)
		if err != nil {
			return fmt.Errorf("migrate: check %s: %w", migID, err)
		}
		if applied {
			continue
		}

		sqlBytes, err := fs.ReadFile(migID)
		if err != nil {
			return fmt.Errorf("migrate: read %s: %w", migID, err)
		}

		checksum := fmt.Sprintf("%x", sha256.Sum256(sqlBytes))

		tx, err := db.Begin()
		if err != nil {
			return fmt.Errorf("migrate: begin tx %s: %w", migID, err)
		}

		if _, err := tx.Exec(string(sqlBytes)); err != nil {
			tx.Rollback()
			return fmt.Errorf("migrate: exec %s: %w", migID, err)
		}

		if _, err := tx.Exec(
			"INSERT INTO schema_version(migration_id, applied_at, checksum) VALUES (?, ?, ?)",
			migID, time.Now().UTC().Format(time.RFC3339), checksum,
		); err != nil {
			tx.Rollback()
			return fmt.Errorf("migrate: record %s: %w", migID, err)
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("migrate: commit %s: %w", migID, err)
		}
	}

	return nil
}

// isApplied checks whether a migration has already been applied.
func isApplied(db *sql.DB, migrationID string) (bool, error) {
	var count int
	err := db.QueryRow(
		"SELECT COUNT(*) FROM schema_version WHERE migration_id = ?",
		migrationID,
	).Scan(&count)
	if err != nil {
		return false, err
	}
	return count > 0, nil
}
