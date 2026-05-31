package knowledgebase

import (
	"path/filepath"
	"testing"
)

func TestStoreCreateAndGet(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	store, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	kb, err := store.Create("test", "public", nil, nil, ptr("legacy:test"))
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if kb.ID != 1 {
		t.Errorf("expected ID=1, got %d", kb.ID)
	}
	if kb.CollectionName != "kb_1" {
		t.Errorf("expected kb_1, got %s", kb.CollectionName)
	}

	got, err := store.Get(kb.ID)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Name != "test" {
		t.Errorf("expected name=test, got %s", got.Name)
	}
}

func TestServiceDefaults(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	svc, err := NewService(dbPath)
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	kb, err := svc.EnsurePublicDefault()
	if err != nil {
		t.Fatalf("EnsurePublicDefault: %v", err)
	}
	if kb.Scope != ScopePublic {
		t.Errorf("expected public, got %s", kb.Scope)
	}

	// Second call should return same KB
	kb2, err := svc.EnsurePublicDefault()
	if err != nil {
		t.Fatalf("EnsurePublicDefault #2: %v", err)
	}
	if kb2.ID != kb.ID {
		t.Error("expected same KB on second call")
	}
}

func TestResolve(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	svc, err := NewService(dbPath)
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	// Create default
	svc.EnsurePublicDefault()

	// Resolve by kb_id
	kbID := int64(1)
	res, err := svc.Resolve(ResolveRequest{KBID: &kbID})
	if err != nil {
		t.Fatalf("Resolve by id: %v", err)
	}
	if res.SelectedVia != "kb_id" {
		t.Errorf("expected kb_id, got %s", res.SelectedVia)
	}

	// Resolve by scope
	res, err = svc.Resolve(ResolveRequest{})
	if err != nil {
		t.Fatalf("Resolve by scope: %v", err)
	}
	if res.SelectedVia != "scope" {
		t.Errorf("expected scope, got %s", res.SelectedVia)
	}

	// Resolve missing ID
	missingID := int64(999)
	_, err = svc.Resolve(ResolveRequest{KBID: &missingID})
	if err == nil {
		t.Error("expected error for missing KB")
	}
}
