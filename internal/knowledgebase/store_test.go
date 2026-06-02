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

	kb, err := store.Create("test", "public", nil, nil, "test-model", 128)
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

func TestStoreGetByName(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	store, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	uid := int64(123)
	aid := int64(456)

	// Create a KB with specific owner
	kb, err := store.Create("my_project", "agent_private", &uid, &aid, "test-model", 128)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Lookup by exact match
	found, err := store.GetByName("agent_private", &uid, &aid, "my_project")
	if err != nil {
		t.Fatalf("GetByName: %v", err)
	}
	if found == nil {
		t.Fatal("expected to find KB")
	}
	if found.ID != kb.ID {
		t.Errorf("expected ID=%d, got %d", kb.ID, found.ID)
	}

	// Different user should not find it
	otherUID := int64(999)
	notFound, err := store.GetByName("agent_private", &otherUID, &aid, "my_project")
	if err != nil {
		t.Fatalf("GetByName: %v", err)
	}
	if notFound != nil {
		t.Error("expected nil for different owner")
	}

	// Public KB with nil owners
	pubKB, err := store.Create("shared", "public", nil, nil, "test-model", 128)
	if err != nil {
		t.Fatalf("Create public: %v", err)
	}
	pubFound, err := store.GetByName("public", nil, nil, "shared")
	if err != nil {
		t.Fatalf("GetByName public: %v", err)
	}
	if pubFound == nil || pubFound.ID != pubKB.ID {
		t.Error("expected to find public KB")
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

func TestResolveCollection(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	svc, err := NewService(dbPath)
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	uid := int64(123)
	aid := int64(456)

	// Resolve with collection name — should auto-create
	col := "my_project"
	res, err := svc.Resolve(ResolveRequest{Collection: &col, UserID: &uid, AgentID: &aid})
	if err != nil {
		t.Fatalf("Resolve by collection: %v", err)
	}
	if res.SelectedVia != "collection" {
		t.Errorf("expected collection, got %s", res.SelectedVia)
	}
	if res.KnowledgeBase.Name != "my_project" {
		t.Errorf("expected name=my_project, got %s", res.KnowledgeBase.Name)
	}
	if res.KnowledgeBase.Scope != ScopeAgentPrivate {
		t.Errorf("expected agent_private, got %s", res.KnowledgeBase.Scope)
	}

	// Second resolve with same collection should return existing KB
	res2, err := svc.Resolve(ResolveRequest{Collection: &col, UserID: &uid, AgentID: &aid})
	if err != nil {
		t.Fatalf("Resolve by collection #2: %v", err)
	}
	if res2.KnowledgeBase.ID != res.KnowledgeBase.ID {
		t.Error("expected same KB on second resolve")
	}

	// collection="default" should fall through to scope resolution
	defCol := "default"
	res3, err := svc.Resolve(ResolveRequest{Collection: &defCol, UserID: &uid, AgentID: &aid})
	if err != nil {
		t.Fatalf("Resolve with collection=default: %v", err)
	}
	if res3.SelectedVia != "scope" {
		t.Errorf("expected scope for collection=default, got %s", res3.SelectedVia)
	}

	// Different user with same collection name gets a different KB
	otherUID := int64(999)
	res4, err := svc.Resolve(ResolveRequest{Collection: &col, UserID: &otherUID, AgentID: &aid})
	if err != nil {
		t.Fatalf("Resolve by collection for other user: %v", err)
	}
	if res4.KnowledgeBase.ID == res.KnowledgeBase.ID {
		t.Error("expected different KB for different user")
	}
}

func TestResolveCollectionScopeIsolation(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	svc, err := NewService(dbPath)
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	name := "mykb"
	uid := int64(123)
	aid := int64(456)

	// User 123 creates a private "mykb"
	res1, err := svc.Resolve(ResolveRequest{Collection: &name, UserID: &uid, AgentID: &aid})
	if err != nil {
		t.Fatalf("private collection: %v", err)
	}
	if res1.KnowledgeBase.Scope != ScopeAgentPrivate {
		t.Errorf("expected agent_private, got %s", res1.KnowledgeBase.Scope)
	}

	// Same user, same name, but explicit scope=public → different KB in public pool
	publicScope := "public"
	res2, err := svc.Resolve(ResolveRequest{Collection: &name, Scope: &publicScope})
	if err != nil {
		t.Fatalf("public collection: %v", err)
	}
	if res2.KnowledgeBase.Scope != ScopePublic {
		t.Errorf("expected public scope, got %s", res2.KnowledgeBase.Scope)
	}
	if res2.KnowledgeBase.ID == res1.KnowledgeBase.ID {
		t.Error("public and private KBs with same name should be different")
	}

	// No user_id/agent_id → defaults to public pool, same KB as explicit public
	res3, err := svc.Resolve(ResolveRequest{Collection: &name})
	if err != nil {
		t.Fatalf("no-identity collection: %v", err)
	}
	if res3.KnowledgeBase.Scope != ScopePublic {
		t.Errorf("expected public scope for no-identity request, got %s", res3.KnowledgeBase.Scope)
	}
	if res3.KnowledgeBase.ID != res2.KnowledgeBase.ID {
		t.Error("no-identity should resolve to same public KB as explicit scope=public")
	}

	// Different agent, same user → different KB (agent-level isolation)
	otherAid := int64(789)
	res4, err := svc.Resolve(ResolveRequest{Collection: &name, UserID: &uid, AgentID: &otherAid})
	if err != nil {
		t.Fatalf("different agent collection: %v", err)
	}
	if res4.KnowledgeBase.ID == res1.KnowledgeBase.ID {
		t.Error("different agent should get a different KB")
	}
	if *res4.KnowledgeBase.OwnerAgentID != otherAid {
		t.Errorf("expected owner_agent_id=%d, got %d", otherAid, *res4.KnowledgeBase.OwnerAgentID)
	}
}

func TestEnsureAccess(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	svc, err := NewService(dbPath)
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	uid := int64(123)
	otherUID := int64(999)

	// Create a private KB
	kb, err := svc.store.Create("private_kb", "agent_private", &uid, nil, "test-model", 128)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Owner can access
	if err := svc.ensureAccess(kb, &uid); err != nil {
		t.Errorf("owner should have access: %v", err)
	}

	// Other user cannot access
	if err := svc.ensureAccess(kb, &otherUID); err == nil {
		t.Error("other user should be denied access")
	}

	// Public KB is accessible to anyone
	pubKB, err := svc.store.Create("public_kb", "public", nil, nil, "test-model", 128)
	if err != nil {
		t.Fatalf("Create public: %v", err)
	}
	if err := svc.ensureAccess(pubKB, nil); err != nil {
		t.Errorf("nil user should access public KB: %v", err)
	}
	if err := svc.ensureAccess(pubKB, &otherUID); err != nil {
		t.Errorf("any user should access public KB: %v", err)
	}

	// kb_id resolve with access control
	kbID := kb.ID
	res, err := svc.Resolve(ResolveRequest{KBID: &kbID, UserID: &uid})
	if err != nil {
		t.Fatalf("owner resolve by kb_id: %v", err)
	}
	if res.KnowledgeBase.ID != kb.ID {
		t.Error("expected correct KB")
	}

	// kb_id resolve denied for non-owner
	_, err = svc.Resolve(ResolveRequest{KBID: &kbID, UserID: &otherUID})
	if err == nil {
		t.Error("expected access denied for non-owner")
	}
}

func TestCheckEmbeddingMatch_Match(t *testing.T) {
	svc := &Service{embeddingModel: "mxbai-embed-large", embeddingDims: 1024}
	kb := &KnowledgeBase{Name: "test", EmbeddingModel: "mxbai-embed-large", EmbeddingDims: 1024}
	if err := svc.CheckEmbeddingMatch(kb); err != nil {
		t.Fatalf("expected match, got error: %v", err)
	}
}

func TestCheckEmbeddingMatch_DimsMismatch(t *testing.T) {
	svc := &Service{embeddingModel: "mxbai-embed-large", embeddingDims: 1024}
	kb := &KnowledgeBase{Name: "test", EmbeddingModel: "mxbai-embed-large", EmbeddingDims: 768}
	if err := svc.CheckEmbeddingMatch(kb); err == nil {
		t.Fatal("expected dims mismatch error")
	}
}

func TestCheckEmbeddingMatch_ModelMismatch(t *testing.T) {
	svc := &Service{embeddingModel: "mxbai-embed-large", embeddingDims: 1024}
	kb := &KnowledgeBase{Name: "test", EmbeddingModel: "bge-m3", EmbeddingDims: 1024}
	if err := svc.CheckEmbeddingMatch(kb); err == nil {
		t.Fatal("expected model mismatch error")
	}
}


func TestSetEmbeddingInfo(t *testing.T) {
	svc := &Service{}
	svc.SetEmbeddingInfo("test-model", 512)
	if svc.embeddingModel != "test-model" || svc.embeddingDims != 512 {
		t.Fatalf("SetEmbeddingInfo: got (%s, %d)", svc.embeddingModel, svc.embeddingDims)
	}
}
