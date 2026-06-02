package knowledgebase

import "fmt"

// Service provides higher-level knowledge base resolution with default creation.
type Service struct {
	store          *Store
	embeddingModel string
	embeddingDims  int
}

// NewService creates a knowledge base service.
func NewService(dbPath string) (*Service, error) {
	store, err := NewStore(dbPath)
	if err != nil {
		return nil, err
	}
	return &Service{store: store}, nil
}

// Store returns the underlying Store for operations like idempotency.
func (s *Service) Store() *Store {
	return s.store
}

// SetEmbeddingInfo records the current embedding model and dims for use
// when creating new KBs and checking existing KB compatibility.
func (s *Service) SetEmbeddingInfo(model string, dims int) {
	s.embeddingModel = model
	s.embeddingDims = dims
}

// CheckEmbeddingMatch returns an error if the KB's embedding info does not
// match the current embedder.
func (s *Service) CheckEmbeddingMatch(kb *KnowledgeBase) error {
	if kb.EmbeddingDims != s.embeddingDims {
		return fmt.Errorf("KB %q embedding dims mismatch: KB=%d, current=%d", kb.Name, kb.EmbeddingDims, s.embeddingDims)
	}
	if kb.EmbeddingModel != s.embeddingModel {
		return fmt.Errorf("KB %q embedding model mismatch: KB=%s, current=%s", kb.Name, kb.EmbeddingModel, s.embeddingModel)
	}
	return nil
}

// EnsurePublicDefault returns the public default KB, creating it if needed.
func (s *Service) EnsurePublicDefault() (*KnowledgeBase, error) {
	kb, err := s.store.GetPublicDefault()
	if err != nil {
		return nil, err
	}
	if kb != nil {
		return kb, nil
	}
	return s.store.Create("公共知识库", string(ScopePublic), nil, nil, s.embeddingModel, s.embeddingDims)
}

// EnsureAgentPrivateDefault returns the agent-private default KB, creating it if needed.
func (s *Service) EnsureAgentPrivateDefault(userID, agentID int64) (*KnowledgeBase, error) {
	kb, err := s.store.GetAgentPrivateDefault(userID, agentID)
	if err != nil {
		return nil, err
	}
	if kb != nil {
		return kb, nil
	}
	return s.store.Create(
		fmt.Sprintf("Agent %d 知识库", agentID),
		string(ScopeAgentPrivate),
		ptr(userID), ptr(agentID),
		s.embeddingModel, s.embeddingDims,
	)
}

// Create inserts a new knowledge base with the given parameters.
func (s *Service) Create(name, scope string, ownerUserID, ownerAgentID *int64) (*KnowledgeBase, error) {
	return s.store.Create(name, scope, ownerUserID, ownerAgentID, s.embeddingModel, s.embeddingDims)
}

// ListAccessible returns all KBs accessible to the given user, including defaults.
func (s *Service) ListAccessible(userID *int64) ([]*KnowledgeBase, error) {
	kbs, err := s.store.ListAccessible(userID)
	if err != nil {
		return nil, err
	}

	// Ensure defaults exist
	pub, _ := s.EnsurePublicDefault()
	if pub != nil && !containsKB(kbs, pub.ID) {
		kbs = append(kbs, pub)
	}

	return kbs, nil
}

// Resolve resolves request parameters to a specific knowledge base.
//
// Resolution order (aligns with Python):
//  1. kb_id       → direct Get (with access control)
//  2. collection  → GetByName lookup (auto-create if not found)
//  3. scope       → default KB (existing behavior)
func (s *Service) Resolve(req ResolveRequest) (*Resolution, error) {
	// 1. Explicit kb_id
	if req.KBID != nil {
		kb, err := s.store.Get(*req.KBID)
		if err != nil {
			return nil, err
		}
		if kb == nil {
			return nil, &AccessError{KBID: *req.KBID, Msg: fmt.Sprintf("knowledge base %d not found", *req.KBID)}
		}
		if err := s.ensureAccess(kb, req.UserID); err != nil {
			return nil, err
		}
		return &Resolution{KnowledgeBase: kb, SelectedVia: "kb_id"}, nil
	}

	// 2. Collection name lookup within scope (aligns with Python collection parameter)
	if req.Collection != nil && *req.Collection != "" && *req.Collection != "default" {
		scope := normalizeScope(req.Scope, req.UserID, req.AgentID)
		kb, err := s.store.GetByName(string(scope), req.UserID, req.AgentID, *req.Collection)
		if err != nil {
			return nil, err
		}
		if kb != nil {
			return &Resolution{KnowledgeBase: kb, SelectedVia: "collection"}, nil
		}
		// Auto-create (like Python's ChromaDB auto-creating a collection)
		kb, err = s.store.Create(*req.Collection, string(scope), req.UserID, req.AgentID, s.embeddingModel, s.embeddingDims)
		if err != nil {
			return nil, err
		}
		return &Resolution{KnowledgeBase: kb, SelectedVia: "collection"}, nil
	}

	// 3. Scope-based resolution (default KB)
	scope := normalizeScope(req.Scope, req.UserID, req.AgentID)

	if scope == ScopePublic {
		kb, err := s.EnsurePublicDefault()
		if err != nil {
			return nil, err
		}
		return &Resolution{KnowledgeBase: kb, SelectedVia: "scope"}, nil
	}

	// agent_private
	if req.UserID == nil || req.AgentID == nil {
		return nil, &AccessError{Msg: "agent_private requires user_id and agent_id"}
	}

	kb, err := s.EnsureAgentPrivateDefault(*req.UserID, *req.AgentID)
	if err != nil {
		return nil, err
	}
	return &Resolution{KnowledgeBase: kb, SelectedVia: "scope"}, nil
}

func normalizeScope(scope *string, userID, agentID *int64) Scope {
	if scope == nil {
		if userID != nil && agentID != nil {
			return ScopeAgentPrivate
		}
		return ScopePublic
	}
	switch *scope {
	case "agent_private", "private":
		return ScopeAgentPrivate
	default:
		return ScopePublic
	}
}

// ensureAccess checks that the caller (identified by userID) can access the KB.
// Public KBs are accessible to everyone. Agent-private KBs require owner match.
func (s *Service) ensureAccess(kb *KnowledgeBase, userID *int64) error {
	if kb.Scope == ScopePublic {
		return nil
	}
	if userID == nil || kb.OwnerUserID == nil || *kb.OwnerUserID != *userID {
		return &AccessError{KBID: kb.ID, Msg: "access denied"}
	}
	return nil
}

func containsKB(kbs []*KnowledgeBase, id int64) bool {
	for _, kb := range kbs {
		if kb.ID == id {
			return true
		}
	}
	return false
}

func ptr[T any](v T) *T { return &v }
