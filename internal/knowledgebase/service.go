package knowledgebase

import "fmt"

// Service provides higher-level knowledge base resolution with default creation.
type Service struct {
	store *Store
}

// NewService creates a knowledge base service.
func NewService(dbPath string) (*Service, error) {
	store, err := NewStore(dbPath)
	if err != nil {
		return nil, err
	}
	return &Service{store: store}, nil
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
	return s.store.Create("公共知识库", string(ScopePublic), nil, nil)
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
	)
}

// Create inserts a new knowledge base with the given parameters.
func (s *Service) Create(name, scope string, ownerUserID, ownerAgentID *int64) (*KnowledgeBase, error) {
	return s.store.Create(name, scope, ownerUserID, ownerAgentID)
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
		return &Resolution{KnowledgeBase: kb, SelectedVia: "kb_id"}, nil
	}

	// 2. Scope-based resolution
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

func containsKB(kbs []*KnowledgeBase, id int64) bool {
	for _, kb := range kbs {
		if kb.ID == id {
			return true
		}
	}
	return false
}

func ptr[T any](v T) *T { return &v }
