package knowledgebase

// Scope defines knowledge base visibility.
type Scope string

const (
	ScopePublic       Scope = "public"
	ScopeAgentPrivate Scope = "agent_private"
)

// KnowledgeBase represents a knowledge base entity.
type KnowledgeBase struct {
	ID                  int64   `json:"id"`
	Name                string  `json:"name"`
	Scope               Scope   `json:"scope"`
	OwnerUserID         *int64  `json:"owner_user_id"`
	OwnerAgentID        *int64  `json:"owner_agent_id"`
	CollectionName      string  `json:"collection_name"`
	LegacyCollectionKey *string `json:"legacy_collection_key,omitempty"`
	Status              string  `json:"status"`
	CreatedAt           string  `json:"created_at"`
	UpdatedAt           string  `json:"updated_at"`
}

// IndexName returns the ES index name for this knowledge base.
func (kb *KnowledgeBase) IndexName() string { return kb.CollectionName }

// Resolution represents a resolved knowledge base for a request.
type Resolution struct {
	KnowledgeBase *KnowledgeBase
	SelectedVia   string // "kb_id" | "legacy_key" | "scope"
}

// AccessError is returned when a caller cannot access a knowledge base.
type AccessError struct {
	KBID int64
	Msg  string
}

func (e *AccessError) Error() string { return e.Msg }

// ResolveRequest contains parameters for resolving a knowledge base.
type ResolveRequest struct {
	KBID               *int64
	Scope              *string
	UserID             *int64
	AgentID            *int64
	LegacyCollection   *string
	LegacyCollectionKey *string
}
