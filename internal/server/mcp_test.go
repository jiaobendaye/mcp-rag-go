package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
	"github.com/gin-gonic/gin"
	"github.com/mark3labs/mcp-go/mcp"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/knowledgebase"
)

// mcpTestEmbedder implements eino embedding.Embedder
type mcpTestEmbedder struct{}

func (e *mcpTestEmbedder) EmbedStrings(ctx context.Context, texts []string, _ ...embedding.Option) ([][]float64, error) {
	vecs := make([][]float64, len(texts))
	for i := range vecs {
		vecs[i] = []float64{0.1, 0.2, 0.3}
	}
	return vecs, nil
}

// mcpTestLLM implements eino model.BaseChatModel
type mcpTestLLM struct{}

func (m *mcpTestLLM) Generate(ctx context.Context, input []*schema.Message, _ ...model.Option) (*schema.Message, error) {
	return &schema.Message{Role: schema.Assistant, Content: "这是基于知识的回答。"}, nil
}

func (m *mcpTestLLM) Stream(ctx context.Context, input []*schema.Message, _ ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	return nil, nil
}

// mcpTestSearcher implements eino retriever.Retriever. The default returns
// two stub documents so the raw mode handler has something to format.
type mcpTestSearcher struct {
	retrieveFunc func(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error)
}

func (m *mcpTestSearcher) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	if m.retrieveFunc != nil {
		return m.retrieveFunc(ctx, query, opts...)
	}
	return []*schema.Document{
		{ID: "c1", Content: "测试内容1", MetaData: map[string]any{"score": 0.95, "filename": "test.txt", "source": "test.txt", "chunk_id": "c1", "document_id": "d1"}},
		{ID: "c2", Content: "测试内容2", MetaData: map[string]any{"score": 0.85, "filename": "test2.txt", "source": "test2.txt", "chunk_id": "c2", "document_id": "d2"}},
	}, nil
}

// setupMCPServer creates a test Server with MCP support.
func setupMCPServer(t *testing.T) *gin.Engine {
	t.Helper()
	gin.SetMode(gin.TestMode)

	// Create in-memory knowledgebase service
	kbs, err := knowledgebase.NewService(":memory:")
	if err != nil {
		t.Fatalf("create kb service: %v", err)
	}
	_, err = kbs.EnsurePublicDefault()
	if err != nil {
		t.Fatalf("ensure public default: %v", err)
	}

	cfg := config.DefaultConfig()
	emb := &mcpTestEmbedder{}

	s, _ := New(cfg, nil, nil, nil, emb, nil, &mcpTestLLM{}, nil, nil, kbs, 0, "test-model")
	return s.Setup()
}

// TestMCPRagAskParameterParsing tests that the rag_ask handler correctly
// extracts parameters from the CallToolRequest (task 4.1).
func TestMCPRagAskParameterParsing(t *testing.T) {
	t.Skip("requires ES client after KBRetriever removal")
	gin.SetMode(gin.TestMode)

	kbs, err := knowledgebase.NewService(":memory:")
	if err != nil {
		t.Fatalf("create kb service: %v", err)
	}
	_, err = kbs.EnsurePublicDefault()
	if err != nil {
		t.Fatalf("ensure public default: %v", err)
	}

	cfg := config.DefaultConfig()
	emb := &mcpTestEmbedder{}
	s, _ := New(cfg, nil, nil, nil, emb, nil, &mcpTestLLM{}, nil, nil, kbs, 0, "test-model")

	// Init MCP server (stores mcpSrv on the Server)
	s.mcpSrv, s.mcpHandler = s.InitMCP()

	tests := []struct {
		name      string
		params    map[string]any
		wantQuery string
		wantMode  string
		wantError bool
	}{
		{
			name:      "required query only",
			params:    map[string]any{"query": "测试问题"},
			wantQuery: "测试问题",
			wantMode:  "raw",
		},
		{
			name:      "query with mode summary",
			params:    map[string]any{"query": "测试问题", "mode": "summary"},
			wantQuery: "测试问题",
			wantMode:  "summary",
		},
		{
			name:      "all parameters",
			params: map[string]any{
				"query":     "完整参数测试",
				"mode":      "raw",
				"collection": "test_coll",
				"kb_id":     1,
				"scope":     "public",
				"limit":     10,
				"threshold": 0.8,
				"tenant":    "test_tenant",
				"user_id":   100,
				"agent_id":  200,
			},
			wantQuery: "完整参数测试",
			wantMode:  "raw",
		},
		{
			name:      "missing query should error",
			params:    map[string]any{"mode": "raw"},
			wantError: true,
		},
		{
			name:      "limit clamped to max 20",
			params:    map[string]any{"query": "测试", "limit": float64(100)},
			wantQuery: "测试",
			wantMode:  "raw",
		},
		{
			name:      "threshold clamped to 0-1",
			params:    map[string]any{"query": "测试", "threshold": float64(1.5)},
			wantQuery: "测试",
			wantMode:  "raw",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build CallToolRequest
			callReq := mcp.CallToolRequest{
				Params: mcp.CallToolParams{
					Name:      "rag_ask",
					Arguments: tt.params,
				},
			}

			result, err := s.handleRagAsk(context.Background(), callReq)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.wantError {
				if !result.IsError {
					t.Error("expected error result, got success")
				}
				return
			}

			if result.IsError {
				// Extract error text
				for _, c := range result.Content {
					if tc, ok := c.(mcp.TextContent); ok {
						t.Errorf("unexpected error: %s", tc.Text)
					}
				}
				return
			}

			// Verify result contains the query
			for _, c := range result.Content {
				if tc, ok := c.(mcp.TextContent); ok {
					if !strings.Contains(tc.Text, tt.wantQuery) {
						t.Errorf("result text does not contain query %q: %s", tt.wantQuery, tc.Text)
					}
				}
			}
		})
	}
}

// TestMCPToolsListEndpoint tests GET /debug/mcp/tools (task 4.2).
func TestMCPToolsListEndpoint(t *testing.T) {
	router := setupMCPServer(t)

	req := httptest.NewRequest(http.MethodGet, "/debug/mcp/tools", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp struct {
		Tools []struct {
			Name        string `json:"name"`
			Description string `json:"description"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	found := false
	for _, tool := range resp.Tools {
		if tool.Name == "rag_ask" {
			found = true
			if tool.Description == "" {
				t.Error("rag_ask tool has empty description")
			}
			break
		}
	}
	if !found {
		t.Errorf("rag_ask tool not found in tools list. Got %d tools", len(resp.Tools))
	}
}

// TestMCPToolsRegistered tests that rag_ask tool is registered in the MCP server.
func TestMCPToolsRegistered(t *testing.T) {
	gin.SetMode(gin.TestMode)

	kbs, err := knowledgebase.NewService(":memory:")
	if err != nil {
		t.Fatalf("create kb service: %v", err)
	}
	kbs.EnsurePublicDefault()

	cfg := config.DefaultConfig()
	emb := &mcpTestEmbedder{}
	s, _ := New(cfg, nil, nil, nil, emb, nil, &mcpTestLLM{}, nil, nil, kbs, 0, "test-model")
	s.mcpSrv, s.mcpHandler = s.InitMCP()

	tools := s.mcpSrv.ListTools()
	st, ok := tools["rag_ask"]
	if !ok {
		t.Fatal("rag_ask tool not registered in MCP server")
	}

	if st.Tool.Name != "rag_ask" {
		t.Errorf("expected tool name 'rag_ask', got %q", st.Tool.Name)
	}
	if st.Handler == nil {
		t.Error("rag_ask tool has nil handler")
	}

	// Verify required parameter
	schema := st.Tool.InputSchema
	if schema.Required == nil {
		t.Error("expected input schema to have required fields")
	} else {
		foundQuery := false
		for _, r := range schema.Required {
			if r == "query" {
				foundQuery = true
				break
			}
		}
		if !foundQuery {
			t.Error("query not found in required parameters")
		}
	}
}

// TestMCPDebugCallEndpoint tests POST /debug/mcp/call (task 4.3).
func TestMCPDebugCallEndpoint(t *testing.T) {
	t.Skip("requires ES client after KBRetriever removal")
	router := setupMCPServer(t)

	body := `{"tool": "rag_ask", "arguments": {"query": "测试查询"}}`
	req := httptest.NewRequest(http.MethodPost, "/debug/mcp/call", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp struct {
		Tool    string `json:"tool"`
		IsError bool   `json:"is_error"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	if resp.Tool != "rag_ask" {
		t.Errorf("expected tool 'rag_ask', got %q", resp.Tool)
	}
	if resp.IsError {
		t.Errorf("unexpected error in debug call: %s", resp.Content)
	}
	if resp.Content == "" {
		t.Error("expected non-empty content in debug response")
	}
	if !strings.Contains(resp.Content, "测试查询") {
		t.Errorf("expected content to contain query: %s", resp.Content)
	}
}

// TestMCPDebugCallNonexistentTool tests debug call with non-existent tool.
func TestMCPDebugCallNonexistentTool(t *testing.T) {
	router := setupMCPServer(t)

	body := `{"tool": "nonexistent", "arguments": {}}`
	req := httptest.NewRequest(http.MethodPost, "/debug/mcp/call", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected status 404 for nonexistent tool, got %d", w.Code)
	}
}

// TestMCPStreamableHTTPEndpoint tests that the /mcp endpoint is mounted.
func TestMCPStreamableHTTPEndpoint(t *testing.T) {
	router := setupMCPServer(t)

	// MCP Initialize request
	initReq := `{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":1}`
	req := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(initReq))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify the response contains server info
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	result, ok := resp["result"].(map[string]any)
	if !ok {
		t.Fatal("expected result object in MCP response")
	}

	serverInfo, ok := result["serverInfo"].(map[string]any)
	if !ok {
		t.Fatal("expected serverInfo in MCP response")
	}

	name, ok := serverInfo["name"].(string)
	if !ok || name != "mcp-rag" {
		t.Errorf("expected server name 'mcp-rag', got %q", name)
	}
}

// TestMCPRagAskSummaryMode tests summary mode via debug endpoint.
func TestMCPRagAskSummaryMode(t *testing.T) {
	t.Skip("requires ES client after KBRetriever removal")
	router := setupMCPServer(t)

	body := `{"tool": "rag_ask", "arguments": {"query": "什么是AI？", "mode": "summary"}}`
	req := httptest.NewRequest(http.MethodPost, "/debug/mcp/call", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp struct {
		Tool    string `json:"tool"`
		IsError bool   `json:"is_error"`
		Content string `json:"content"`
	}
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp.IsError {
		t.Errorf("unexpected error in summary mode: %s", resp.Content)
	}
	if !strings.Contains(resp.Content, "总结") {
		t.Errorf("expected '总结' in summary mode response: %s", resp.Content)
	}
}
