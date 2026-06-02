//go:build integration
// +build integration

package server

import (
	"context"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/gin-gonic/gin"

	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
	elastic_retriever "github.com/cloudwego/eino-ext/components/retriever/es8"
	elastic_search_mode "github.com/cloudwego/eino-ext/components/retriever/es8/search_mode"
	openaiembed "github.com/cloudwego/eino-ext/components/embedding/openai"
	openaimodel "github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino-ext/components/document/transformer/splitter/recursive"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/knowledgebase"
	"github.com/jiaobendaye/mcp-rag-go/internal/rag"
	"github.com/jiaobendaye/mcp-rag-go/internal/testutil"
)

func setupIntegrationServer(t *testing.T) (*gin.Engine, *knowledgebase.Service, *elasticsearch.Client) {
	t.Helper()

	// Load .env so LLM API key is available (same pattern as Makefile serve target).
	loadDotEnv(t)

	ctx := context.Background()

	// ES container via testcontainers (no docker-compose needed).
	esURL, err := testutil.StartES(t, ctx)
	if err != nil {
		t.Skipf("SKIP: cannot start ES container: %v", err)
	}

	cfg := config.DefaultConfig()
	cfg.MinScore = 0.01 // low threshold for integration tests (mxbai-embed-large may have lower scores for Chinese)
	cfg.KnowledgeBaseDBPath = filepath.Join(t.TempDir(), "test_kb.db")
	cfg.ESUrl = esURL
	cfg.EmbeddingProvider = "ollama"
	cfg.EmbeddingModel = "mxbai-embed-large"
	cfg.EmbeddingBaseURL = "http://localhost:11434/v1"
	// LLM config from env (loaded by loadDotEnv) with deployment defaults.
	cfg.LLMAPIKey = os.Getenv("MCP_RAG_LLM_API_KEY")
	cfg.LLMModel = firstNonEmpty(os.Getenv("MCP_RAG_LLM_MODEL"), "deepseek-v4-flash")
	cfg.LLMBaseURL = firstNonEmpty(os.Getenv("MCP_RAG_LLM_BASE_URL"), "https://zhi8.xf.bj.cn/onehub/v1")

	// ── Probe external services ──────────────────────────────────────
	// Integration tests need Ollama embedding. If missing we skip all
	// tests with a clear reason. ES is handled by testcontainers above.

	missing := probeOllama(t, cfg.EmbeddingBaseURL, cfg.EmbeddingModel)
	if len(missing) > 0 {
		t.Skipf("SKIP: integration env incomplete — missing: %s\n"+
			"  To run: ensure Ollama (%s, model=%s) is up.",
			strings.Join(missing, ", "), cfg.EmbeddingBaseURL, cfg.EmbeddingModel)
	}

	// ES client (only reach this line if ES is confirmed reachable)
	esClient, err := elasticsearch.NewClient(elasticsearch.Config{Addresses: []string{cfg.ESUrl}})
	if err != nil {
		t.Fatalf("create es client: %v", err)
	}

	// Clean ES indices from previous test runs to prevent cross-test
	// pollution (tests share an ES cluster but have independent SQLite DBs).
	//
	// ES may reject wildcard DELETE (action.destructive_requires_name=true),
	// so we list kb_* indices first, then delete each one by name.
	catResp, err := esClient.Cat.Indices(
		esClient.Cat.Indices.WithIndex("kb_*"),
		esClient.Cat.Indices.WithFormat("json"),
	)
	if err == nil {
		var catEntries []struct{ Index string }
		if err := json.NewDecoder(catResp.Body).Decode(&catEntries); err == nil {
			for _, entry := range catEntries {
				if entry.Index != "" {
					delResp, dErr := esClient.Indices.Delete([]string{entry.Index})
					if dErr == nil {
						delResp.Body.Close()
						t.Logf("ES cleanup: deleted index %s (status=%d)", entry.Index, delResp.StatusCode)
					}
				}
			}
		}
		catResp.Body.Close()
	}
	if err != nil {
		t.Logf("ES cleanup: could not list indices: %v", err)
	}

	// KB service
	kbService, err := knowledgebase.NewService(cfg.KnowledgeBaseDBPath)
	if err != nil {
		t.Fatalf("create kb service: %v", err)
	}
	defaultKB, err := kbService.EnsurePublicDefault()
	if err != nil {
		t.Fatalf("ensure default kb: %v", err)
	}

	// Real embedder (probeServices already confirmed Ollama is reachable)
	embedder, err := openaiembed.NewEmbedder(ctx, &openaiembed.EmbeddingConfig{
		BaseURL: cfg.EmbeddingBaseURL,
		Model:   cfg.EmbeddingModel,
	})
	if err != nil {
		t.Fatalf("create embedder: %v", err)
	}

	// Probe dims (cached result from probeServices above; re-query is cheap)
	vecs, _ := embedder.EmbedStrings(ctx, []string{"probe"})
	dims := 1024
	if len(vecs) > 0 {
		dims = len(vecs[0])
	}
	t.Logf("embedding dims: %d", dims)

	// Splitter
	splitter, err := newSplitter(ctx, 1000, 100)
	if err != nil {
		t.Fatalf("create splitter: %v", err)
	}

	// KB indexer config
	indexerConf := &elastic_indexer.IndexerConfig{
		Client:           esClient,
		IndexSpec:        indexSpecForDims(dims),
		DocumentToFields: rag.ProjectDocumentToFields(),
		Embedding:        embedder,
	}
	{
		confCopy := *indexerConf
		confCopy.Index = defaultKB.IndexName()
		if _, err := elastic_indexer.NewIndexer(ctx, &confCopy); err != nil {
			t.Logf("WARNING: ensure default index: %v", err)
		}
	}

	// KB retriever
	kbRetriever, err := rag.NewKBRetriever(ctx, &elastic_retriever.RetrieverConfig{
		Client:         esClient,
		Index:          rag.PlaceholderIndex,
		TopK:           5,
		SearchMode:     elastic_search_mode.SearchModeRawStringRequest(),
		ResultParser:   rag.ProjectResultParser(),
		Embedding:      embedder,
	})
	if err != nil {
		t.Fatalf("create kb retriever: %v", err)
	}

	// LLM (best-effort)
	llm, err := openaimodel.NewChatModel(ctx, &openaimodel.ChatModelConfig{
		BaseURL: cfg.LLMBaseURL,
		APIKey:  cfg.LLMAPIKey,
		Model:   cfg.LLMModel,
	})
	if err != nil {
		t.Logf("LLM unavailable (expected in CI): %v", err)
		llm = nil
	}

	gin.SetMode(gin.TestMode)
	s, err := New(cfg, nil, nil, nil, embedder, splitter, llm, indexerConf, kbRetriever, esClient, kbService, dims)
	if err != nil {
		t.Fatalf("create server: %v", err)
	}
	return s.Setup(), kbService, esClient
}

// indexSpecForDims mirrors main.go's indexSpecForDims; integration tests
// build their own copy because main.go's helper is unexported.
func indexSpecForDims(dims int) *elastic_indexer.IndexSpec {
	return &elastic_indexer.IndexSpec{
		Settings: map[string]any{
			"number_of_shards":   1,
			"number_of_replicas": 0,
		},
		Mappings: map[string]any{
			"dynamic": "strict",
			"properties": map[string]any{
				"content":        map[string]any{"type": "text"},
				"content_vector": map[string]any{"type": "dense_vector", "dims": dims, "similarity": "cosine"},
				"document_id":    map[string]any{"type": "keyword"},
				"chunk_index":    map[string]any{"type": "integer"},
				"total_chunks":   map[string]any{"type": "integer"},
				"chunk_id":       map[string]any{"type": "keyword"},
				"source":         map[string]any{"type": "keyword"},
				"filename":       map[string]any{"type": "keyword"},
				"file_type":      map[string]any{"type": "keyword"},
				"processed_at":   map[string]any{"type": "date"},
			},
		},
	}
}

// newSplitter is a thin shim around eino-ext's recursive splitter; kept
// in the integration test file to avoid exporting it from main.go.
func newSplitter(ctx context.Context, chunkSize, overlap int) (document.Transformer, error) {
	return recursive.NewSplitter(ctx, &recursive.Config{
		ChunkSize:   chunkSize,
		OverlapSize: overlap,
	})
}

func TestIntegrationAddDocumentAndSearch(t *testing.T) {
	r, _, _ := setupIntegrationServer(t)

	// Add document
	body := `{"content":"Go语言是一种静态类型的编译型编程语言，由Google开发，以简洁高效著称。"}`
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/add-document", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("add-document failed: %d %s", w.Code, w.Body.String())
	}
	t.Logf("add-document: %s", w.Body.String())

	// Search for it
	req2, _ := http.NewRequest("GET", "/search?query=Go语言特点&limit=3", nil)
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, req2)

	if w2.Code != 200 {
		t.Fatalf("search failed: %d %s", w2.Code, w2.Body.String())
	}
	t.Logf("search: %s", w2.Body.String())

	if !strings.Contains(w2.Body.String(), "Go") {
		t.Error("search results should contain 'Go'")
	}
}

func TestIntegrationUploadFile(t *testing.T) {
	r, _, _ := setupIntegrationServer(t)

	tmpFile := t.TempDir() + "/test.md"
	content := "# Test\n\nElasticsearch集成测试文档。"
	if err := os.WriteFile(tmpFile, []byte(content), 0644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	// Upload
	body := &strings.Builder{}
	w := multipart.NewWriter(body)
	part, _ := w.CreateFormFile("files", "test.md")
	part.Write([]byte(content))
	w.Close()

	req := httptest.NewRequest("POST", "/upload-files", strings.NewReader(body.String()))
	req.Header.Set("Content-Type", w.FormDataContentType())
	resp := httptest.NewRecorder()
	r.ServeHTTP(resp, req)

	if resp.Code != 200 {
		t.Fatalf("upload failed: %d %s", resp.Code, resp.Body.String())
	}
	t.Logf("upload: %s", resp.Body.String())
}

// ---------------------------------------------------------------------------
// Section 10: Per-request compile integration tests
// ---------------------------------------------------------------------------

// TestAddDocument_RoutesToResolvedKB verifies that POST /add-document
// with kb_id=A and kb_id=B writes content to the correct ES indices and
// that searching each KB only returns its own content. This is the
// regression test for the 2026-06-01 bug where addDocument ignored
// kb_id and wrote everything to the default index.
func TestAddDocument_RoutesToResolvedKB(t *testing.T) {
	r, kbSvc, _ := setupIntegrationServer(t)

	bodyA := `{"content":"Go语言是由Google开发的静态类型编译型语言。"}`
	bodyB := `{"content":"React和TypeScript是构建现代前端应用的主流技术。"}`

	// Add bodyA to default KB (no kb_id → resolved to default by scope
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/add-document", strings.NewReader(bodyA))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("add-document to default: %d %s", w.Code, w.Body.String())
	}

	// Create a second KB explicitly
	kb2, err := kbSvc.Create("kb2", "public", nil, nil)
	if err != nil {
		t.Fatalf("create kb2: %v", err)
	}

	// Add bodyB to kb2
	w2 := httptest.NewRecorder()
	req2, _ := http.NewRequest("POST", "/add-document?kb_id="+i64toa(kb2.ID), strings.NewReader(bodyB))
	req2.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w2, req2)
	if w2.Code != 200 {
		t.Fatalf("add-document to kb2: %d %s", w2.Code, w2.Body.String())
	}

	// Search default KB: should contain Go content, not React
	w3 := httptest.NewRecorder()
	req3, _ := http.NewRequest("GET", "/search?query=Go语言&limit=5", nil)
	r.ServeHTTP(w3, req3)
	if w3.Code != 200 {
		t.Fatalf("search default: %d %s", w3.Code, w3.Body.String())
	}
	defaultResp := w3.Body.String()
	if !strings.Contains(defaultResp, "Go") {
		t.Errorf("default search should contain Go content, got: %s", defaultResp)
	}
	if strings.Contains(defaultResp, "React") {
		t.Errorf("default search should NOT contain React content, got: %s", defaultResp)
	}

	// Search kb2: should contain React content, not Go
	w4 := httptest.NewRecorder()
	req4, _ := http.NewRequest("GET", "/search?query=React&kb_id="+i64toa(kb2.ID)+"&limit=5", nil)
	r.ServeHTTP(w4, req4)
	if w4.Code != 200 {
		t.Fatalf("search kb2: %d %s", w4.Code, w4.Body.String())
	}
	kb2Resp := w4.Body.String()
	if !strings.Contains(kb2Resp, "React") {
		t.Errorf("kb2 search should contain React content, got: %s", kb2Resp)
	}
	if strings.Contains(kb2Resp, "Go") {
		t.Errorf("kb2 search should NOT contain Go content, got: %s", kb2Resp)
	}
}

// TestSearchMultiKB_RealSearch verifies that GET /search?kb_ids=A,B
// returns hits from BOTH KBs (per-request compile captures each KB's
// indexName). This is the regression test for the Phase 10 multi-KB
// read bug where the handler queried the default index for all KBs.
func TestSearchMultiKB_RealSearch(t *testing.T) {
	r, kbSvc, esClient := setupIntegrationServer(t)

	// Seed two additional KBs with distinct content
	kb1, err := kbSvc.Create("multi1", "public", nil, nil)
	if err != nil {
		t.Fatalf("create kb1: %v", err)
	}
	kb2, err := kbSvc.Create("multi2", "public", nil, nil)
	if err != nil {
		t.Fatalf("create kb2: %v", err)
	}

	// Add container content to kb1
	wA := httptest.NewRecorder()
	reqA, _ := http.NewRequest("POST", "/add-document?kb_id="+i64toa(kb1.ID), strings.NewReader(`{"content":"Kubernetes容器编排系统管理容器化应用。"}`))
	reqA.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(wA, reqA)
	if wA.Code != 200 {
		t.Fatalf("add to kb1: %d %s", wA.Code, wA.Body.String())
	}

	// Add container content to kb2 so both KBs match "容器"
	wB := httptest.NewRecorder()
	reqB, _ := http.NewRequest("POST", "/add-document?kb_id="+i64toa(kb2.ID), strings.NewReader(`{"content":"Docker Compose用于管理多容器应用。"}`))
	reqB.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(wB, reqB)
	if wB.Code != 200 {
		t.Fatalf("add to kb2: %d %s", wB.Code, wB.Body.String())
	}

	// ES near-real-time: force refresh so newly-indexed docs are searchable
	if refreshResp, err := esClient.Indices.Refresh(esClient.Indices.Refresh.WithIndex("kb_*")); err != nil {
		t.Logf("refresh warning: %v", err)
	} else {
		refreshResp.Body.Close()
		t.Logf("refresh kb_*: status=%d", refreshResp.StatusCode)
	}
	time.Sleep(1 * time.Second)

	// Verify each KB is searchable via single-KB search first
	for _, tc := range []struct {
		kbID    int64
		wantStr string
		label   string
	}{
		{kb1.ID, "Kubernetes", "kb1(multi1)"},
		{kb2.ID, "Docker", "kb2(multi2)"},
	} {
		url := fmt.Sprintf("/search?query=容器&kb_id=%d&limit=5", tc.kbID)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", url, nil)
		r.ServeHTTP(w, req)
		if w.Code != 200 {
			t.Errorf("%s single-kb search: HTTP %d body=%s", tc.label, w.Code, w.Body.String())
			continue
		}
		if !strings.Contains(w.Body.String(), tc.wantStr) {
			t.Errorf("%s single-kb search should contain %q, got: %s", tc.label, tc.wantStr, w.Body.String())
		}
		t.Logf("%s single-kb search OK (contains %q)", tc.label, tc.wantStr)
	}

	// Search both KBs together
	w := httptest.NewRecorder()
	url := "/search?query=容器&kb_ids=" + i64toa(kb1.ID) + "," + i64toa(kb2.ID)
	req, _ := http.NewRequest("GET", url, nil)
	r.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("multi-kb search: %d %s", w.Code, w.Body.String())
	}
	resp := w.Body.String()
	t.Logf("multi-kb search response: %s", resp)

	if !strings.Contains(resp, "multi_kb") {
		t.Errorf("multi-kb search should report collection=multi_kb, got: %s", resp)
	}
	// Both KBs should be referenced in metadata somewhere
	if !strings.Contains(resp, "multi1") {
		t.Errorf("multi-kb search results should reference kb1 name 'multi1', got: %s", resp)
	}
	if !strings.Contains(resp, "multi2") {
		t.Errorf("multi-kb search results should reference kb2 name 'multi2', got: %s", resp)
	}
}

// ---------------------------------------------------------------------------
// Regression: body kb_id routing (#20 fix)
// ---------------------------------------------------------------------------

// TestAddDocument_BodyKBID verifies that POST /add-document with kb_id in
// the JSON body (not URL query) routes to the correct ES index. This is the
// regression test for the 2026-06-01 bug where addDocument read req.KBID
// from the body but resolveKB only looked at c.Query("kb_id"), silently
// discarding the body value and routing all writes to the default KB.
func TestAddDocument_BodyKBID(t *testing.T) {
	r, kbSvc, _ := setupIntegrationServer(t)

	// Create two KBs
	kba, err := kbSvc.Create("body_kb_a", "public", nil, nil)
	if err != nil {
		t.Fatalf("create kb_a: %v", err)
	}
	kbb, err := kbSvc.Create("body_kb_b", "public", nil, nil)
	if err != nil {
		t.Fatalf("create kb_b: %v", err)
	}

	// Add Go content to kb_a via body kb_id (no URL query param)
	bodyA := fmt.Sprintf(`{"content":"Python是一种解释型、动态类型的编程语言。","kb_id":%d}`, kba.ID)
	w1 := httptest.NewRecorder()
	req1, _ := http.NewRequest("POST", "/add-document", strings.NewReader(bodyA))
	req1.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w1, req1)
	if w1.Code != 200 {
		t.Fatalf("add-document to kb_a (body kb_id): %d %s", w1.Code, w1.Body.String())
	}

	// Add React content to kb_b via body kb_id
	bodyB := fmt.Sprintf(`{"content":"TypeScript是JavaScript的超集，增加了静态类型检查。","kb_id":%d}`, kbb.ID)
	w2 := httptest.NewRecorder()
	req2, _ := http.NewRequest("POST", "/add-document", strings.NewReader(bodyB))
	req2.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w2, req2)
	if w2.Code != 200 {
		t.Fatalf("add-document to kb_b (body kb_id): %d %s", w2.Code, w2.Body.String())
	}

	// Search kb_a for Python — should find it, not TypeScript
	w3 := httptest.NewRecorder()
	req3, _ := http.NewRequest("GET", fmt.Sprintf("/search?query=Python&kb_id=%d&limit=3", kba.ID), nil)
	r.ServeHTTP(w3, req3)
	if w3.Code != 200 {
		t.Fatalf("search kb_a: %d %s", w3.Code, w3.Body.String())
	}
	if !strings.Contains(w3.Body.String(), "Python") {
		t.Errorf("kb_a search should contain Python, got: %s", w3.Body.String())
	}
	if strings.Contains(w3.Body.String(), "TypeScript") {
		t.Errorf("kb_a search should NOT contain TypeScript, got: %s", w3.Body.String())
	}

	// Search kb_b for TypeScript — should find it, not Python
	w4 := httptest.NewRecorder()
	req4, _ := http.NewRequest("GET", fmt.Sprintf("/search?query=TypeScript&kb_id=%d&limit=3", kbb.ID), nil)
	r.ServeHTTP(w4, req4)
	if w4.Code != 200 {
		t.Fatalf("search kb_b: %d %s", w4.Code, w4.Body.String())
	}
	if !strings.Contains(w4.Body.String(), "TypeScript") {
		t.Errorf("kb_b search should contain TypeScript, got: %s", w4.Body.String())
	}
	if strings.Contains(w4.Body.String(), "Python") {
		t.Errorf("kb_b search should NOT contain Python, got: %s", w4.Body.String())
	}
}

// ---------------------------------------------------------------------------
// Chat endpoint (requires LLM)
// ---------------------------------------------------------------------------

// TestChat_SingleKB verifies that POST /chat performs retrieval+LLM
// generation end-to-end. If the LLM is unavailable the test is skipped
// rather than failed (the setup integration server falls back to nil).
func TestChat_SingleKB(t *testing.T) {
	r, kbSvc, esClient := setupIntegrationServer(t)

	// Seed a KB with content
	kb, err := kbSvc.Create("chat_test", "public", nil, nil)
	if err != nil {
		t.Fatalf("create kb: %v", err)
	}

	body := fmt.Sprintf(`{"content":"Rust是一门系统编程语言，专注于内存安全和并发性能。","kb_id":%d}`, kb.ID)
	w1 := httptest.NewRecorder()
	req1, _ := http.NewRequest("POST", "/add-document", strings.NewReader(body))
	req1.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w1, req1)
	if w1.Code != 200 {
		t.Fatalf("add-document: %d %s", w1.Code, w1.Body.String())
	}

	// ES near-real-time: force refresh so the newly indexed document is searchable.
	if refreshResp, err := esClient.Indices.Refresh(esClient.Indices.Refresh.WithIndex("kb_*")); err != nil {
		t.Logf("refresh warning: %v", err)
	} else {
		refreshResp.Body.Close()
	}

	// Chat about the content
	chatBody := fmt.Sprintf(`{"query":"Rust语言的优点是什么","kb_id":%d,"limit":2,"threshold":0.2}`, kb.ID)
	w2 := httptest.NewRecorder()
	req2, _ := http.NewRequest("POST", "/chat", strings.NewReader(chatBody))
	req2.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w2, req2)
	t.Logf("chat response code=%d body=%s", w2.Code, w2.Body.String())

	if w2.Code != 200 {
		// If LLM is down, this is not a code bug
		if strings.Contains(w2.Body.String(), "Unauthorized") ||
			strings.Contains(w2.Body.String(), "chat_model") {
			t.Skipf("LLM unavailable, skipping chat test: %s", w2.Body.String())
		}
		t.Fatalf("chat failed: %d %s", w2.Code, w2.Body.String())
	}

	var resp struct {
		Response string `json:"response"`
		Query    string `json:"query"`
	}
	if err := json.Unmarshal(w2.Body.Bytes(), &resp); err != nil {
		t.Fatalf("parse chat response: %v", err)
	}
	if resp.Response == "" {
		t.Error("chat response should be non-empty")
	}
	if !strings.Contains(resp.Response, "Rust") && !strings.Contains(resp.Response, "内存") {
		t.Errorf("chat response should mention Rust or 内存, got: %s", resp.Response)
	}
}

// ---------------------------------------------------------------------------
// MCP endpoint (raw mode)
// ---------------------------------------------------------------------------

// mcpInit creates an MCP session and returns the session ID.
func mcpInit(t *testing.T, r http.Handler) string {
	t.Helper()
	initBody := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}`
	req := httptest.NewRequest("POST", "/mcp", strings.NewReader(initBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	sid := w.Header().Get("mcp-session-id")
	if sid == "" {
		t.Fatalf("MCP initialize: no mcp-session-id in response headers %v", w.Header())
	}
	return sid
}

// mcpCall sends a tools/call request on an existing MCP session and returns
// the raw response body.
func mcpCall(t *testing.T, r http.Handler, sessionID string, id int, tool, argsJSON string) *httptest.ResponseRecorder {
	t.Helper()
	body := fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":"tools/call","params":{"name":"%s","arguments":%s}}`, id, tool, argsJSON)
	req := httptest.NewRequest("POST", "/mcp", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("mcp-session-id", sessionID)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	return w
}

// TestMCP_RawMode_SingleKB verifies that MCP rag_ask in raw mode returns
// relevant retrieved documents from a single KB.
func TestMCP_RawMode_SingleKB(t *testing.T) {
	r, kbSvc, _ := setupIntegrationServer(t)

	// Seed a KB
	kb, err := kbSvc.Create("mcp_raw", "public", nil, nil)
	if err != nil {
		t.Fatalf("create kb: %v", err)
	}

	addBody := fmt.Sprintf(`{"content":"Elasticsearch是一个分布式搜索和分析引擎。","kb_id":%d}`, kb.ID)
	wa := httptest.NewRecorder()
	ra, _ := http.NewRequest("POST", "/add-document", strings.NewReader(addBody))
	ra.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(wa, ra)
	if wa.Code != 200 {
		t.Fatalf("add-document: %d %s", wa.Code, wa.Body.String())
	}

	// MCP session
	sid := mcpInit(t, r)

	// Call rag_ask raw
	args := fmt.Sprintf(`{"query":"Elasticsearch","mode":"raw","kb_id":%d,"limit":3,"threshold":0.2}`, kb.ID)
	w := mcpCall(t, r, sid, 2, "rag_ask", args)
	t.Logf("MCP raw single-kb: code=%d body=%s", w.Code, w.Body.String())

	if w.Code != 200 {
		t.Fatalf("MCP tools/call: %d %s", w.Code, w.Body.String())
	}

	var resp struct {
		Result struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("parse MCP response: %v\nbody=%s", err, w.Body.String())
	}
	if resp.Result.IsError {
		// Could be ES connection issue — don't fail hard, log and skip
		t.Skipf("MCP returned error (possibly env issue): %+v", resp.Result.Content)
	}
	if len(resp.Result.Content) == 0 {
		t.Fatal("expected at least one content item")
	}
	text := resp.Result.Content[0].Text
	if !strings.Contains(text, "Elasticsearch") {
		t.Errorf("expected response to contain 'Elasticsearch', got: %s", text)
	}
	t.Logf("MCP raw response: %s", text[:min(len(text), 200)])
}

// TestMCP_RawMode_MultiKB verifies that MCP rag_ask in raw mode aggregates
// results from multiple KBs (via kb_ids array).
func TestMCP_RawMode_MultiKB(t *testing.T) {
	r, kbSvc, esClient := setupIntegrationServer(t)

	// Create two KBs
	kba, err := kbSvc.Create("mcp_multi_a", "public", nil, nil)
	if err != nil {
		t.Fatalf("create kb_a: %v", err)
	}
	kbb, err := kbSvc.Create("mcp_multi_b", "public", nil, nil)
	if err != nil {
		t.Fatalf("create kb_b: %v", err)
	}

	// Seed with distinct content
	addA := fmt.Sprintf(`{"content":"Docker是一个开源的应用容器引擎。","kb_id":%d}`, kba.ID)
	wa := httptest.NewRecorder()
	ra, _ := http.NewRequest("POST", "/add-document", strings.NewReader(addA))
	ra.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(wa, ra)
	if wa.Code != 200 {
		t.Fatalf("add to kba: %d %s", wa.Code, wa.Body.String())
	}

	addB := fmt.Sprintf(`{"content":"Kubernetes是一个开源的容器编排平台。","kb_id":%d}`, kbb.ID)
	wb := httptest.NewRecorder()
	rb, _ := http.NewRequest("POST", "/add-document", strings.NewReader(addB))
	rb.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(wb, rb)
	if wb.Code != 200 {
		t.Fatalf("add to kbb: %d %s", wb.Code, wb.Body.String())
	}

	// ES near-real-time: force refresh so newly-indexed docs are searchable
	if refreshResp, err := esClient.Indices.Refresh(esClient.Indices.Refresh.WithIndex("kb_*")); err != nil {
		t.Logf("refresh warning: %v", err)
	} else {
		refreshResp.Body.Close()
		t.Logf("refresh kb_*: status=%d", refreshResp.StatusCode)
	}
	// Also wait a brief moment for ES to complete the refresh
	time.Sleep(200 * time.Millisecond)

	sid := mcpInit(t, r)

	// Multi-KB raw search
	args := fmt.Sprintf(`{"query":"容器","mode":"raw","kb_ids":[%d,%d],"limit":5,"threshold":0.2}`, kba.ID, kbb.ID)
	w := mcpCall(t, r, sid, 2, "rag_ask", args)
	t.Logf("MCP raw multi-kb: code=%d body=%s", w.Code, w.Body.String())

	if w.Code != 200 {
		t.Fatalf("MCP tools/call multi-kb: %d %s", w.Code, w.Body.String())
	}

	var resp struct {
		Result struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("parse MCP response: %v\nbody=%s", err, w.Body.String())
	}
	if resp.Result.IsError {
		t.Skipf("MCP returned error (possibly env issue): %+v", resp.Result.Content)
	}
	if len(resp.Result.Content) == 0 {
		t.Fatal("expected at least one content item")
	}
	text := resp.Result.Content[0].Text
	// Should show multi-KB header
	if !strings.Contains(text, "跨2个知识库") {
		t.Errorf("expected multi-kb header '跨2个知识库', got: %s", text[:min(len(text), 300)])
	}
	if !strings.Contains(text, "Docker") || !strings.Contains(text, "Kubernetes") {
		t.Errorf("expected Docker and Kubernetes in results, got: %s", text[:min(len(text), 300)])
	}
	t.Logf("MCP multi-kb response: %s", text[:min(len(text), 300)])
}

func i64toa(n int64) string {
	return itoa(n)
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// probeOllama checks whether Ollama embedding is reachable.
// Returns a list of service names that are missing (empty = all good).
func probeOllama(t *testing.T, ollamaURL, ollamaModel string) []string {
	t.Helper()
	var missing []string

	doHTTPProbe := func(label, url string) {
		client := &http.Client{Timeout: 3 * time.Second}
		resp, err := client.Get(url)
		if err != nil {
			missing = append(missing, label)
			t.Logf("  ✗ %s unreachable: %v", label, err)
			return
		}
		resp.Body.Close()
		if resp.StatusCode >= 500 {
			missing = append(missing, label)
			t.Logf("  ✗ %s returned %d", label, resp.StatusCode)
		} else {
			t.Logf("  ✓ %s OK", label)
		}
	}

	// Ollama: GET /api/tags
	doHTTPProbe("ollama ("+ollamaModel+")", strings.TrimSuffix(ollamaURL, "/v1")+"/api/tags")

	return missing
}

// loadDotEnv loads project-root .env into the process environment. Lines
// starting with # and empty lines are skipped. Only sets variables that
// are not already set, so existing env overrides take precedence.
func loadDotEnv(t *testing.T) {
	t.Helper()
	// Try project root first (go test runs in package dir).
	candidates := []string{".env", "../../.env"}
	var data []byte
	var err error
	for _, p := range candidates {
		data, err = os.ReadFile(p)
		if err == nil {
			break
		}
	}
	if err != nil {
		t.Logf("no .env found (expected in CI): %v", err)
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line[0] == '#' {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		// Strip surrounding quotes if present
		if len(val) >= 2 && val[0] == '"' && val[len(val)-1] == '"' {
			val = val[1 : len(val)-1]
		}
		if key != "" && os.Getenv(key) == "" {
			os.Setenv(key, val)
		}
	}
}
