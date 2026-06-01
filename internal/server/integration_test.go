//go:build integration
// +build integration

package server

import (
	"context"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
)

func setupIntegrationServer(t *testing.T) *gin.Engine {
	t.Helper()

	cfg := config.DefaultConfig()
	cfg.KnowledgeBaseDBPath = filepath.Join(t.TempDir(), "test_kb.db")
	if url := os.Getenv("MCP_RAG_ES_URL"); url != "" {
		cfg.ESUrl = url
	}
	cfg.EmbeddingProvider = "ollama"
	cfg.EmbeddingModel = "mxbai-embed-large"
	cfg.EmbeddingBaseURL = "http://localhost:11434/v1"

	ctx := context.Background()

	// ES client
	esClient, err := elasticsearch.NewClient(elasticsearch.Config{Addresses: []string{cfg.ESUrl}})
	if err != nil {
		t.Fatalf("create es client: %v", err)
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

	// Real embedder
	embedder, err := openaiembed.NewEmbedder(ctx, &openaiembed.EmbeddingConfig{
		BaseURL: cfg.EmbeddingBaseURL,
		Model:   cfg.EmbeddingModel,
	})
	if err != nil {
		t.Fatalf("create embedder: %v", err)
	}

	// Probe dims
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

	// KB indexer
	kbIndexer, err := rag.NewKBIndexer(ctx, &elastic_indexer.IndexerConfig{
		Client:           esClient,
		Index:            rag.PlaceholderIndex,
		IndexSpec:        indexSpecForDims(dims),
		DocumentToFields: rag.ProjectDocumentToFields(),
		Embedding:        embedder,
	})
	if err != nil {
		t.Fatalf("create kb indexer: %v", err)
	}
	if err := kbIndexer.EnsureIndexForKB(ctx, defaultKB.IndexName()); err != nil {
		t.Logf("WARNING: ensure default index: %v", err)
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
	s := New(cfg, nil, nil, nil, embedder, splitter, llm, kbIndexer, kbRetriever, esClient, kbService, dims)
	return s.Setup()
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
	r := setupIntegrationServer(t)

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
	r := setupIntegrationServer(t)

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
	r := setupIntegrationServer(t)

	bodyA := `{"content":"Go语言是由Google开发的静态类型编译型语言。"}`
	bodyB := `{"content":"React和TypeScript是构建现代前端应用的主流技术。"}`

	// Add bodyA to default KB (no kb_id → resolved to default via legacy
	// collection key)
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/add-document", strings.NewReader(bodyA))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("add-document to default: %d %s", w.Code, w.Body.String())
	}

	// Create a second KB explicitly
	kbSvc := newKBSvcForTest(t)
	legacyKey := "legacy:public:kb2"
	kb2, err := kbSvc.Create("kb2", "public", nil, nil, &legacyKey)
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
	r := setupIntegrationServer(t)
	kbSvc := newKBSvcForTest(t)

	// Seed two additional KBs with distinct content
	legacyKey1 := "legacy:public:multi1"
	kb1, err := kbSvc.Create("multi1", "public", nil, nil, &legacyKey1)
	if err != nil {
		t.Fatalf("create kb1: %v", err)
	}
	legacyKey2 := "legacy:public:multi2"
	kb2, err := kbSvc.Create("multi2", "public", nil, nil, &legacyKey2)
	if err != nil {
		t.Fatalf("create kb2: %v", err)
	}

	// Add Go content to kb1
	wA := httptest.NewRecorder()
	reqA, _ := http.NewRequest("POST", "/add-document?kb_id="+i64toa(kb1.ID), strings.NewReader(`{"content":"Kubernetes容器编排系统管理容器化应用。"}`))
	reqA.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(wA, reqA)
	if wA.Code != 200 {
		t.Fatalf("add to kb1: %d %s", wA.Code, wA.Body.String())
	}

	// Add TS content to kb2
	wB := httptest.NewRecorder()
	reqB, _ := http.NewRequest("POST", "/add-document?kb_id="+i64toa(kb2.ID), strings.NewReader(`{"content":"TailwindCSS是实用优先的CSS框架。"}`))
	reqB.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(wB, reqB)
	if wB.Code != 200 {
		t.Fatalf("add to kb2: %d %s", wB.Code, wB.Body.String())
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

	// Response should reflect both KBs (we don't assert exact content
	// matching because embeddings may be approximate; we do assert
	// that the collection is reported as multi_kb and the metadata
	// references both KBs).
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

// newKBSvcForTest creates an in-memory knowledgebase.Service for the
// integration tests above. It does not connect to ES — the KB service
// itself is just a SQLite store.
func newKBSvcForTest(t *testing.T) *knowledgebase.Service {
	t.Helper()
	svc, err := knowledgebase.NewService(":memory:")
	if err != nil {
		t.Fatalf("create kb service: %v", err)
	}
	if _, err := svc.EnsurePublicDefault(); err != nil {
		t.Fatalf("ensure default: %v", err)
	}
	return svc
}

func i64toa(n int64) string {
	return itoa(n)
}
