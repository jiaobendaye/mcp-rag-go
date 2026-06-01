//go:build integration
// +build integration

package server

import (
	"context"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/gin-gonic/gin"

	openaiembed "github.com/cloudwego/eino-ext/components/embedding/openai"
	openaimodel "github.com/cloudwego/eino-ext/components/model/openai"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/knowledgebase"
	"github.com/jiaobendaye/mcp-rag-go/internal/rag"
)

func setupIntegrationServer(t *testing.T) *gin.Engine {
	t.Helper()

	cfg := config.DefaultConfig()
	cfg.KnowledgeBaseDBPath = t.TempDir() + "/test_kb.db"
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

	esIndexer := rag.NewES8Indexer(esClient, defaultKB.IndexName())
	_ = esIndexer.EnsureIndex(ctx, dims)

	// Build index chain
	indexChain, err := rag.BuildIndexChain(ctx, embedder, esIndexer, 1000, 100)
	if err != nil {
		t.Fatalf("build chain: %v", err)
	}

	// LLM
	llm, err := openaimodel.NewChatModel(ctx, &openaimodel.ChatModelConfig{
		BaseURL: cfg.LLMBaseURL,
		APIKey:  cfg.LLMAPIKey,
		Model:   cfg.LLMModel,
	})
	if err != nil {
		t.Logf("LLM unavailable (expected in CI): %v", err)
		llm = nil
	}

	chatSvc := rag.NewChatService(esIndexer, embedder, llm, nil)
	gin.SetMode(gin.TestMode)
	s := New(cfg, nil, nil, nil, indexChain, chatSvc, esIndexer, embedder, kbService, esIndexer, dims)
	return s.Setup()
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
