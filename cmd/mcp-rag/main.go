// Package main is the entry point for mcp-rag-go.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/gin-gonic/gin"
	"github.com/spf13/cobra"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/knowledgebase"
	"github.com/jiaobendaye/mcp-rag-go/internal/rag"
	"github.com/jiaobendaye/mcp-rag-go/internal/server"
)

var (
	configPath string
)

var rootCmd = &cobra.Command{
	Use:   "mcp-rag",
	Short: "MCP-RAG: RAG service with MCP protocol support",
}

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the MCP-RAG HTTP server",
	RunE:  runServe,
}

func init() {
	serveCmd.Flags().StringVarP(&configPath, "config", "c", "config.yaml", "path to config file")
	rootCmd.AddCommand(serveCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runServe(cmd *cobra.Command, args []string) error {
	ctx := context.Background()

	// 1. Load configuration
	cfg, err := config.Load(configPath)
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	log.Printf("Starting MCP-RAG server on port %d", cfg.HTTPPort)

	// 2. Connect to Elasticsearch
	esClient, err := elasticsearch.NewClient(elasticsearch.Config{
		Addresses: []string{cfg.ESUrl},
	})
	if err != nil {
		return fmt.Errorf("create es client: %w", err)
	}

	// Verify ES connection
	if res, err := esClient.Ping(); err != nil {
		return fmt.Errorf("es ping: %w", err)
	} else {
		res.Body.Close()
	}

	// 3. Create knowledge base service
	kbService, err := knowledgebase.NewService(cfg.KnowledgeBaseDBPath)
	if err != nil {
		return fmt.Errorf("create kb service: %w", err)
	}
	defaultKB, err := kbService.EnsurePublicDefault()
	if err != nil {
		return fmt.Errorf("ensure default kb: %w", err)
	}
	log.Printf("Default knowledge base: %s (index: %s)", defaultKB.Name, defaultKB.IndexName())

	// 4. Create ES indexer
	esIndexer := rag.NewES8Indexer(esClient, defaultKB.IndexName())

	// Probe embedding dimensions
	dims, err := probeEmbeddingDims(ctx, cfg)
	if err != nil {
		return fmt.Errorf("probe embedding dims: %w", err)
	}
	log.Printf("Detected embedding dimensions: %d", dims)

	if err := esIndexer.EnsureIndex(ctx, dims); err != nil {
		return fmt.Errorf("ensure index: %w", err)
	}

	// 5. Create embedder
	embedder := &openAIEmbedder{
		baseURL:  cfg.EmbeddingBaseURL,
		apiKey:   cfg.EmbeddingAPIKey,
		model:    cfg.EmbeddingModel,
		provider: cfg.EmbeddingProvider,
	}

	// 5. Create LLM
	llm := &openAILLM{
		baseURL: cfg.LLMBaseURL,
		apiKey:  cfg.LLMAPIKey,
		model:   cfg.LLMModel,
	}

	// 6. Create pipeline and services
	pipeline := rag.NewIndexPipeline(embedder, esIndexer, cfg.ChunkSize, cfg.ChunkOverlap)
	chatSvc := rag.NewChatService(esIndexer, embedder, llm)

	// 7. Setup HTTP server
	gin.SetMode(gin.ReleaseMode)
	srv := server.New(cfg, pipeline, chatSvc, esIndexer, embedder, kbService, esIndexer)
	router := srv.Setup()

	// 8. Start server
	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.HTTPPort),
		Handler: router,
	}

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down server...")
		httpServer.Shutdown(context.Background())
	}()

	log.Printf("MCP-RAG server listening on http://0.0.0.0:%d", cfg.HTTPPort)
	log.Printf("Endpoints:")
	log.Printf("  GET  /health")
	log.Printf("  POST /add-document")
	log.Printf("  POST /upload-files")
	log.Printf("  GET  /search?query=...&limit=5")
	log.Printf("  POST /chat")

	return httpServer.ListenAndServe()
}

// probeEmbeddingDims determines embedding dimensions by embedding a probe text.
func probeEmbeddingDims(ctx context.Context, cfg *config.Config) (int, error) {
	emb := &openAIEmbedder{
		baseURL:  cfg.EmbeddingBaseURL,
		apiKey:   cfg.EmbeddingAPIKey,
		model:    cfg.EmbeddingModel,
		provider: cfg.EmbeddingProvider,
	}

	vecs, err := emb.EmbedStrings(ctx, []string{"probe"})
	if err != nil {
		return 0, fmt.Errorf("probe embedding: %w", err)
	}
	if len(vecs) == 0 {
		return 0, fmt.Errorf("empty probe response")
	}
	return len(vecs[0]), nil
}

// openAIEmbedder implements rag.Embedder using OpenAI-compatible API.
type openAIEmbedder struct {
	baseURL  string
	apiKey   string
	model    string
	provider string
	httpCli  *http.Client
}

func (e *openAIEmbedder) getClient() *http.Client {
	if e.httpCli == nil {
		e.httpCli = &http.Client{}
	}
	return e.httpCli
}

func (e *openAIEmbedder) base() string {
	if e.baseURL != "" {
		return e.baseURL
	}
	switch e.provider {
	case "ark", "doubao":
		return "https://ark.cn-beijing.volces.com/api/v3"
	case "dashscope", "aliyun":
		return "https://dashscope.aliyuncs.com/compatible-mode/v1"
	default:
		return "https://api.openai.com/v1"
	}
}

func (e *openAIEmbedder) EmbedStrings(ctx context.Context, texts []string) ([][]float64, error) {
	reqBody := map[string]any{
		"model": e.modelName(),
		"input": texts,
	}
	data, _ := json.Marshal(reqBody)

	req, err := http.NewRequestWithContext(ctx, "POST", e.base()+"/embeddings", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.getClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embed api error: status=%d body=%s", resp.StatusCode, string(body))
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}

	vecs := make([][]float64, len(result.Data))
	for i, d := range result.Data {
		vecs[i] = d.Embedding
	}
	return vecs, nil
}

func (e *openAIEmbedder) modelName() string {
	if e.model != "" {
		return e.model
	}
	return "text-embedding-3-small"
}

// openAILLM implements rag.LLMGenerator using OpenAI-compatible API.
type openAILLM struct {
	baseURL string
	apiKey  string
	model   string
	httpCli *http.Client
}

func (l *openAILLM) getClient() *http.Client {
	if l.httpCli == nil {
		l.httpCli = &http.Client{}
	}
	return l.httpCli
}

func (l *openAILLM) base() string {
	if l.baseURL != "" {
		return l.baseURL
	}
	return "https://api.openai.com/v1"
}

func (l *openAILLM) Generate(ctx context.Context, prompt string) (string, error) {
	reqBody := map[string]any{
		"model": l.modelName(),
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	data, _ := json.Marshal(reqBody)

	req, err := http.NewRequestWithContext(ctx, "POST", l.base()+"/chat/completions", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+l.apiKey)

	resp, err := l.getClient().Do(req)
	if err != nil {
		return "", fmt.Errorf("llm request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("llm api error: status=%d body=%s", resp.StatusCode, string(body))
	}

	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode llm response: %w", err)
	}

	if len(result.Choices) == 0 {
		return "", fmt.Errorf("empty response from LLM")
	}

	return result.Choices[0].Message.Content, nil
}

func (l *openAILLM) modelName() string {
	if l.model != "" {
		return l.model
	}
	return "gpt-4o-mini"
}
