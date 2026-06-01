// Package main is the entry point for mcp-rag-go.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/gin-gonic/gin"
	"github.com/spf13/cobra"

	openaiembed "github.com/cloudwego/eino-ext/components/embedding/openai"
	openaimodel "github.com/cloudwego/eino-ext/components/model/openai"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/knowledgebase"
	"github.com/jiaobendaye/mcp-rag-go/internal/observability"
	"github.com/jiaobendaye/mcp-rag-go/internal/rag"
	"github.com/jiaobendaye/mcp-rag-go/internal/server"
)

var configPath string

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

	// ConfigManager for hot-reload and CRUD API
	configManager, err := config.NewConfigManager(configPath)
	if err != nil {
		return fmt.Errorf("create config manager: %w", err)
	}
	cfg := configManager.Get()
	log.Printf("Starting MCP-RAG server on port %d", cfg.HTTPPort)

	// Observability
	metricsCollector := observability.NewMetricsCollector(observability.DefaultMetricsConfig())

	// Retrieval cache
	retrievalCache := rag.NewRetrievalCache()

	// Connect to ES
	esClient, err := elasticsearch.NewClient(elasticsearch.Config{Addresses: []string{cfg.ESUrl}})
	if err != nil {
		return fmt.Errorf("create es client: %w", err)
	}
	if res, err := esClient.Ping(); err != nil {
		return fmt.Errorf("es ping: %w", err)
	} else {
		res.Body.Close()
	}

	// Knowledge base service
	kbService, err := knowledgebase.NewService(cfg.KnowledgeBaseDBPath)
	if err != nil {
		return fmt.Errorf("create kb service: %w", err)
	}
	defaultKB, err := kbService.EnsurePublicDefault()
	if err != nil {
		return fmt.Errorf("ensure default kb: %w", err)
	}
	log.Printf("Default KB: %s (%s)", defaultKB.Name, defaultKB.IndexName())

	// eino-ext embedder
	embedder, err := openaiembed.NewEmbedder(ctx, &openaiembed.EmbeddingConfig{
		BaseURL: cfg.EmbeddingBaseURL,
		APIKey:  cfg.EmbeddingAPIKey,
		Model:   cfg.EmbeddingModel,
	})
	if err != nil {
		return fmt.Errorf("create embedder: %w", err)
	}

	// Probe dims
	vecs, err := embedder.EmbedStrings(ctx, []string{"probe"})
	if err != nil {
		return fmt.Errorf("probe embedding: %w", err)
	}
	dims := len(vecs[0])
	log.Printf("Embedding dims: %d", dims)

	// ES indexer
	esIndexer := rag.NewES8Indexer(esClient, defaultKB.IndexName())
	if err := esIndexer.EnsureIndex(ctx, dims); err != nil {
		return fmt.Errorf("ensure index: %w", err)
	}

	// eino-ext LLM
	llm, err := openaimodel.NewChatModel(ctx, &openaimodel.ChatModelConfig{
		BaseURL: cfg.LLMBaseURL,
		APIKey:  cfg.LLMAPIKey,
		Model:   cfg.LLMModel,
	})
	if err != nil {
		return fmt.Errorf("create llm: %w", err)
	}

	// Build index chain
	chain, err := rag.BuildIndexChain(ctx, embedder, esIndexer, cfg.ChunkSize, cfg.ChunkOverlap)
	if err != nil {
		return fmt.Errorf("build index chain: %w", err)
	}

	// Build retrieval graph
	graph, err := rag.BuildRetrievalGraph(ctx, embedder, esIndexer, llm, cfg.SearchMode)
	if err != nil {
		return fmt.Errorf("build retrieval graph: %w", err)
	}

	// Services
	chatSvc := rag.NewChatService(esIndexer, embedder, llm, graph)

	// Setup HTTP
	gin.SetMode(gin.ReleaseMode)
	srv := server.New(cfg, configManager, metricsCollector, retrievalCache, chain, chatSvc, esIndexer, embedder, kbService, esIndexer)
	router := srv.Setup()

	httpServer := &http.Server{Addr: fmt.Sprintf(":%d", cfg.HTTPPort), Handler: router}
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")
		httpServer.Shutdown(context.Background())
	}()

	log.Printf("MCP-RAG listening on :%d", cfg.HTTPPort)
	return httpServer.ListenAndServe()
}
