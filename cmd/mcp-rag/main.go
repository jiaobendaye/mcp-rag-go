// Package main is the entry point for mcp-rag-go.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/gin-gonic/gin"
	"github.com/spf13/cobra"

	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components/document"

	lfcallbacks "github.com/cloudwego/eino-ext/callbacks/langfuse"
	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
	openaiembed "github.com/cloudwego/eino-ext/components/embedding/openai"
	openaimodel "github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino-ext/components/document/transformer/splitter/recursive"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	swappable "github.com/jiaobendaye/mcp-rag-go/internal/embedder"
	"github.com/jiaobendaye/mcp-rag-go/internal/knowledgebase"
	"github.com/jiaobendaye/mcp-rag-go/internal/observability"
	"github.com/jiaobendaye/mcp-rag-go/internal/rag"
	"github.com/jiaobendaye/mcp-rag-go/internal/server"
)

var configPath string

// Build-time variables set via -ldflags.
var (
	version = "dev"
	commit  = "unknown"
	builtAt = "unknown"
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

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print version information",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Printf("mcp-rag %s (commit %s, built %s)\n", version, commit, builtAt)
	},
}

func init() {
	serveCmd.Flags().StringVarP(&configPath, "config", "c", "config.yaml", "path to config file")
	rootCmd.AddCommand(serveCmd, versionCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// indexSpecForDims returns the IndexSpec used by eino-ext to auto-create
// ES indices for our content_vector + content + metadata field shape.
// dims is the embedding dimension.
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

func runServe(cmd *cobra.Command, args []string) error {
	ctx := context.Background()

	// ConfigManager for hot-reload and CRUD API
	configManager, err := config.NewConfigManager(configPath)
	if err != nil {
		return fmt.Errorf("create config manager: %w", err)
	}
	cfg := configManager.Get()
	server.BuildInfo.Version = version
	server.BuildInfo.Commit = commit
	server.BuildInfo.BuiltAt = builtAt
	slog.SetDefault(observability.NewLogger(cfg.LogLevel))
	slog.Info("Starting MCP-RAG server", "port", cfg.HTTPPort)

	// Observability
	metricsCollector := observability.NewMetricsCollector(observability.DefaultMetricsConfig())

	// Langfuse tracing (via eino callbacks)
	var langfuseFlusher func()
	if lfHost := os.Getenv("LANGFUSE_BASE_URL"); lfHost != "" {
		lfHandler, flusher := lfcallbacks.NewLangfuseHandler(&lfcallbacks.Config{
			Host:      lfHost,
			PublicKey: os.Getenv("LANGFUSE_PUBLIC_KEY"),
			SecretKey: os.Getenv("LANGFUSE_SECRET_KEY"),
			Name:      "mcp-rag",
		})
		callbacks.AppendGlobalHandlers(lfHandler)
		langfuseFlusher = flusher
		slog.Info("Langfuse tracing enabled", "host", lfHost)
	}

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
	slog.Info("Default KB", "name", defaultKB.Name, "index", defaultKB.IndexName())

	// eino-ext embedder (wrapped in SwappableEmbedder for future hot-swap)
	rawEmbedder, err := openaiembed.NewEmbedder(ctx, &openaiembed.EmbeddingConfig{
		BaseURL: cfg.EmbeddingBaseURL,
		APIKey:  cfg.EmbeddingAPIKey,
		Model:   cfg.EmbeddingModel,
	})
	if err != nil {
		return fmt.Errorf("create embedder: %w", err)
	}

	// Probe dims
	vecs, err := rawEmbedder.EmbedStrings(ctx, []string{"probe"})
	if err != nil {
		return fmt.Errorf("probe embedding: %w", err)
	}
	dims := len(vecs[0])
	slog.Info("Embedding dims", "dims", dims)

	// Wrap in swappable proxy for future hot-swap support
	embedder := swappable.NewSwappableEmbedder(rawEmbedder)

	// Record embedding info on KB service so newly-created KBs are bound to this model
	kbService.SetEmbeddingInfo(cfg.EmbeddingModel, dims)

	// Splitter (used as a transformer template by per-request compile)
	splitter, err := newSplitter(ctx, cfg.ChunkSize, cfg.ChunkOverlap)
	if err != nil {
		return fmt.Errorf("create splitter: %w", err)
	}

	// KB indexer config (used by per-request BuildIndexChain)
	indexerConf := &elastic_indexer.IndexerConfig{
		Client:           esClient,
		IndexSpec:        indexSpecForDims(dims),
		DocumentToFields: rag.ProjectDocumentToFields(),
		Embedding:        embedder,
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

	// Setup HTTP
	gin.SetMode(gin.ReleaseMode)
	srv, err := server.New(cfg, configManager, metricsCollector, retrievalCache,
		embedder, splitter, llm, indexerConf, esClient, kbService, dims, cfg.EmbeddingModel)
	if err != nil {
		return fmt.Errorf("create server: %w", err)
	}
	router := srv.Setup()

	httpServer := &http.Server{Addr: fmt.Sprintf(":%d", cfg.HTTPPort), Handler: router}
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		slog.Info("Shutting down...")
		if langfuseFlusher != nil {
			slog.Info("Flushing Langfuse events...")
			langfuseFlusher()
		}
		httpServer.Shutdown(context.Background())
	}()

	slog.Info("MCP-RAG listening", "port", cfg.HTTPPort)
	return httpServer.ListenAndServe()
}

// newSplitter creates a recursive text splitter transformer configured
// with the project's chunk size and overlap. Per-request compile in
// BuildIndexChainAt reuses the same splitter for every request.
func newSplitter(ctx context.Context, chunkSize, overlap int) (document.Transformer, error) {
	return recursive.NewSplitter(ctx, &recursive.Config{
		ChunkSize:   chunkSize,
		OverlapSize: overlap,
	})
}
