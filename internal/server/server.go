// Package server provides the HTTP API for MCP-RAG.
package server

import (
	"context"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"

	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino/compose"
	"github.com/gin-gonic/gin"
	mcpserver "github.com/mark3labs/mcp-go/server"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/knowledgebase"
	"github.com/jiaobendaye/mcp-rag-go/internal/observability"
	"github.com/jiaobendaye/mcp-rag-go/internal/rag"
	"github.com/jiaobendaye/mcp-rag-go/internal/security"
)

// Server holds all dependencies for HTTP handlers.
type Server struct {
	cfg              *config.Config
	configManager    *config.ConfigManager
	metricsCollector *observability.MetricsCollector
	retrievalCache   *rag.RetrievalCache
	chain            compose.Runnable[document.Source, []string]
	chatSvc          *rag.ChatService
	searcher         rag.Searcher
	embedder         rag.Embedder
	kbs              *knowledgebase.Service
	esIndexer        *rag.ES8Indexer
	mcpSrv           *mcpserver.MCPServer
	mcpHandler       *mcpserver.StreamableHTTPServer
}

// New creates a new Server with all dependencies.
func New(
	cfg *config.Config,
	configManager *config.ConfigManager,
	metricsCollector *observability.MetricsCollector,
	retrievalCache *rag.RetrievalCache,
	chain compose.Runnable[document.Source, []string],
	chatSvc *rag.ChatService,
	searcher rag.Searcher,
	embedder rag.Embedder,
	kbs *knowledgebase.Service,
	esIndexer *rag.ES8Indexer,
) *Server {
	return &Server{
		cfg:              cfg,
		configManager:    configManager,
		metricsCollector: metricsCollector,
		retrievalCache:   retrievalCache,
		chain:            chain,
		chatSvc:          chatSvc,
		searcher:         searcher,
		embedder:         embedder,
		kbs:              kbs,
		esIndexer:        esIndexer,
	}
}

// Setup registers all routes on the Gin engine.
func (s *Server) Setup() *gin.Engine {
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())

	// Request tracing middleware (must be before SecurityMiddleware to set headers early)
	r.Use(TracingMiddleware())

	// Security middleware (auth + rate-limit, no-op when disabled)
	r.Use(SecurityMiddleware(s.cfg))

	// Root and legacy redirects
	r.GET("/", func(c *gin.Context) { c.Redirect(http.StatusFound, "/app") })
	r.GET("/doc", func(c *gin.Context) { c.Redirect(http.StatusFound, "/docs") })
	r.GET("/documents-page", func(c *gin.Context) { c.Redirect(http.StatusFound, "/app/documents") })
	r.GET("/config-page", func(c *gin.Context) { c.Redirect(http.StatusFound, "/app/config") })

	// API docs
	r.GET("/docs", s.serveDocs)
	r.GET("/openapi.json", s.serveOpenAPI)

	// SPA static file serving (after /docs so /docs is not captured by SPA)
	s.initSPA(r)

	// System
	r.GET("/health", s.health)
	r.GET("/metrics", s.metrics)
	r.GET("/ready", s.ready)

	// Config management (always registered for SPA compatibility)
	r.GET("/config", s.configGet)
	r.POST("/config", s.configSet)
	r.POST("/config/bulk", s.configSetBulk)
	r.POST("/config/reset", s.configReset)
	r.POST("/config/reload", s.configReload)

	// Model discovery
	r.GET("/providers/:provider/models", s.providerModels)

	// Document
	r.POST("/add-document", s.addDocument)
	r.POST("/upload-files", s.uploadFiles)
	r.GET("/list-documents", s.listDocuments)
	r.DELETE("/delete-document", s.deleteDocument)
	r.GET("/list-files", s.listFiles)
	r.DELETE("/delete-file", s.deleteFile)

	// Knowledge Bases
	r.GET("/knowledge-bases", s.listKnowledgeBases)
	r.POST("/knowledge-bases", s.createKnowledgeBase)
	r.GET("/collections", s.listCollections)

	// Search & Chat
	r.GET("/search", s.search)
	r.POST("/chat", s.chat)

	// MCP (Streamable HTTP)
	s.mcpSrv, s.mcpHandler = s.InitMCP()
	r.Any("/mcp", gin.WrapH(s.mcpHandler))
	r.Any("/mcp/*path", gin.WrapH(s.mcpHandler))

	// Debug endpoints
	r.GET("/debug/mcp/tools", s.mcpListTools)
	r.POST("/debug/mcp/call", s.mcpDebugCall)

	return r
}

// resolveKB resolves kb_id or legacy collection to a concrete ES index.
func (s *Server) resolveKB(c *gin.Context) (*knowledgebase.Resolution, string, error) {
	kbID := parseIntPtr(c.Query("kb_id"))
	scope := strPtr(c.Query("scope"))
	collection := c.Query("collection")
	if collection == "" {
		collection = c.DefaultQuery("collection", "default")
	}
	userID := parseIntPtr(c.Query("user_id"))
	agentID := parseIntPtr(c.Query("agent_id"))

	// Fallback to legacy ES index when KB service is not available
	if s.kbs == nil {
		return nil, s.cfg.ESIndex, nil
	}

	legacyKey := ""
	if kbID == nil && scope == nil {
		legacyKey = "legacy:public:" + collection
	}

	resolution, err := s.kbs.Resolve(knowledgebase.ResolveRequest{
		KBID: kbID, Scope: scope, UserID: userID, AgentID: agentID,
		LegacyCollection: &collection, LegacyCollectionKey: &legacyKey,
	})
	if err != nil {
		return nil, "", err
	}
	return resolution, resolution.KnowledgeBase.IndexName(), nil
}

func parseIntPtr(s string) *int64 {
	if s == "" {
		return nil
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return nil
	}
	return &v
}

func strPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// health responds with structured service status (compatible with Python format).
func (s *Server) health(c *gin.Context) {
	// Build runtime info (mirrors Python format)
	runtime := gin.H{
		"embedding_model": gin.H{
			"provider": s.cfg.EmbeddingProvider,
			"model":    s.cfg.EmbeddingModel,
		},
		"llm_model": gin.H{
			"provider": s.cfg.LLMProvider,
			"model":    s.cfg.LLMModel,
		},
		"vector_store": gin.H{
			"type": "elasticsearch",
		},
		"knowledge_base": gin.H{
			"type": "sqlite",
		},
	}

	revision := int64(0)
	if s.configManager != nil {
		revision = s.configManager.Revision()
	}

	// Determine readiness
	bootstrapped := s.embedder != nil && s.esIndexer != nil && s.kbs != nil
	ready := bootstrapped
	if ready && s.esIndexer != nil {
		if err := s.esIndexer.HealthCheck(context.Background()); err != nil {
			ready = false
		}
	}
	healthy := ready

	reasons := []string{}
	if !bootstrapped {
		reasons = append(reasons, "service not fully bootstrapped")
	}
	if bootstrapped && !ready {
		reasons = append(reasons, "elasticsearch not reachable")
	}

	resp := gin.H{
		"status":          "healthy",
		"healthy":         healthy,
		"ready":           ready,
		"bootstrapped":    bootstrapped,
		"runtime":         runtime,
		"config_revision": revision,
		"reasons":         reasons,
	}

	if s.metricsCollector != nil {
		hd := s.metricsCollector.HealthDetail()
		resp["uptime_seconds"] = hd.UptimeSeconds
		resp["total_requests"] = hd.TotalRequests
		resp["error_count"] = hd.ErrorCount
		resp["operations"] = hd.Operations
	}

	c.JSON(http.StatusOK, resp)
}

// metrics returns the full metrics snapshot.
func (s *Server) metrics(c *gin.Context) {
	if s.metricsCollector == nil {
		c.JSON(http.StatusOK, gin.H{"error": "metrics collector not configured"})
		return
	}
	snap := s.metricsCollector.Snapshot()
	c.JSON(http.StatusOK, snap)
}

// ready checks component-level readiness (compatible with Python format).
func (s *Server) ready(c *gin.Context) {
	// Build runtime info
	runtime := gin.H{
		"embedding_model": gin.H{
			"provider": s.cfg.EmbeddingProvider,
			"model":    s.cfg.EmbeddingModel,
		},
		"llm_model": gin.H{
			"provider": s.cfg.LLMProvider,
			"model":    s.cfg.LLMModel,
		},
		"vector_store": gin.H{
			"type": "elasticsearch",
		},
		"knowledge_base": gin.H{
			"type": "sqlite",
		},
	}

	revision := int64(0)
	if s.configManager != nil {
		revision = s.configManager.Revision()
	}

	bootstrapped := s.embedder != nil && s.esIndexer != nil && s.kbs != nil
	ready := bootstrapped
	if ready && s.esIndexer != nil {
		if err := s.esIndexer.HealthCheck(context.Background()); err != nil {
			ready = false
		}
	}

	resp := gin.H{
		"bootstrapped":    bootstrapped,
		"ready":           ready,
		"runtime":         runtime,
		"config_revision": revision,
	}

	status := http.StatusOK
	if !ready {
		status = http.StatusServiceUnavailable
	}
	c.JSON(status, resp)
}

// configGet returns all configuration values in flat format (Python-compatible).
func (s *Server) configGet(c *gin.Context) {
	result := s.buildConfigGetResponse()
	c.JSON(http.StatusOK, result)
}

// buildConfigGetResponse builds the full config response, working with or without configManager.
func (s *Server) buildConfigGetResponse() gin.H {
	revision := int64(0)
	configPath := ""

	if s.configManager != nil {
		cfg := s.configManager.GetAll()
		rawCfg := s.configManager.Get()
		revision = s.configManager.Revision()
		configPath = s.configManager.ConfigPath()

		result := make(gin.H, len(cfg)+20)
		result["config_revision"] = revision
		result["config_path"] = configPath

		for k, v := range cfg {
			result[k] = v
		}

		result["provider_configs"] = s.buildProviderConfigs()

		result["cache"] = gin.H{
			"enabled":     false,
			"max_entries": 256,
			"ttl_seconds": 300,
		}

		result["security"] = gin.H{
			"enabled":          rawCfg.SecurityEnabled,
			"allow_anonymous":  rawCfg.SecurityAllowAnonymous,
			"api_keys":         rawCfg.SecurityAPIKeys,
			"tenant_api_keys":  rawCfg.SecurityTenantAPIKeys,
		}

		result["rate_limit"] = gin.H{
			"requests_per_window": rawCfg.RateLimitRequestsPerWindow,
			"window_seconds":      rawCfg.RateLimitWindowSeconds,
			"burst":               rawCfg.RateLimitBurst,
		}

		result["quotas"] = gin.H{
			"max_upload_files":      rawCfg.QuotaMaxUploadFiles,
			"max_upload_bytes":      rawCfg.QuotaMaxUploadBytes,
			"max_upload_file_bytes": rawCfg.QuotaMaxUploadFileBytes,
			"max_index_documents":   rawCfg.QuotaMaxIndexDocuments,
			"max_index_chunks":      rawCfg.QuotaMaxIndexChunks,
			"max_index_chars":       rawCfg.QuotaMaxIndexChars,
		}

		result["host"] = "0.0.0.0"
		result["port"] = result["http_port"]
		result["debug"] = false
		result["vector_db_type"] = "elasticsearch"
		result["max_retrieval_results"] = result["top_k"]
		result["similarity_threshold"] = result["min_score"]

		return result
	}

	// Fallback: use startup config directly
	result := gin.H{
		"http_port":             s.cfg.HTTPPort,
		"host":                  "0.0.0.0",
		"port":                  s.cfg.HTTPPort,
		"debug":                 false,
		"vector_db_type":        "elasticsearch",
		"es_url":                s.cfg.ESUrl,
		"es_index":              s.cfg.ESIndex,
		"knowledge_base_db_path": s.cfg.KnowledgeBaseDBPath,
		"embedding_provider":     s.cfg.EmbeddingProvider,
		"embedding_model":        s.cfg.EmbeddingModel,
		"embedding_base_url":     s.cfg.EmbeddingBaseURL,
		"llm_provider":           s.cfg.LLMProvider,
		"llm_model":              s.cfg.LLMModel,
		"llm_base_url":           s.cfg.LLMBaseURL,
		"chunk_size":             s.cfg.ChunkSize,
		"chunk_overlap":          s.cfg.ChunkOverlap,
		"top_k":                  s.cfg.TopK,
		"min_score":              s.cfg.MinScore,
		"search_mode":            s.cfg.SearchMode,
		"max_retrieval_results":  s.cfg.TopK,
		"similarity_threshold":   s.cfg.MinScore,
		"config_revision":        revision,
		"config_path":            configPath,
		"cache": gin.H{
			"enabled": false, "max_entries": 256, "ttl_seconds": 300,
		},
		"security": gin.H{
			"enabled": s.cfg.SecurityEnabled, "allow_anonymous": s.cfg.SecurityAllowAnonymous,
			"api_keys": s.cfg.SecurityAPIKeys, "tenant_api_keys": s.cfg.SecurityTenantAPIKeys,
		},
		"rate_limit": gin.H{
			"requests_per_window": s.cfg.RateLimitRequestsPerWindow,
			"window_seconds":      s.cfg.RateLimitWindowSeconds,
			"burst":               s.cfg.RateLimitBurst,
		},
		"quotas": gin.H{
			"max_upload_files": s.cfg.QuotaMaxUploadFiles, "max_upload_bytes": s.cfg.QuotaMaxUploadBytes,
			"max_upload_file_bytes": s.cfg.QuotaMaxUploadFileBytes,
			"max_index_documents":   s.cfg.QuotaMaxIndexDocuments,
			"max_index_chunks":      s.cfg.QuotaMaxIndexChunks,
			"max_index_chars":       s.cfg.QuotaMaxIndexChars,
		},
	}

	// provider_configs from startup config
	result["provider_configs"] = s.buildProviderConfigsFromStartup()

	return result
}

// buildProviderConfigs constructs the provider_configs object from configManager.
func (s *Server) buildProviderConfigs() gin.H {
	cfg := s.configManager.Get()
	return s.buildProviderConfigsFromCfg(cfg)
}

// buildProviderConfigsFromStartup constructs provider_configs from startup config (fallback).
func (s *Server) buildProviderConfigsFromStartup() gin.H {
	return s.buildProviderConfigsFromCfg(s.cfg)
}

// buildProviderConfigsFromCfg builds provider_configs from a *Config value.
func (s *Server) buildProviderConfigsFromCfg(cfg *config.Config) gin.H {

	providerConfigs := gin.H{}

	// Main embedding provider
	embeddingCfg := gin.H{
		"base_url":          cfg.EmbeddingBaseURL,
		"api_key":           maskAPIKey(cfg.EmbeddingAPIKey),
		"model":             cfg.EmbeddingModel,
		"llm_model":         nil,
		"embedding_model":   cfg.EmbeddingModel,
		"chat_models":       []string{},
		"embedding_models":  []string{cfg.EmbeddingModel},
	}
	providerConfigs[cfg.EmbeddingProvider] = embeddingCfg

	// Main LLM provider (may be same as embedding provider)
	if cfg.LLMProvider != cfg.EmbeddingProvider {
		providerConfigs[cfg.LLMProvider] = gin.H{
			"base_url":         cfg.LLMBaseURL,
			"api_key":          maskAPIKey(cfg.LLMAPIKey),
			"model":            cfg.LLMModel,
			"llm_model":        cfg.LLMModel,
			"embedding_model":  nil,
			"chat_models":      []string{cfg.LLMModel},
			"embedding_models": []string{},
		}
	} else {
		// Merge LLM model into existing provider
		embeddingCfg["llm_model"] = cfg.LLMModel
		embeddingCfg["model"] = cfg.LLMModel
		embeddingCfg["chat_models"] = []string{cfg.LLMModel}
		providerConfigs[cfg.EmbeddingProvider] = embeddingCfg
	}

	return providerConfigs
}

// maskAPIKey masks an API key for display: show first 4 and last 4 chars.
func maskAPIKey(key string) string {
	if key == "" {
		return ""
	}
	if len(key) <= 8 {
		return "****"
	}
	return key[:4] + "****" + key[len(key)-4:]
}

// configSet updates a single configuration value.
func (s *Server) configSet(c *gin.Context) {
	if s.configManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"detail": "config manager not available"})
		return
	}
	var req struct {
		Key   string      `json:"key"`
		Value interface{} `json:"value"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid request body"})
		return
	}
	if req.Key == "" {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "key is required"})
		return
	}

	if err := s.configManager.Set(req.Key, req.Value); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"key": req.Key, "value": req.Value, "status": "updated"})
}

// configSetBulk updates multiple configuration values at once.
// Supports both flat {key: value} and SPA's {"updates": {key: value}} format.
func (s *Server) configSetBulk(c *gin.Context) {
	if s.configManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"detail": "config manager not available"})
		return
	}
	var bulk map[string]interface{}
	if err := c.ShouldBindJSON(&bulk); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid request body"})
		return
	}

	// Unwrap SPA format: {"updates": {...}}
	if updates, ok := bulk["updates"].(map[string]interface{}); ok {
		bulk = updates
	}

	for key, value := range bulk {
		if err := s.configManager.Set(key, value); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error(), "key": key})
			return
		}
	}

	c.JSON(http.StatusOK, gin.H{"status": "updated", "count": len(bulk)})
}

// configReset resets configuration to defaults.
func (s *Server) configReset(c *gin.Context) {
	if s.configManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"detail": "config manager not available"})
		return
	}
	s.configManager.Reset()
	c.JSON(http.StatusOK, gin.H{"status": "reset"})
}

// configReload reloads configuration from disk.
func (s *Server) configReload(c *gin.Context) {
	if s.configManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"detail": "config manager not available"})
		return
	}
	if err := s.configManager.Reload(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"status":   "reloaded",
		"revision": s.configManager.Revision(),
	})
}

// listDocuments returns paginated documents.
func (s *Server) listDocuments(c *gin.Context) {
	_, indexName, err := s.resolveKB(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "100"))
	offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))
	result, err := s.esIndexer.ListDocuments(indexName, limit, offset)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	c.JSON(http.StatusOK, result)
}

// deleteDocument removes a document by ID.
func (s *Server) deleteDocument(c *gin.Context) {
	_, indexName, err := s.resolveKB(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}
	var req struct {
		DocumentID string `json:"document_id"`
	}
	if err := c.ShouldBindJSON(&req); err != nil || req.DocumentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "document_id is required"})
		return
	}
	if err := s.esIndexer.DeleteDocument(indexName, req.DocumentID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	// Invalidate cache for this collection
	if s.retrievalCache != nil {
		s.retrievalCache.InvalidateScope(indexName)
	}
	c.JSON(http.StatusOK, gin.H{"message": "Document deleted successfully"})
}

// listFiles returns aggregated file information.
func (s *Server) listFiles(c *gin.Context) {
	_, indexName, err := s.resolveKB(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}
	files, err := s.esIndexer.ListFiles(indexName)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"files": files})
}

// deleteFile removes all chunks for a filename.
func (s *Server) deleteFile(c *gin.Context) {
	_, indexName, err := s.resolveKB(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}
	var req struct {
		Filename string `json:"filename"`
	}
	if err := c.ShouldBindJSON(&req); err != nil || req.Filename == "" {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "filename is required"})
		return
	}
	if err := s.esIndexer.DeleteFile(indexName, req.Filename); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	// Invalidate cache for this collection
	if s.retrievalCache != nil {
		s.retrievalCache.InvalidateScope(indexName)
	}
	c.JSON(http.StatusOK, gin.H{"message": "File deleted successfully"})
}

// listKnowledgeBases returns accessible knowledge bases.
func (s *Server) listKnowledgeBases(c *gin.Context) {
	if s.kbs == nil {
		c.JSON(http.StatusOK, gin.H{"knowledge_bases": []gin.H{}})
		return
	}
	userID := parseIntPtr(c.Query("user_id"))
	kbs, err := s.kbs.ListAccessible(userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"knowledge_bases": kbs})
}

// createKnowledgeBase creates a new knowledge base.
func (s *Server) createKnowledgeBase(c *gin.Context) {
	if s.kbs == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"detail": "knowledge base service not available"})
		return
	}
	var req struct {
		Name         string `json:"name"`
		Scope        string `json:"scope"`
		OwnerUserID  *int64 `json:"owner_user_id"`
		OwnerAgentID *int64 `json:"owner_agent_id"`
	}
	if err := c.ShouldBindJSON(&req); err != nil || req.Name == "" {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "name is required"})
		return
	}
	if req.Scope == "" {
		req.Scope = "public"
	}
	kb, err := s.kbs.Resolve(knowledgebase.ResolveRequest{
		Scope: &req.Scope, UserID: req.OwnerUserID, AgentID: req.OwnerAgentID,
		LegacyCollection: &req.Name,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	// Ensure ES index exists
	dims := 1024 // default, will be probed later
	s.kbs.EnsurePublicDefault() // ensure at least public default
	_ = dims
	c.JSON(http.StatusOK, kb.KnowledgeBase)
}

// listCollections returns legacy collection names (backward compat).
func (s *Server) listCollections(c *gin.Context) {
	userID := parseIntPtr(c.Query("user_id"))
	kbs, err := s.kbs.ListAccessible(userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	names := make([]string, len(kbs))
	for i, kb := range kbs {
		names[i] = kb.CollectionName
	}
	c.JSON(http.StatusOK, gin.H{"collections": names})
}

func (s *Server) addDocument(c *gin.Context) {
	var req struct {
		Content string `json:"content"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid request body"})
		return
	}
	if req.Content == "" {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "content is required"})
		return
	}

	// Index quota check
	iq := security.NewIndexQuotaPolicy(s.cfg.QuotaMaxIndexDocuments, s.cfg.QuotaMaxIndexChunks, s.cfg.QuotaMaxIndexChars)
	chars := len([]rune(req.Content))
	if d := iq.Check(1, 1+chars/s.cfg.ChunkSize, chars); !d.Allowed {
		c.JSON(http.StatusRequestEntityTooLarge, gin.H{"detail": d.Reason})
		return
	}
	// Write content to temp file for Chain
	if s.chain == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "index chain not configured"})
		return
	}
	tmpFile, err := os.CreateTemp("", "mcp-rag-doc-*.txt")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)
	if _, err := tmpFile.WriteString(req.Content); err != nil {
		tmpFile.Close()
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	tmpFile.Close()

	ids, err := s.chain.Invoke(c.Request.Context(), document.Source{URI: tmpPath})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	docID := ""
	if len(ids) > 0 {
		docID = ids[0] // Use first chunk ID as document reference
	}

	// Invalidate cache for the default index
	if s.retrievalCache != nil {
		s.retrievalCache.InvalidateScope(s.cfg.ESIndex)
	}

	c.JSON(http.StatusOK, gin.H{
		"message":     "Document added successfully",
		"document_id": docID,
		"chunk_count": len(ids),
	})
}

// uploadFiles handles multipart file upload and indexing.
func (s *Server) uploadFiles(c *gin.Context) {
	form, err := c.MultipartForm()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid multipart form"})
		return
	}

	files := form.File["files"]
	if len(files) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "No files provided"})
		return
	}

	// Upload quota check
	uploadQuota := security.NewUploadQuotaPolicy(
		s.cfg.QuotaMaxUploadFiles,
		s.cfg.QuotaMaxUploadBytes,
		s.cfg.QuotaMaxUploadFileBytes,
	)
	fileSizes := make([]int64, len(files))
	for i, fh := range files {
		fileSizes[i] = fh.Size
	}
	if decision := uploadQuota.Check(fileSizes); !decision.Allowed {
		c.JSON(http.StatusRequestEntityTooLarge, gin.H{"detail": decision.Reason})
		return
	}

	type fileResult struct {
		Filename      string `json:"filename"`
		FileType      string `json:"file_type"`
		ContentLength int64  `json:"content_length"`
		Processed     bool   `json:"processed"`
		Error         string `json:"error,omitempty"`
	}

	var results []fileResult
	successful := 0

	for _, fh := range files {
		// Save to temp file
		tmpPath := fh.Filename
		if err := s.saveTemp(fh); err != nil {
			results = append(results, fileResult{
				Filename: fh.Filename,
				Error:    err.Error(),
			})
			continue
		}

		// Index via Chain
		_, err = s.chain.Invoke(c.Request.Context(), document.Source{URI: tmpPath})
		os.Remove(tmpPath)
		if err != nil {
			results = append(results, fileResult{Filename: fh.Filename, ContentLength: fh.Size, Error: err.Error()})
			continue
		}
		successful++
		results = append(results, fileResult{Filename: fh.Filename, ContentLength: fh.Size, Processed: true})
	}

	// Invalidate cache for the default index
	if s.retrievalCache != nil {
		s.retrievalCache.InvalidateScope(s.cfg.ESIndex)
	}

	c.JSON(http.StatusOK, gin.H{
		"total_files": len(files),
		"successful":  successful,
		"failed":      len(files) - successful,
		"results":     results,
	})
}

// search performs vector similarity search.
func (s *Server) search(c *gin.Context) {
	query := c.Query("query")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "Query is required"})
		return
	}

	limit := 5
	if l := c.Query("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 {
			limit = parsed
		}
	}

	// Resolve KB
	resolution, indexName, err := s.resolveKB(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}

	// Check cache first
	if s.retrievalCache != nil {
		cacheKey := rag.RetrievalCacheKey{
			Collection: indexName,
			Query:      query,
			Mode:       s.cfg.SearchMode,
			Limit:      limit,
			Threshold:  s.cfg.MinScore,
		}
		if cached, ok := s.retrievalCache.Get(cacheKey); ok && cached != nil {
			c.JSON(http.StatusOK, cached)
			return
		}
	}

	// Embed the query
	vecs, err := s.embedder.EmbedStrings(c.Request.Context(), []string{query})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	// Search with configured mode (hybrid/rrf/knn)
	hits, err := s.searcher.SearchWithMode(c.Request.Context(), query, toFloat32(vecs[0]), limit, s.cfg.MinScore, s.cfg.SearchMode)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	// Format response compatible with Python MCP-RAG
	results := make([]gin.H, 0, len(hits))
	for _, h := range hits {
		result := gin.H{
			"content":  h.Content,
			"score":    h.Score,
			"source":   h.Source,
			"filename": h.Filename,
		}

		// Enriched metadata with KB info
		metadata := gin.H{}
		if resolution != nil {
			metadata["knowledge_base_id"] = resolution.KnowledgeBase.ID
			metadata["knowledge_base_name"] = resolution.KnowledgeBase.Name
			metadata["knowledge_base_scope"] = resolution.KnowledgeBase.Scope
		}
		result["metadata"] = metadata

		// Vector/keyword score and retrieval method
		switch s.cfg.SearchMode {
		case "hybrid", "rrf":
			result["vector_score"] = h.Score
			result["keyword_score"] = h.Score
			result["retrieval_method"] = "hybrid"
		case "knn":
			result["vector_score"] = h.Score
			result["keyword_score"] = 0.0
			result["retrieval_method"] = "vector"
		default:
			result["vector_score"] = h.Score
			result["retrieval_method"] = s.cfg.SearchMode
		}

		results = append(results, result)
	}

	searchResp := gin.H{
		"query":      query,
		"collection": indexName,
		"results":    results,
	}

	// Store in cache
	if s.retrievalCache != nil {
		cacheKey := rag.RetrievalCacheKey{
			Collection: indexName,
			Query:      query,
			Mode:       s.cfg.SearchMode,
			Limit:      limit,
			Threshold:  s.cfg.MinScore,
		}
		searchHits := make([]rag.SearchHit, len(hits))
		copy(searchHits, hits)
		s.retrievalCache.Set(cacheKey, &rag.SearchResponse{
			Query:      query,
			Collection: indexName,
			Results:    searchHits,
		})
	}

	c.JSON(http.StatusOK, searchResp)
}

// chat handles RAG-based conversation.
func (s *Server) chat(c *gin.Context) {
	var req rag.ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid request body"})
		return
	}
	if req.Query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "query is required"})
		return
	}

	// Resolve KB from request fields
	if req.KBID != nil || req.Scope != "" || req.Collection != "" {
		resolution, indexName, err := s.resolveKBFromChat(&req)
		if err == nil {
			req.Collection = indexName
			_ = resolution
		}
	}

	resp, err := s.chatSvc.Chat(c.Request.Context(), &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	c.JSON(http.StatusOK, resp)
}

// resolveKBFromChat resolves KB from ChatRequest fields.
func (s *Server) resolveKBFromChat(req *rag.ChatRequest) (*knowledgebase.Resolution, string, error) {
	if s.kbs == nil {
		return nil, s.cfg.ESIndex, nil
	}

	scope := req.Scope
	var scopePtr *string
	if scope != "" {
		scopePtr = &scope
	}

	collection := req.Collection
	if collection == "" {
		collection = "default"
	}

	legacyKey := ""
	if req.KBID == nil && scopePtr == nil {
		legacyKey = "legacy:public:" + collection
	}

	resolution, err := s.kbs.Resolve(knowledgebase.ResolveRequest{
		KBID:               req.KBID,
		Scope:              scopePtr,
		UserID:             req.UserID,
		AgentID:            req.AgentID,
		LegacyCollection:   &collection,
		LegacyCollectionKey: &legacyKey,
	})
	if err != nil {
		return nil, "", err
	}
	return resolution, resolution.KnowledgeBase.IndexName(), nil
}

// saveTemp saves an uploaded file to a temporary location.
func (s *Server) saveTemp(fh *multipart.FileHeader) error {
	src, err := fh.Open()
	if err != nil {
		return err
	}
	defer src.Close()

	dst, err := os.Create(fh.Filename)
	if err != nil {
		return err
	}
	defer dst.Close()

	_, err = io.Copy(dst, src)
	return err
}

func toFloat32(f64 []float64) []float32 {
	result := make([]float32, len(f64))
	for i, v := range f64 {
		result[i] = float32(v)
	}
	return result
}
