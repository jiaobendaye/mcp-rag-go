// Package server provides the HTTP API for MCP-RAG.
package server

import (
	"context"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/cloudwego/eino/components/document"
	"github.com/elastic/go-elasticsearch/v8"
	"github.com/gin-gonic/gin"
	mcpserver "github.com/mark3labs/mcp-go/server"

	elastic_indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
	elastic_retriever "github.com/cloudwego/eino-ext/components/retriever/es8"
	elastic_search_mode "github.com/cloudwego/eino-ext/components/retriever/es8/search_mode"

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

	// Per-request compile components
	splitter    document.Transformer
	embedder    rag.Embedder
	llm         rag.LLMGenerator
	indexerConf *elastic_indexer.IndexerConfig

	// Direct ES client for admin endpoints and per-request retriever creation
	esClient *elasticsearch.Client

	// Knowledge base service
	kbs *knowledgebase.Service

	embedDims       int
	embeddingModel  string
	classifier *rag.QueryClassifier

	mcpSrv     *mcpserver.MCPServer
	mcpHandler *mcpserver.StreamableHTTPServer
}

// New creates a new Server with all dependencies.
func New(
	cfg *config.Config,
	configManager *config.ConfigManager,
	metricsCollector *observability.MetricsCollector,
	retrievalCache *rag.RetrievalCache,
	embedder rag.Embedder,
	splitter document.Transformer,
	llm rag.LLMGenerator,
	indexerConf *elastic_indexer.IndexerConfig,
	esClient *elasticsearch.Client,
	kbs *knowledgebase.Service,
	embedDims int,
	embeddingModel string,
) (*Server, error) {
	return &Server{
		cfg:              cfg,
		configManager:    configManager,
		metricsCollector: metricsCollector,
		retrievalCache:   retrievalCache,
		embedder:         embedder,
		splitter:         splitter,
		llm:              llm,
		indexerConf:      indexerConf,
		esClient:         esClient,
		kbs:              kbs,
		embedDims:        embedDims,
		classifier:       rag.NewQueryClassifier(),
	}, nil
}

// Setup registers all routes on the Gin engine.
func (s *Server) Setup() *gin.Engine {
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())

	// Request tracing middleware (must before SecurityMiddleware to set headers early)
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

// resolveKB resolves kb_id to a concrete ES index.
// bodyKBID is an optional explicit kb_id supplied by the caller (e.g. from
// a JSON request body); if non-nil it takes precedence over the URL query
// `kb_id` parameter. Pass nil to read from the URL query only.
func (s *Server) resolveKB(c *gin.Context, bodyKBID *int64) (*knowledgebase.Resolution, string, error) {
	if s.kbs == nil {
		return nil, "", fmt.Errorf("knowledge base service not configured")
	}

	kbID := bodyKBID
	if kbID == nil {
		kbID = parseIntPtr(c.Query("kb_id"))
	}
	collection := strPtr(c.Query("collection"))
	scope := strPtr(c.Query("scope"))
	userID := parseIntPtr(c.Query("user_id"))
	agentID := parseIntPtr(c.Query("agent_id"))

	resolution, err := s.kbs.Resolve(knowledgebase.ResolveRequest{
		KBID: kbID, Collection: collection, Scope: scope, UserID: userID, AgentID: agentID,
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

// parseKBIDs parses kb_ids from various formats (comma-separated string, JSON array, int).
func parseKBIDs(raw any) []int64 {
	if raw == nil {
		return nil
	}
	switch v := raw.(type) {
	case string:
		var ids []int64
		for _, part := range strings.Split(v, ",") {
			trimmed := strings.TrimSpace(part)
			if id, err := strconv.ParseInt(trimmed, 10, 64); err == nil {
				ids = append(ids, id)
			}
		}
		return ids
	case []any:
		var ids []int64
		for _, item := range v {
			switch n := item.(type) {
			case float64:
				ids = append(ids, int64(n))
			case int64:
				ids = append(ids, n)
			case int:
				ids = append(ids, int64(n))
			case string:
				if id, err := strconv.ParseInt(strings.TrimSpace(n), 10, 64); err == nil {
					ids = append(ids, id)
				}
			}
		}
		return ids
	default:
		return nil
	}
}

func strPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// health responds with structured service status (compatible with Python format).
func (s *Server) health(c *gin.Context) {
	runtime := s.buildRuntimeSnapshot()

	revision := int64(0)
	if s.configManager != nil {
		revision = s.configManager.Revision()
	}

	// Determine readiness
	bootstrapped := s.embedder != nil && s.indexerConf != nil && s.kbs != nil
	ready := bootstrapped
	if ready && s.esClient != nil {
		res, err := s.esClient.Ping()
		if err != nil {
			ready = false
		} else {
			res.Body.Close()
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

// buildRuntimeSnapshot builds the full 7-component runtime object.
func (s *Server) buildRuntimeSnapshot() gin.H {
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

	// document_processor
	if s.indexerConf != nil {
		runtime["document_processor"] = "ready"
	} else {
		runtime["document_processor"] = nil
	}

	// hybrid_service
	if s.esClient != nil {
		runtime["hybrid_service"] = "ready"
	} else {
		runtime["hybrid_service"] = nil
	}

	// retrieval_cache
	if s.retrievalCache != nil {
		runtime["retrieval_cache"] = s.retrievalCache.Stats()
	} else {
		runtime["retrieval_cache"] = nil
	}

	// provider_budget (not yet implemented)
	runtime["provider_budget"] = gin.H{
		"enabled": false,
		"reason":  "not yet implemented",
	}

	return runtime
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
	runtime := s.buildRuntimeSnapshot()

	revision := int64(0)
	if s.configManager != nil {
		revision = s.configManager.Revision()
	}

	bootstrapped := s.embedder != nil && s.indexerConf != nil && s.kbs != nil
	ready := bootstrapped
	reasons := []string{}
	if !bootstrapped {
		reasons = append(reasons, "service not fully bootstrapped")
	}
	if ready && s.esClient != nil {
		res, err := s.esClient.Ping()
		if err != nil {
			ready = false
			reasons = append(reasons, "elasticsearch not reachable")
		} else {
			res.Body.Close()
		}
	}

	resp := gin.H{
		"bootstrapped":    bootstrapped,
		"ready":           ready,
		"runtime":         runtime,
		"config_revision": revision,
	}
	if len(reasons) > 0 {
		resp["reasons"] = reasons
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

// listDocuments returns paginated documents. Admin operation: uses raw ES client.
func (s *Server) listDocuments(c *gin.Context) {
	_, indexName, err := s.resolveKB(c, nil)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "100"))
	offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))
	result, err := rag.AdminListDocuments(s.esClient, indexName, limit, offset)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	c.JSON(http.StatusOK, result)
}

// deleteDocument removes a document by ID. Admin operation: uses raw ES client.
func (s *Server) deleteDocument(c *gin.Context) {
	_, indexName, err := s.resolveKB(c, nil)
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
	if err := rag.AdminDeleteDocument(s.esClient, indexName, req.DocumentID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	if s.retrievalCache != nil {
		s.retrievalCache.InvalidateScope(indexName)
	}
	c.JSON(http.StatusOK, gin.H{"message": "Document deleted successfully"})
}

// listFiles returns aggregated file information. Admin operation.
func (s *Server) listFiles(c *gin.Context) {
	_, indexName, err := s.resolveKB(c, nil)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}
	files, err := rag.AdminListFiles(s.esClient, indexName)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"files": files})
}

// deleteFile removes all chunks for a filename. Admin operation.
func (s *Server) deleteFile(c *gin.Context) {
	_, indexName, err := s.resolveKB(c, nil)
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
	if err := rag.AdminDeleteFile(s.esClient, indexName, req.Filename); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
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
		if req.OwnerUserID != nil && req.OwnerAgentID != nil {
			req.Scope = "agent_private"
		} else {
			req.Scope = "public"
		}
	}

	// Create new KB
	kb, err := s.kbs.Create(req.Name, req.Scope, req.OwnerUserID, req.OwnerAgentID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	c.JSON(http.StatusOK, kb)
}

// listCollections returns collection names for all accessible KBs.
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

// addDocument writes a document to the resolved KB's index. The index
// chain is built per-request with the concrete ES index name.
func (s *Server) addDocument(c *gin.Context) {
	var req struct {
		Content string `json:"content"`
		KBID    *int64 `json:"kb_id,omitempty"`
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

	if s.indexerConf == nil || s.splitter == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "indexer not configured"})
		return
	}

	// Resolve KB. Body's req.KBID takes precedence over query ?kb_id=.
	// resolveKB handles both: pass req.KBID (may be nil if neither set).
	resolution, indexName, err := s.resolveKB(c, req.KBID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}
	if err := s.kbs.CheckEmbeddingMatch(resolution.KnowledgeBase); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}

	// Write content to temp file for the FileLoader
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

	// Per-request index chain: build with concrete indexName, no KBParams.
	chain, err := rag.BuildIndexChain(c.Request.Context(), s.splitter, s.indexerConf, indexName)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	ids, err := chain.Invoke(c.Request.Context(), document.Source{URI: tmpPath})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	docID := ""
	if len(ids) > 0 {
		docID = ids[0]
	}

	// Invalidate cache for the index we actually wrote to
	if s.retrievalCache != nil {
		s.retrievalCache.InvalidateScope(indexName)
	}

	c.JSON(http.StatusOK, gin.H{
		"message":     "Document added successfully",
		"document_id": docID,
		"chunk_count": len(ids),
	})
}

// uploadFiles handles multipart file upload and indexing. All files in
// the batch share the same KB, so the index chain is built once.
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

	// Resolve the target KB once for the entire batch
	resolution, indexName, err := s.resolveKB(c, nil)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}
	if err := s.kbs.CheckEmbeddingMatch(resolution.KnowledgeBase); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}

	if s.indexerConf == nil || s.splitter == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "indexer not configured"})
		return
	}

	// Build the index chain once for this KB (all files share the same
	// index; compile overhead ~12µs is negligible vs file I/O + ES writes).
	chain, err := rag.BuildIndexChain(c.Request.Context(), s.splitter, s.indexerConf, indexName)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
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

		_, err = chain.Invoke(c.Request.Context(), document.Source{URI: tmpPath})
		os.Remove(tmpPath)
		if err != nil {
			results = append(results, fileResult{Filename: fh.Filename, ContentLength: fh.Size, Error: err.Error()})
			continue
		}
		successful++
		results = append(results, fileResult{Filename: fh.Filename, ContentLength: fh.Size, Processed: true})
	}

	// Invalidate cache for the resolved index once
	if s.retrievalCache != nil {
		s.retrievalCache.InvalidateScope(indexName)
	}

	c.JSON(http.StatusOK, gin.H{
		"total_files": len(files),
		"successful":  successful,
		"failed":      len(files) - successful,
		"results":     results,
	})
}

// searchRawHit is a flat JSON representation of a single search result
// returned to HTTP and MCP clients. It is used by both /search and
// /chat responses, and is built from per-KB retriever outputs.
type searchRawHit struct {
	Content         string  `json:"content"`
	Score           float64 `json:"score"`
	Source          string  `json:"source"`
	Filename        string  `json:"filename"`
	ChunkID         string  `json:"chunk_id"`
	DocumentID      string  `json:"document_id"`
	ChunkIndex      int     `json:"chunk_index"`
	VectorScore     float64 `json:"vector_score,omitempty"`
	KeywordScore    float64 `json:"keyword_score,omitempty"`
	RetrievalMethod string  `json:"retrieval_method,omitempty"`
	Metadata        gin.H   `json:"metadata,omitempty"`
}

// docsToHits converts a slice of eino *schema.Document to the flat
// searchRawHit form. score falls back to whatever the result parser
// placed in MetaData["score"].
func docsToHits(docs []*rag.RetrievedDoc) []searchRawHit {
	out := make([]searchRawHit, 0, len(docs))
	for _, d := range docs {
		var score float64
		if v, ok := d.MetaData["score"]; ok {
			score, _ = v.(float64)
		}
		var filename, source, chunkID, documentID string
		var chunkIndex int
		if v, ok := d.MetaData["filename"].(string); ok {
			filename = v
		}
		if v, ok := d.MetaData["source"].(string); ok {
			source = v
		}
		if v, ok := d.MetaData["chunk_id"].(string); ok {
			chunkID = v
		}
		if v, ok := d.MetaData["document_id"].(string); ok {
			documentID = v
		}
		if v, ok := d.MetaData["chunk_index"]; ok {
			switch n := v.(type) {
			case int:
				chunkIndex = n
			case float64:
				chunkIndex = int(n)
			}
		}
		out = append(out, searchRawHit{
			Content:    d.Content,
			Score:      score,
			Source:     source,
			Filename:   filename,
			ChunkID:    chunkID,
			DocumentID: documentID,
			ChunkIndex: chunkIndex,
		})
	}
	return out
}

// search handles /search. Resolves KB, then runs a per-request
// BuildRetrievalGraphAt compile with closure-captured indexName. We
// use the same graph as chat but stop after the retrieve node to skip
// the LLM step (use the docs directly).
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

	// Multi-KB aggregation
	kbIDsRaw := c.Query("kb_ids")
	if kbIDsRaw == "" {
		kbIDsRaw = c.Query("kb_ids[]")
	}
	var kbIDs []int64
	if kbIDsRaw != "" {
		for _, part := range strings.Split(kbIDsRaw, ",") {
			trimmed := strings.TrimSpace(part)
			if id, err := strconv.ParseInt(trimmed, 10, 64); err == nil && id > 0 {
				kbIDs = append(kbIDs, id)
			}
		}
	}

	if len(kbIDs) > 0 {
		s.searchMultiKB(c, query, kbIDs, limit)
		return
	}

	// Single KB
	resolution, indexName, err := s.resolveKB(c, nil)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}
	if err := s.kbs.CheckEmbeddingMatch(resolution.KnowledgeBase); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}

	// Cache check
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

	// Per-request compile + retrieve (no LLM)
	docs, err := s.retrieveAt(c.Request.Context(), indexName, query, limit, s.cfg.MinScore, s.cfg.SearchMode)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	hits := docsToHits(docs)
	for i := range hits {
		method := s.cfg.SearchMode
		if method == "hybrid" || method == "rrf" {
			hits[i].VectorScore = hits[i].Score
			hits[i].KeywordScore = hits[i].Score
			hits[i].RetrievalMethod = "hybrid"
		} else if method == "knn" {
			hits[i].VectorScore = hits[i].Score
			hits[i].KeywordScore = 0
			hits[i].RetrievalMethod = "vector"
		} else {
			hits[i].VectorScore = hits[i].Score
			hits[i].RetrievalMethod = method
		}
		if resolution != nil {
			hits[i].Metadata = gin.H{
				"knowledge_base_id":    resolution.KnowledgeBase.ID,
				"knowledge_base_name":  resolution.KnowledgeBase.Name,
				"knowledge_base_scope": resolution.KnowledgeBase.Scope,
			}
		}
	}

	searchResp := gin.H{
		"query":      query,
		"collection": indexName,
		"results":    hits,
	}

	if s.retrievalCache != nil {
		cacheKey := rag.RetrievalCacheKey{
			Collection: indexName,
			Query:      query,
			Mode:       s.cfg.SearchMode,
			Limit:      limit,
			Threshold:  s.cfg.MinScore,
		}
		searchHits := make([]rag.SearchHit, 0, len(hits))
		for _, h := range hits {
			searchHits = append(searchHits, rag.SearchHit{
				Content:    h.Content,
				Score:      h.Score,
				Source:     h.Source,
				Filename:   h.Filename,
				ChunkID:    h.ChunkID,
				DocumentID: h.DocumentID,
				ChunkIndex: h.ChunkIndex,
			})
		}
		s.retrievalCache.Set(cacheKey, &rag.SearchResponse{
			Query:      query,
			Collection: indexName,
			Results:    searchHits,
		})
	}

	c.JSON(http.StatusOK, searchResp)
}

// searchMultiKB performs parallel search across multiple KBs. Each
// goroutine builds its own per-request compiled retriever, queries
// its own ES index, and the results are merged.
func (s *Server) searchMultiKB(c *gin.Context, query string, kbIDs []int64, limit int) {
	ctx := c.Request.Context()

	type kbInfo struct {
		kb        *knowledgebase.KnowledgeBase
		indexName string
	}
	seen := map[string]bool{}
	var kbs []kbInfo
	for _, id := range kbIDs {
		resolution, err := s.kbs.Resolve(knowledgebase.ResolveRequest{KBID: &id})
		if err != nil {
			continue
		}
		indexName := resolution.KnowledgeBase.IndexName()
		if seen[indexName] {
			continue
		}
		seen[indexName] = true
		kbs = append(kbs, kbInfo{resolution.KnowledgeBase, indexName})
	}

	if len(kbs) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "no valid knowledge bases found"})
		return
	}

	type searchResult struct {
		docs []*rag.RetrievedDoc
		err  error
	}
	results := make([]searchResult, len(kbs))
	var wg sync.WaitGroup
	concurrency := len(kbs)
	if concurrency > 10 {
		concurrency = 10
	}
	sem := make(chan struct{}, concurrency)

	for i, kb := range kbs {
		wg.Add(1)
		go func(idx int, idxName string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			docs, err := s.retrieveAt(ctx, idxName, query, limit, s.cfg.MinScore, s.cfg.SearchMode)
			results[idx] = searchResult{docs: docs, err: err}
		}(i, kb.indexName)
	}
	wg.Wait()

	var merged []searchRawHit
	for i, info := range kbs {
		r := results[i]
		if r.err != nil {
			continue
		}
		hits := docsToHits(r.docs)
		for j := range hits {
			hits[j].RetrievalMethod = s.cfg.SearchMode
			if hits[j].RetrievalMethod == "hybrid" || hits[j].RetrievalMethod == "rrf" {
				hits[j].VectorScore = hits[j].Score
				hits[j].KeywordScore = hits[j].Score
			} else {
				hits[j].VectorScore = hits[j].Score
			}
			hits[j].Metadata = gin.H{
				"knowledge_base_id":    info.kb.ID,
				"knowledge_base_name":  info.kb.Name,
				"knowledge_base_scope": info.kb.Scope,
			}
		}
		merged = append(merged, hits...)
	}

	// Sort by score desc and truncate
	sortHitsByScore(merged)
	if limit > 0 && len(merged) > limit {
		merged = merged[:limit]
	}

	c.JSON(http.StatusOK, gin.H{
		"query":      query,
		"collection": "multi_kb",
		"results":    merged,
	})
}

func sortHitsByScore(hits []searchRawHit) {
	// Simple insertion sort; n is small (limit=5..20)
	for i := 1; i < len(hits); i++ {
		for j := i; j > 0 && hits[j-1].Score < hits[j].Score; j-- {
			hits[j-1], hits[j] = hits[j], hits[j-1]
		}
	}
}

// retrieveAt runs a per-request compiled retriever for the given KB index
// and returns the raw *rag.RetrievedDoc slice. Used by /search and
// /chat to do the retrieve step only; LLM generation is handled by
// chat after this returns.
// retrieveAt performs a search against the per-request indexName. It
// creates a per-index elastic_retriever bound to the target ES index
// (no placeholder/WithIndex routing needed since we compile per-request).
//
// Steps:
//  1. Embed the user's text query via s.embedder to get the dense vector
//  2. Build the hybrid/knn/keyword ES body via rag.BuildHybridQueryJSON
//  3. Create a per-index retriever from s.esClient
//  4. Retrieve with that retriever and map results to []*rag.RetrievedDoc
func (s *Server) retrieveAt(ctx context.Context, indexName, query string, topK int, minScore float64, searchMode string) ([]*rag.RetrievedDoc, error) {
	if s.esClient == nil {
		return nil, fmt.Errorf("retrieveAt: nil esClient")
	}
	if s.embedder == nil {
		return nil, fmt.Errorf("retrieveAt: nil embedder (cannot build hybrid query JSON)")
	}

	// 1. Embed the text query → dense vector.
	vectors, err := s.embedder.EmbedStrings(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("retrieveAt: embed query: %w", err)
	}
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return nil, fmt.Errorf("retrieveAt: embedder returned empty vector")
	}
	vector := vectors[0]

	// 2. Build the ES query body JSON.
	body, err := rag.BuildHybridQueryJSON(query, vector, topK, minScore, rag.DefaultWeights, searchMode)
	if err != nil {
		return nil, fmt.Errorf("retrieveAt: build query body: %w", err)
	}

	// 3. Create a per-index retriever bound to the target ES index.
	scoreThreshold := minScore
	retriever, err := elastic_retriever.NewRetriever(ctx, &elastic_retriever.RetrieverConfig{
		Client:         s.esClient,
		Index:          indexName,
		TopK:           topK,
		ScoreThreshold: &scoreThreshold,
		SearchMode:     elastic_search_mode.SearchModeRawStringRequest(),
		ResultParser:   rag.ProjectResultParser(),
	})
	if err != nil {
		return nil, fmt.Errorf("retrieveAt: create retriever for %q: %w", indexName, err)
	}

	// 4. Retrieve using the per-index retriever (no WithIndex needed).
	docs, err := retriever.Retrieve(ctx, body)
	if err != nil {
		return nil, err
	}
	return docs, nil
}

// chat handles /chat. Single-KB: builds a per-request
// BuildRetrievalGraphAt with closure-captured indexName and invokes
// the full retrieve+LLM pipeline. Multi-KB: delegates to chatMultiKB.
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

	if len(req.KBIDs) > 0 {
		s.chatMultiKB(c, &req)
		return
	}

	// Resolve KB (allow body-supplied kb_id to take precedence)
	if req.KBID == nil {
		req.KBID = parseIntPtr(c.Query("kb_id"))
	}
	resolution, indexName, err := s.resolveKBFromChat(&req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}
	if err := s.kbs.CheckEmbeddingMatch(resolution.KnowledgeBase); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}

	limit := req.Limit
	if limit <= 0 {
		limit = s.cfg.TopK
	}

	// Compile graph per-request with indexNames baked in.
	chain, err := rag.BuildRetrievalGraph(c.Request.Context(), s.esClient, s.llm, s.embedder, []string{indexName}, limit, s.cfg.MinScore, s.cfg.SearchMode)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	answer, err := chain.Invoke(c.Request.Context(), req.Query)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"query":      req.Query,
		"collection": indexName,
		"response":   answer,
		"sources":    []any{},
	})
}

// chatMultiKB resolves multiple KBs and invokes the pre-compiled graph
// once with all index names. The graph's multi_retrieve node fans out
// across all KB indices concurrently, merges results, and the LLM produces
// a single answer covering all knowledge bases.
func (s *Server) chatMultiKB(c *gin.Context, req *rag.ChatRequest) {
	limit := req.Limit
	if limit <= 0 {
		limit = s.cfg.TopK
	}

	seen := map[string]bool{}
	var indexNames []string
	for _, id := range req.KBIDs {
		resolution, err := s.kbs.Resolve(knowledgebase.ResolveRequest{KBID: &id})
		if err != nil {
			continue
		}
		idx := resolution.KnowledgeBase.IndexName()
		if seen[idx] {
			continue
		}
		seen[idx] = true
		indexNames = append(indexNames, idx)
	}

	if len(indexNames) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "no valid knowledge bases found"})
		return
	}

	// Compile graph per-request with all indexNames baked in.
	chain, err := rag.BuildRetrievalGraph(c.Request.Context(), s.esClient, s.llm, s.embedder, indexNames, limit, s.cfg.MinScore, s.cfg.SearchMode)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}
	answer, err := chain.Invoke(c.Request.Context(), req.Query)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"query":      req.Query,
		"collection": "multi_kb",
		"response":   answer,
		"sources":    []any{},
	})
}

// resolveKBFromChat resolves KB from ChatRequest fields.
func (s *Server) resolveKBFromChat(req *rag.ChatRequest) (*knowledgebase.Resolution, string, error) {
	if s.kbs == nil {
		return nil, "", fmt.Errorf("knowledge base service not configured")
	}

	scope := req.Scope
	var scopePtr *string
	if scope != "" {
		scopePtr = &scope
	}

	var collectionPtr *string
	if req.Collection != "" {
		collectionPtr = &req.Collection
	}

	resolution, err := s.kbs.Resolve(knowledgebase.ResolveRequest{
		KBID:       req.KBID,
		Collection: collectionPtr,
		Scope:      scopePtr,
		UserID:     req.UserID,
		AgentID:    req.AgentID,
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
