// Package server provides the HTTP API for MCP-RAG.
package server

import (
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"

	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino/compose"
	"github.com/gin-gonic/gin"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/knowledgebase"
	"github.com/jiaobendaye/mcp-rag-go/internal/rag"
	"github.com/jiaobendaye/mcp-rag-go/internal/security"
)

// Server holds all dependencies for HTTP handlers.
type Server struct {
	cfg        *config.Config
	chain compose.Runnable[document.Source, []string]
	chatSvc    *rag.ChatService
	searcher   rag.Searcher
	embedder   rag.Embedder
	kbs        *knowledgebase.Service
	esIndexer  *rag.ES8Indexer
}

// New creates a new Server with all dependencies.
func New(cfg *config.Config, chain compose.Runnable[document.Source, []string], chatSvc *rag.ChatService, searcher rag.Searcher, embedder rag.Embedder, kbs *knowledgebase.Service, esIndexer *rag.ES8Indexer) *Server {
	return &Server{
		cfg:        cfg,
		chain:      chain,
		chatSvc:    chatSvc,
		searcher:   searcher,
		embedder:   embedder,
		kbs:        kbs,
		esIndexer:  esIndexer,
	}
}

// Setup registers all routes on the Gin engine.
func (s *Server) Setup() *gin.Engine {
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())

	// Security middleware (auth + rate-limit, no-op when disabled)
	r.Use(SecurityMiddleware(s.cfg))

	// System
	r.GET("/health", s.health)

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

// health responds with service status.
func (s *Server) health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
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
	c.JSON(http.StatusOK, gin.H{"message": "File deleted successfully"})
}

// listKnowledgeBases returns accessible knowledge bases.
func (s *Server) listKnowledgeBases(c *gin.Context) {
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
	var req struct {
		Name          string `json:"name"`
		Scope         string `json:"scope"`
		OwnerUserID   *int64 `json:"owner_user_id"`
		OwnerAgentID  *int64 `json:"owner_agent_id"`
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
		results = append(results, gin.H{
			"content":  h.Content,
			"score":    h.Score,
			"metadata": gin.H{},
			"source":   h.Source,
			"filename": h.Filename,
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"query":      query,
		"collection": s.cfg.ESIndex,
		"results":    results,
	})
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

	resp, err := s.chatSvc.Chat(c.Request.Context(), &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	c.JSON(http.StatusOK, resp)
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
