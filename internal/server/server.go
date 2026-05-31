// Package server provides the HTTP API for MCP-RAG.
package server

import (
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"

	"github.com/gin-gonic/gin"

	"github.com/jiaobendaye/mcp-rag-go/internal/config"
	"github.com/jiaobendaye/mcp-rag-go/internal/rag"
)

// Server holds all dependencies for HTTP handlers.
type Server struct {
	cfg      *config.Config
	pipeline *rag.IndexPipeline
	chatSvc  *rag.ChatService
	searcher rag.Searcher
	embedder rag.Embedder
}

// New creates a new Server with all dependencies.
func New(cfg *config.Config, pipeline *rag.IndexPipeline, chatSvc *rag.ChatService, searcher rag.Searcher, embedder rag.Embedder) *Server {
	return &Server{
		cfg:      cfg,
		pipeline: pipeline,
		chatSvc:  chatSvc,
		searcher: searcher,
		embedder: embedder,
	}
}

// Setup registers all routes on the Gin engine.
func (s *Server) Setup() *gin.Engine {
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())

	// System
	r.GET("/health", s.health)

	// Document
	r.POST("/add-document", s.addDocument)
	r.POST("/upload-files", s.uploadFiles)

	// Search & Chat
	r.GET("/search", s.search)
	r.POST("/chat", s.chat)

	return r
}

// health responds with service status.
func (s *Server) health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

// addDocument handles text document indexing.
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

	result, err := s.pipeline.IndexText(c.Request.Context(), req.Content, "manual_input")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":     "Document added successfully",
		"document_id": result.DocumentID,
		"chunk_count": result.ChunkCount,
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

		// Index the file
		result, err := s.pipeline.IndexFile(c.Request.Context(), tmpPath)
		if err != nil {
			results = append(results, fileResult{
				Filename:      fh.Filename,
				ContentLength: fh.Size,
				Error:         err.Error(),
			})
			continue
		}

		successful++
		results = append(results, fileResult{
			Filename:      fh.Filename,
			ContentLength: fh.Size,
			Processed:     true,
		})

		// Cleanup temp file
		os.Remove(tmpPath)

		// Result is unused but confirms successful indexing
		_ = result
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

	// Search
	hits, err := s.searcher.Search(c.Request.Context(), toFloat32(vecs[0]), limit, s.cfg.MinScore)
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
