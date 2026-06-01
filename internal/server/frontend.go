package server

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"
)

// initSPA registers SPA static file serving routes.
func (s *Server) initSPA(r *gin.Engine) {
	staticDir := s.cfg.StaticDir
	if staticDir == "" {
		staticDir = "./static"
	}

	// Serve static files from the static directory
	r.Static("/static", staticDir)

	// SPA entry points
	r.GET("/app", s.serveSPA)
	r.GET("/app/*path", s.serveSPA)
}

// serveSPA serves the SPA index.html, falling back to a helpful page if not built.
func (s *Server) serveSPA(c *gin.Context) {
	staticDir := s.cfg.StaticDir
	if staticDir == "" {
		staticDir = "./static"
	}

	// Try known locations for SPA index.html
	for _, sub := range []string{"app", "spa", ""} {
		path := filepath.Join(staticDir, sub, "index.html")
		if _, err := os.Stat(path); err == nil {
			c.File(path)
			return
		}
	}

	// No SPA build found, serve fallback
	c.Data(http.StatusOK, "text/html; charset=utf-8", []byte(spaFallbackHTML))
}

// serveDocs serves the Scalar API documentation page.
func (s *Server) serveDocs(c *gin.Context) {
	c.Data(http.StatusOK, "text/html; charset=utf-8", []byte(scalarHTML))
}

// serveOpenAPI serves the OpenAPI 3.0 specification.
func (s *Server) serveOpenAPI(c *gin.Context) {
	c.JSON(http.StatusOK, buildOpenAPISpec())
}

// providerModels returns available models for a configured provider (stub-based).
// Uses the current config to build the model list, rather than calling remote APIs.
func (s *Server) providerModels(c *gin.Context) {
	provider := c.Param("provider")
	family := c.Query("family")

	// Use config from manager if available, otherwise use startup cfg
	var embedProvider, llmProvider, embedModel, llmModel string
	if s.configManager != nil {
		cfg := s.configManager.Get()
		embedProvider = cfg.EmbeddingProvider
		llmProvider = cfg.LLMProvider
		embedModel = cfg.EmbeddingModel
		llmModel = cfg.LLMModel
	} else {
		embedProvider = s.cfg.EmbeddingProvider
		llmProvider = s.cfg.LLMProvider
		embedModel = s.cfg.EmbeddingModel
		llmModel = s.cfg.LLMModel
	}

	models := []gin.H{}

	// Check if this provider matches the configured embedding provider
	if strings.EqualFold(provider, embedProvider) {
		if embedModel != "" {
			modelFamily := inferFamily(embedModel)
			if family == "" || family == modelFamily {
				models = append(models, gin.H{
					"id":     embedModel,
					"label":  embedModel,
					"family": modelFamily,
					"source": "config",
				})
			}
		}
	}

	// Check if this provider matches the configured LLM provider
	if strings.EqualFold(provider, llmProvider) {
		if llmModel != "" && llmModel != embedModel {
			modelFamily := inferFamily(llmModel)
			if family == "" || family == modelFamily {
				models = append(models, gin.H{
					"id":     llmModel,
					"label":  llmModel,
					"family": modelFamily,
					"source": "config",
				})
			}
		}
	}

	// Deduplicate
	seen := map[string]bool{}
	unique := make([]gin.H, 0, len(models))
	for _, m := range models {
		id := m["id"].(string)
		if !seen[id] {
			seen[id] = true
			unique = append(unique, m)
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"provider": provider,
		"family":   family,
		"models":   unique,
	})
}

// inferFamily classifies a model as "chat" or "embedding" based on its name.
func inferFamily(name string) string {
	lower := strings.ToLower(name)
	embeddingKeywords := []string{"embedding", "bge-", "m3e", "e5", "text-embedding"}
	for _, kw := range embeddingKeywords {
		if strings.Contains(lower, kw) {
			return "embedding"
		}
	}
	return "chat"
}

// buildOpenAPISpec returns a minimal OpenAPI 3.0 spec.
func buildOpenAPISpec() gin.H {
	return gin.H{
		"openapi": "3.0.3",
		"info": gin.H{
			"title":       "MCP-RAG API",
			"description": "RAG service with MCP protocol support",
			"version":     "1.0.0",
		},
		"servers": []gin.H{
			{"url": "http://localhost:8060", "description": "Local server"},
		},
		"paths": gin.H{
			"/health": gin.H{
				"get": gin.H{
					"summary":     "Service health",
					"operationId": "health",
					"responses": gin.H{
						"200": gin.H{"description": "Health status"},
					},
				},
			},
			"/ready": gin.H{
				"get": gin.H{
					"summary":     "Readiness check",
					"operationId": "ready",
					"responses": gin.H{
						"200": gin.H{"description": "Service ready"},
						"503": gin.H{"description": "Service not ready"},
					},
				},
			},
			"/config": gin.H{
				"get": gin.H{
					"summary":     "Get configuration",
					"operationId": "configGet",
					"responses": gin.H{
						"200": gin.H{"description": "Configuration"},
					},
				},
			},
			"/config/bulk": gin.H{
				"post": gin.H{
					"summary":     "Bulk update config",
					"operationId": "configSetBulk",
					"responses": gin.H{
						"200": gin.H{"description": "Config updated"},
					},
				},
			},
			"/chat": gin.H{
				"post": gin.H{
					"summary":     "RAG chat",
					"operationId": "chat",
					"responses": gin.H{
						"200": gin.H{"description": "Chat response"},
					},
				},
			},
			"/search": gin.H{
				"get": gin.H{
					"summary":     "Search documents",
					"operationId": "search",
					"responses": gin.H{
						"200": gin.H{"description": "Search results"},
					},
				},
			},
			"/knowledge-bases": gin.H{
				"get": gin.H{
					"summary":     "List knowledge bases",
					"operationId": "listKnowledgeBases",
					"responses": gin.H{
						"200": gin.H{"description": "Knowledge base list"},
					},
				},
				"post": gin.H{
					"summary":     "Create knowledge base",
					"operationId": "createKnowledgeBase",
					"responses": gin.H{
						"200": gin.H{"description": "Created knowledge base"},
					},
				},
			},
			"/add-document": gin.H{
				"post": gin.H{
					"summary":     "Add document",
					"operationId": "addDocument",
					"responses": gin.H{
						"200": gin.H{"description": "Document added"},
					},
				},
			},
			"/upload-files": gin.H{
				"post": gin.H{
					"summary":     "Upload files",
					"operationId": "uploadFiles",
					"responses": gin.H{
						"200": gin.H{"description": "Files uploaded"},
					},
				},
			},
			"/list-documents": gin.H{
				"get": gin.H{
					"summary":     "List documents",
					"operationId": "listDocuments",
					"responses": gin.H{
						"200": gin.H{"description": "Document list"},
					},
				},
			},
			"/list-files": gin.H{
				"get": gin.H{
					"summary":     "List files",
					"operationId": "listFiles",
					"responses": gin.H{
						"200": gin.H{"description": "File list"},
					},
				},
			},
			"/providers/{provider}/models": gin.H{
				"get": gin.H{
					"summary":     "List provider models",
					"operationId": "providerModels",
					"parameters": []gin.H{
						{"name": "provider", "in": "path", "required": true, "schema": gin.H{"type": "string"}},
						{"name": "family", "in": "query", "schema": gin.H{"type": "string"}},
					},
					"responses": gin.H{
						"200": gin.H{"description": "Model list"},
					},
				},
			},
			"/metrics": gin.H{
				"get": gin.H{
					"summary":     "Observability metrics",
					"operationId": "metrics",
					"responses": gin.H{
						"200": gin.H{"description": "Metrics snapshot"},
					},
				},
			},
		},
	}
}

// spaFallbackHTML is served when no SPA build is found.
const spaFallbackHTML = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MCP-RAG Dashboard</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 80px auto; padding: 20px; line-height: 1.6; }
  h1 { color: #333; } pre { background: #f5f5f5; padding: 16px; border-radius: 8px; overflow-x: auto; }
  .note { color: #666; font-size: 14px; }
</style>
</head>
<body>
<h1>MCP-RAG Dashboard</h1>
<p>SPA 前端尚未构建或部署。</p>
<p>请将 SPA 构建产物放入 <code>static/app/</code> 目录，例如：</p>
<pre>cp -r /path/to/mcp-rag/frontend/dist/* static/app/</pre>
<p class="note">或从 Python 项目复制：<code>cp -r ../mcp-rag/src/mcp_rag/static/app/* static/app/</code></p>
<hr>
<p><a href="/docs">API 文档</a> | <a href="/health">健康检查</a></p>
</body>
</html>`

// scalarHTML is the Scalar API documentation page loaded from CDN.
const scalarHTML = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MCP-RAG API Docs</title>
</head>
<body>
<script id="api-reference" data-url="/openapi.json"></script>
<script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
</body>
</html>`
