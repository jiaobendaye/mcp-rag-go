// Package config provides configuration loading and management for mcp-rag-go.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Config holds all application configuration.
type Config struct {
	// Server
	HTTPPort  int    `yaml:"http_port"`  // HTTP server port, default 8060
	StaticDir string `yaml:"static_dir"` // static files directory, default "./static"

	// Elasticsearch
	ESUrl    string `yaml:"es_url"`    // ES connection URL
	ESIndex  string `yaml:"es_index"`  // ES index name (legacy, for single KB)

	// Knowledge Base
	KnowledgeBaseDBPath string `yaml:"knowledge_base_db_path"`

	// Embedding
	EmbeddingProvider string `yaml:"embedding_provider"` // "openai" | "ark" | "ollama"
	EmbeddingModel    string `yaml:"embedding_model"`    // e.g., "text-embedding-3-small"
	EmbeddingBaseURL  string `yaml:"embedding_base_url"`
	EmbeddingAPIKey   string `yaml:"embedding_api_key"`

	// LLM
	LLMProvider string `yaml:"llm_provider"` // "openai" | "ark" | "ollama"
	LLMModel    string `yaml:"llm_model"`    // e.g., "gpt-4o-mini"
	LLMBaseURL  string `yaml:"llm_base_url"`
	LLMAPIKey   string `yaml:"llm_api_key"`

	// RAG parameters
	ChunkSize    int     `yaml:"chunk_size"`    // chunk size for text splitting, default 4000
	ChunkOverlap int     `yaml:"chunk_overlap"` // chunk overlap, default 200
	TopK         int     `yaml:"top_k"`         // number of results, default 5
	MinScore     float64 `yaml:"min_score"`     // minimum similarity score, default 0.7
	SearchMode   string  `yaml:"search_mode"`   // "hybrid" | "rrf" | "knn", default "hybrid"

	// Security
	SecurityEnabled        bool              `yaml:"security_enabled"`
	SecurityAllowAnonymous  bool              `yaml:"security_allow_anonymous"`
	SecurityAPIKeys         []string          `yaml:"security_api_keys"`
	SecurityTenantAPIKeys   map[string][]string `yaml:"security_tenant_api_keys"`

	// Rate Limit
	RateLimitRequestsPerWindow int `yaml:"rate_limit_requests_per_window"`
	RateLimitWindowSeconds     int `yaml:"rate_limit_window_seconds"`
	RateLimitBurst             int `yaml:"rate_limit_burst"`

	// Quotas
	QuotaMaxUploadFiles      int `yaml:"quota_max_upload_files"`
	QuotaMaxUploadBytes      int `yaml:"quota_max_upload_bytes"`
	QuotaMaxUploadFileBytes  int `yaml:"quota_max_upload_file_bytes"`
	QuotaMaxIndexDocuments   int `yaml:"quota_max_index_documents"`
	QuotaMaxIndexChunks      int `yaml:"quota_max_index_chunks"`
	QuotaMaxIndexChars       int `yaml:"quota_max_index_chars"`
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		HTTPPort:          8060,
		StaticDir:         "./static",
		ESUrl:             "http://localhost:9200",
		ESIndex:           "kb_1",
		KnowledgeBaseDBPath: "./data/knowledge_bases.sqlite3",
		EmbeddingProvider: "openai",
		EmbeddingModel:    "text-embedding-3-small",
		LLMProvider:       "openai",
		LLMModel:          "gpt-4o-mini",
		ChunkSize:         4000,
		ChunkOverlap:      200,
		TopK:        5,
		MinScore:    0.7,
		SearchMode:  "hybrid",

		SecurityEnabled:            false,
		SecurityAllowAnonymous:     true,
		RateLimitRequestsPerWindow: 120,
		RateLimitWindowSeconds:     60,
		RateLimitBurst:             30,
		QuotaMaxUploadFiles:        20,
		QuotaMaxUploadBytes:        50 * 1024 * 1024,
		QuotaMaxUploadFileBytes:    10 * 1024 * 1024,
		QuotaMaxIndexDocuments:     500,
		QuotaMaxIndexChunks:        2000,
		QuotaMaxIndexChars:         500000,
	}
}

// Load reads config from config.yaml and overrides with MCP_RAG_* env vars.
func Load(configPath string) (*Config, error) {
	cfg := DefaultConfig()

	// 1. Load from YAML file (if exists)
	if data, err := os.ReadFile(configPath); err == nil {
		if err := yaml.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("load config %s: %w", configPath, err)
		}
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("read config %s: %w", configPath, err)
	}
	// File not found: use defaults (already set)

	// 2. Override with environment variables (MCP_RAG_ prefix)
	applyEnvOverrides(cfg)

	return cfg, nil
}

// applyEnvOverrides overrides config fields from MCP_RAG_* environment variables.
func applyEnvOverrides(cfg *Config) {
	envMap := map[string]func(string){
		"MCP_RAG_HTTP_PORT":            func(v string) { cfg.HTTPPort = parseInt(v, cfg.HTTPPort) },
		"MCP_RAG_ES_URL":               func(v string) { cfg.ESUrl = v },
		"MCP_RAG_ES_INDEX":             func(v string) { cfg.ESIndex = v },
		"MCP_RAG_EMBEDDING_PROVIDER":   func(v string) { cfg.EmbeddingProvider = v },
		"MCP_RAG_EMBEDDING_MODEL":      func(v string) { cfg.EmbeddingModel = v },
		"MCP_RAG_EMBEDDING_BASE_URL":   func(v string) { cfg.EmbeddingBaseURL = v },
		"MCP_RAG_EMBEDDING_API_KEY":    func(v string) { cfg.EmbeddingAPIKey = v },
		"MCP_RAG_LLM_PROVIDER":         func(v string) { cfg.LLMProvider = v },
		"MCP_RAG_LLM_MODEL":            func(v string) { cfg.LLMModel = v },
		"MCP_RAG_LLM_BASE_URL":         func(v string) { cfg.LLMBaseURL = v },
		"MCP_RAG_LLM_API_KEY":          func(v string) { cfg.LLMAPIKey = v },
		"MCP_RAG_CHUNK_SIZE":           func(v string) { cfg.ChunkSize = parseInt(v, cfg.ChunkSize) },
		"MCP_RAG_CHUNK_OVERLAP":        func(v string) { cfg.ChunkOverlap = parseInt(v, cfg.ChunkOverlap) },
		"MCP_RAG_TOP_K":                func(v string) { cfg.TopK = parseInt(v, cfg.TopK) },
		"MCP_RAG_MIN_SCORE":            func(v string) { cfg.MinScore = parseFloat(v, cfg.MinScore) },
		"MCP_RAG_SEARCH_MODE":          func(v string) { cfg.SearchMode = v },
		"MCP_RAG_SECURITY_ENABLED":     func(v string) { cfg.SecurityEnabled = parseBool(v) },
		"MCP_RAG_SECURITY_ALLOW_ANON":  func(v string) { cfg.SecurityAllowAnonymous = parseBool(v) },
		"MCP_RAG_RATE_LIMIT_RPW":       func(v string) { cfg.RateLimitRequestsPerWindow = parseInt(v, cfg.RateLimitRequestsPerWindow) },
		"MCP_RAG_RATE_LIMIT_WINDOW":    func(v string) { cfg.RateLimitWindowSeconds = parseInt(v, cfg.RateLimitWindowSeconds) },
		"MCP_RAG_RATE_LIMIT_BURST":     func(v string) { cfg.RateLimitBurst = parseInt(v, cfg.RateLimitBurst) },
		"MCP_RAG_QUOTA_UPLOAD_FILES":   func(v string) { cfg.QuotaMaxUploadFiles = parseInt(v, cfg.QuotaMaxUploadFiles) },
		"MCP_RAG_QUOTA_UPLOAD_BYTES":   func(v string) { cfg.QuotaMaxUploadBytes = parseInt(v, cfg.QuotaMaxUploadBytes) },
		"MCP_RAG_QUOTA_UPLOAD_FILE_BYTES": func(v string) { cfg.QuotaMaxUploadFileBytes = parseInt(v, cfg.QuotaMaxUploadFileBytes) },
		"MCP_RAG_QUOTA_INDEX_DOCS":     func(v string) { cfg.QuotaMaxIndexDocuments = parseInt(v, cfg.QuotaMaxIndexDocuments) },
		"MCP_RAG_QUOTA_INDEX_CHUNKS":   func(v string) { cfg.QuotaMaxIndexChunks = parseInt(v, cfg.QuotaMaxIndexChunks) },
		"MCP_RAG_QUOTA_INDEX_CHARS":    func(v string) { cfg.QuotaMaxIndexChars = parseInt(v, cfg.QuotaMaxIndexChars) },
	}

	for envKey, setter := range envMap {
		if val, ok := os.LookupEnv(envKey); ok && val != "" {
			setter(val)
		}
	}

	// Also check for OPENAI_API_KEY as fallback for both embedding and LLM
	if cfg.EmbeddingAPIKey == "" {
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			cfg.EmbeddingAPIKey = key
		}
	}
	if cfg.LLMAPIKey == "" {
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			cfg.LLMAPIKey = key
		}
	}
}

func parseInt(s string, defaultVal int) int {
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return defaultVal
	}
	return v
}

func parseFloat(s string, defaultVal float64) float64 {
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return defaultVal
	}
	return v
}

func parseBool(s string) bool {
	v, err := strconv.ParseBool(strings.TrimSpace(s))
	if err != nil {
		return false
	}
	return v
}
