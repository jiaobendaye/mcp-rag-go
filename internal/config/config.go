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
	HTTPPort int `yaml:"http_port"` // HTTP server port, default 8060

	// Elasticsearch
	ESUrl    string `yaml:"es_url"`    // ES connection URL
	ESIndex  string `yaml:"es_index"`  // ES index name for single KB MVP

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
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		HTTPPort:          8060,
		ESUrl:             "http://localhost:9200",
		ESIndex:           "kb_1",
		EmbeddingProvider: "openai",
		EmbeddingModel:    "text-embedding-3-small",
		LLMProvider:       "openai",
		LLMModel:          "gpt-4o-mini",
		ChunkSize:         4000,
		ChunkOverlap:      200,
		TopK:              5,
		MinScore:          0.7,
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
