package config

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.HTTPPort != 8060 {
		t.Errorf("expected HTTPPort=8060, got %d", cfg.HTTPPort)
	}
	if cfg.ChunkSize != 4000 {
		t.Errorf("expected ChunkSize=4000, got %d", cfg.ChunkSize)
	}
	if cfg.TopK != 5 {
		t.Errorf("expected TopK=5, got %d", cfg.TopK)
	}
	if cfg.MinScore != 0.7 {
		t.Errorf("expected MinScore=0.7, got %f", cfg.MinScore)
	}
	if cfg.ESIndex != "kb_1" {
		t.Errorf("expected ESIndex=kb_1, got %s", cfg.ESIndex)
	}
}

func TestLoadFromYAML(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	// Write a config file with custom values
	content := `
http_port: 9090
es_url: "http://es:9200"
chunk_size: 2000
`
	if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := Load(configPath)
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	if cfg.HTTPPort != 9090 {
		t.Errorf("expected HTTPPort=9090, got %d", cfg.HTTPPort)
	}
	if cfg.ESUrl != "http://es:9200" {
		t.Errorf("expected ESUrl=http://es:9200, got %s", cfg.ESUrl)
	}
	if cfg.ChunkSize != 2000 {
		t.Errorf("expected ChunkSize=2000, got %d", cfg.ChunkSize)
	}
	// Default values should remain for non-overridden fields
	if cfg.TopK != 5 {
		t.Errorf("expected default TopK=5, got %d", cfg.TopK)
	}
}

func TestLoadMissingFile(t *testing.T) {
	cfg, err := Load("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("Load() should succeed with missing file, got: %v", err)
	}
	// Should return defaults
	if cfg.HTTPPort != 8060 {
		t.Errorf("expected default HTTPPort=8060, got %d", cfg.HTTPPort)
	}
}

func TestEnvVarOverride(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	// Write a config that sets http_port to 8080
	content := `http_port: 8080`
	if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	// Set env var to override
	t.Setenv("MCP_RAG_HTTP_PORT", "9090")
	t.Setenv("MCP_RAG_TOP_K", "10")
	t.Setenv("MCP_RAG_MIN_SCORE", "0.85")

	cfg, err := Load(configPath)
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	// Env var should override YAML value
	if cfg.HTTPPort != 9090 {
		t.Errorf("expected env override HTTPPort=9090, got %d", cfg.HTTPPort)
	}
	// Env var should override default
	if cfg.TopK != 10 {
		t.Errorf("expected env override TopK=10, got %d", cfg.TopK)
	}
	if cfg.MinScore != 0.85 {
		t.Errorf("expected env override MinScore=0.85, got %f", cfg.MinScore)
	}
}

func TestInvalidYAML(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	// Malformed YAML
	content := `http_port: [invalid`
	if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	_, err := Load(configPath)
	if err == nil {
		t.Error("Load() should return error for malformed YAML")
	}
}

func TestYAMLMarshalRoundtrip(t *testing.T) {
	cfg := DefaultConfig()
	cfg.HTTPPort = 9999
	cfg.ChunkSize = 1000

	data, err := yaml.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	// Unmarshal back
	var restored Config
	if err := yaml.Unmarshal(data, &restored); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if restored.HTTPPort != 9999 {
		t.Errorf("expected HTTPPort=9999, got %d", restored.HTTPPort)
	}
	if restored.ChunkSize != 1000 {
		t.Errorf("expected ChunkSize=1000, got %d", restored.ChunkSize)
	}
}

func TestEnvVarFallback(t *testing.T) {
	// OPENAI_API_KEY fallback
	t.Setenv("OPENAI_API_KEY", "sk-test-key")

	cfg := DefaultConfig()
	applyEnvOverrides(cfg)

	if cfg.EmbeddingAPIKey != "sk-test-key" {
		t.Errorf("expected EmbeddingAPIKey=sk-test-key, got %s", cfg.EmbeddingAPIKey)
	}
	if cfg.LLMAPIKey != "sk-test-key" {
		t.Errorf("expected LLMAPIKey=sk-test-key, got %s", cfg.LLMAPIKey)
	}

	// MCP_RAG_* should take priority over OPENAI_API_KEY
	t.Setenv("MCP_RAG_LLM_API_KEY", "sk-llm-specific")
	cfg2 := DefaultConfig()
	applyEnvOverrides(cfg2)

	if cfg2.LLMAPIKey != "sk-llm-specific" {
		t.Errorf("expected LLMAPIKey=sk-llm-specific, got %s", cfg2.LLMAPIKey)
	}
	if cfg2.EmbeddingAPIKey != "sk-test-key" {
		t.Errorf("expected EmbeddingAPIKey=sk-test-key (fallback), got %s", cfg2.EmbeddingAPIKey)
	}
}
