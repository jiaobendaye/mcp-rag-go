package config

import (
	"os"
	"testing"
)

func createTempConfig(t *testing.T, content string) string {
	t.Helper()
	path := t.TempDir() + "/config_test.yaml"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write temp config: %v", err)
	}
	return path
}

func TestNewConfigManager(t *testing.T) {
	path := createTempConfig(t, "http_port: 9999\ntop_k: 20\n")
	cm, err := NewConfigManager(path)
	if err != nil {
		t.Fatalf("NewConfigManager: %v", err)
	}

	cfg := cm.Get()
	if cfg.HTTPPort != 9999 {
		t.Errorf("expected HTTPPort=9999, got %d", cfg.HTTPPort)
	}
	if cfg.TopK != 20 {
		t.Errorf("expected TopK=20, got %d", cfg.TopK)
	}
	if cm.Revision() != 1 {
		t.Errorf("expected revision=1, got %d", cm.Revision())
	}
	if cm.ConfigPath() != path {
		t.Errorf("expected configPath=%s, got %s", path, cm.ConfigPath())
	}
}

func TestReloadIfChanged(t *testing.T) {
	path := createTempConfig(t, "http_port: 1111\n")
	cm, err := NewConfigManager(path)
	if err != nil {
		t.Fatalf("NewConfigManager: %v", err)
	}

	// No change
	changed, err := cm.ReloadIfChanged()
	if err != nil {
		t.Fatalf("ReloadIfChanged: %v", err)
	}
	if changed {
		t.Error("expected no change")
	}

	// Change file
	if err := os.WriteFile(path, []byte("http_port: 2222\n"), 0644); err != nil {
		t.Fatalf("write updated config: %v", err)
	}

	changed, err = cm.ReloadIfChanged()
	if err != nil {
		t.Fatalf("ReloadIfChanged after write: %v", err)
	}
	if !changed {
		t.Error("expected change detected")
	}
	if cm.Get().HTTPPort != 2222 {
		t.Errorf("expected HTTPPort=2222, got %d", cm.Get().HTTPPort)
	}
	if cm.Revision() != 2 {
		t.Errorf("expected revision=2, got %d", cm.Revision())
	}
}

func TestReload(t *testing.T) {
	path := createTempConfig(t, "http_port: 1111\n")
	cm, err := NewConfigManager(path)
	if err != nil {
		t.Fatalf("NewConfigManager: %v", err)
	}

	// Modify file
	if err := os.WriteFile(path, []byte("http_port: 2222\n"), 0644); err != nil {
		t.Fatalf("write updated config: %v", err)
	}

	// Force reload
	if err := cm.Reload(); err != nil {
		t.Fatalf("Reload: %v", err)
	}
	if cm.Get().HTTPPort != 2222 {
		t.Errorf("expected HTTPPort=2222, got %d", cm.Get().HTTPPort)
	}
}

func TestSet(t *testing.T) {
	path := createTempConfig(t, "http_port: 1111\ntop_k: 10\nmin_score: 0.5\n")
	cm, err := NewConfigManager(path)
	if err != nil {
		t.Fatalf("NewConfigManager: %v", err)
	}

	// Set int
	if err := cm.Set("http_port", 8888); err != nil {
		t.Fatalf("Set http_port: %v", err)
	}
	if cm.Get().HTTPPort != 8888 {
		t.Errorf("expected HTTPPort=8888, got %d", cm.Get().HTTPPort)
	}

	// Set float from float64 (JSON numbers are float64)
	if err := cm.Set("min_score", 0.9); err != nil {
		t.Fatalf("Set min_score: %v", err)
	}
	if cm.Get().MinScore != 0.9 {
		t.Errorf("expected MinScore=0.9, got %f", cm.Get().MinScore)
	}

	// Set int from float64 (JSON coercion)
	if err := cm.Set("top_k", float64(25)); err != nil {
		t.Fatalf("Set top_k from float64: %v", err)
	}
	if cm.Get().TopK != 25 {
		t.Errorf("expected TopK=25, got %d", cm.Get().TopK)
	}

	// Unknown key
	err = cm.Set("nonexistent", 1)
	if err == nil {
		t.Error("expected error for unknown key")
	}
}

func TestSetFromFloat64(t *testing.T) {
	path := createTempConfig(t, "")
	cm, err := NewConfigManager(path)
	if err != nil {
		t.Fatalf("NewConfigManager: %v", err)
	}

	// JSON unmarshals numbers as float64, so Set receives float64
	if err := cm.Set("http_port", float64(6060)); err != nil {
		t.Fatalf("Set http_port from float64: %v", err)
	}
	if cm.Get().HTTPPort != 6060 {
		t.Errorf("expected HTTPPort=6060, got %d", cm.Get().HTTPPort)
	}

	// Float64 to float64
	if err := cm.Set("min_score", float64(0.3)); err != nil {
		t.Fatalf("Set min_score from float64: %v", err)
	}
	if cm.Get().MinScore != 0.3 {
		t.Errorf("expected MinScore=0.3, got %f", cm.Get().MinScore)
	}
}

func TestReset(t *testing.T) {
	path := createTempConfig(t, "http_port: 9999\ntop_k: 50\n")
	cm, err := NewConfigManager(path)
	if err != nil {
		t.Fatalf("NewConfigManager: %v", err)
	}

	cm.Reset()

	if cm.Get().HTTPPort != 8060 {
		t.Errorf("expected default HTTPPort=8060 after reset, got %d", cm.Get().HTTPPort)
	}
	if cm.Get().TopK != 5 {
		t.Errorf("expected default TopK=5 after reset, got %d", cm.Get().TopK)
	}
}

func TestGetAll(t *testing.T) {
	path := createTempConfig(t, "http_port: 7777\ntop_k: 10\n")
	cm, err := NewConfigManager(path)
	if err != nil {
		t.Fatalf("NewConfigManager: %v", err)
	}

	all := cm.GetAll()
	if all["http_port"] != 7777 {
		t.Errorf("expected http_port=7777, got %v", all["http_port"])
	}
	if all["top_k"] != 10 {
		t.Errorf("expected top_k=10, got %v", all["top_k"])
	}
	// Check that some default keys exist
	if _, ok := all["min_score"]; !ok {
		t.Error("expected min_score in GetAll")
	}
}

func TestNewConfigManagerMissingFile(t *testing.T) {
	// File does not exist - should load defaults
	cm, err := NewConfigManager("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("NewConfigManager should succeed with missing file (use defaults): %v", err)
	}
	if cm.Get().HTTPPort != 8060 {
		t.Errorf("expected default HTTPPort=8060, got %d", cm.Get().HTTPPort)
	}
}
