// Package config provides configuration loading and management for mcp-rag-go.
package config

import (
	"fmt"
	"os"
	"reflect"
	"sync"
	"sync/atomic"
)

// ConfigManager wraps a *Config with hot-reload support and CRUD operations.
type ConfigManager struct {
	mu          sync.RWMutex
	config      *Config
	configPath  string
	lastModTime int64 // st_mtime as UnixNano
	revision    atomic.Int64
}

// NewConfigManager creates a ConfigManager and loads the config file.
func NewConfigManager(configPath string) (*ConfigManager, error) {
	cm := &ConfigManager{configPath: configPath}
	if err := cm.Load(); err != nil {
		return nil, err
	}
	return cm, nil
}

// Load reads the config file and populates the in-memory Config.
func (cm *ConfigManager) Load() error {
	cfg, err := Load(cm.configPath)
	if err != nil {
		return fmt.Errorf("config load: %w", err)
	}

	cm.mu.Lock()
	cm.config = cfg
	cm.mu.Unlock()

	// Track file modification time
	cm.updateModTime()

	cm.revision.Add(1)
	return nil
}

// ReloadIfChanged checks the file mtime and reloads only when it changed.
// Returns true when a reload happened.
func (cm *ConfigManager) ReloadIfChanged() (bool, error) {
	info, err := os.Stat(cm.configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, fmt.Errorf("stat config: %w", err)
	}

	newMtime := info.ModTime().UnixNano()

	cm.mu.RLock()
	current := cm.lastModTime
	cm.mu.RUnlock()

	if newMtime == current {
		return false, nil
	}

	if err := cm.Reload(); err != nil {
		return false, err
	}
	return true, nil
}

// Reload forces a re-read of the config file regardless of mtime.
func (cm *ConfigManager) Reload() error {
	return cm.Load()
}

// Get returns a pointer to the current Config value (read-only intended).
func (cm *ConfigManager) Get() *Config {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.config
}

// Revision returns the current revision counter.
func (cm *ConfigManager) Revision() int64 {
	return cm.revision.Load()
}

// ConfigPath returns the config file path.
func (cm *ConfigManager) ConfigPath() string {
	return cm.configPath
}

// Set updates a single config field identified by its yaml tag name.
// Returns an error when the key is unknown or the type is incompatible.
func (cm *ConfigManager) Set(key string, value any) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	val := reflect.ValueOf(cm.config).Elem()
	typ := val.Type()

	for i := 0; i < val.NumField(); i++ {
		field := typ.Field(i)
		tag := field.Tag.Get("yaml")
		if tag != key {
			continue
		}

		fieldVal := val.Field(i)
		if !fieldVal.CanSet() {
			return fmt.Errorf("field %s is not settable", key)
		}

		rv := reflect.ValueOf(value)
		if !rv.Type().AssignableTo(fieldVal.Type()) {
			// Attempt conversion for numeric types
			if converted, ok := coerceValue(rv, fieldVal.Type()); ok {
				fieldVal.Set(reflect.ValueOf(converted))
				return nil
			}
			return fmt.Errorf("type mismatch for %s: want %s, got %s", key, fieldVal.Type(), rv.Type())
		}
		fieldVal.Set(rv)
		return nil
	}

	return fmt.Errorf("unknown config key: %s", key)
}

// Reset replaces the current config with defaults and increments revision.
func (cm *ConfigManager) Reset() {
	cm.mu.Lock()
	cm.config = DefaultConfig()
	cm.mu.Unlock()
	cm.updateModTime()
	cm.revision.Add(1)
}

// GetAll returns the full config as a flat map keyed by yaml tag names.
func (cm *ConfigManager) GetAll() map[string]any {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	out := make(map[string]any)

	val := reflect.ValueOf(cm.config).Elem()
	typ := val.Type()

	for i := 0; i < val.NumField(); i++ {
		field := typ.Field(i)
		tag := field.Tag.Get("yaml")
		if tag == "" {
			continue
		}
		out[tag] = val.Field(i).Interface()
	}

	return out
}

// updateModTime records the current config file's modification time.
func (cm *ConfigManager) updateModTime() {
	info, err := os.Stat(cm.configPath)
	if err != nil {
		cm.lastModTime = 0
		return
	}
	cm.lastModTime = info.ModTime().UnixNano()
}

// coerceValue attempts to convert a reflect.Value to a target Go type.
func coerceValue(rv reflect.Value, targetType reflect.Type) (any, bool) {
	switch targetType.Kind() {
	case reflect.Int:
		switch rv.Kind() {
		case reflect.Float64:
			return int(rv.Float()), true
		case reflect.Int, reflect.Int64:
			return int(rv.Int()), true
		}
	case reflect.Float64:
		switch rv.Kind() {
		case reflect.Int, reflect.Int64:
			return float64(rv.Int()), true
		case reflect.Float64:
			return rv.Float(), true
		}
	case reflect.String:
		if rv.Kind() == reflect.String {
			return rv.String(), true
		}
	case reflect.Bool:
		if rv.Kind() == reflect.Bool {
			return rv.Bool(), true
		}
	case reflect.Slice:
		// Handle []string
		if rv.Kind() == reflect.Slice {
			return rv.Interface(), true
		}
	}
	return nil, false
}
