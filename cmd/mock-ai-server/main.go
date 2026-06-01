// Command mock-ai-server implements an OpenAI-compatible mock server for testing.
// It supports /v1/embeddings, /v1/chat/completions, and /v1/models endpoints,
// returning deterministic responses without any real AI backend.
package main

import (
	"crypto/md5"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"time"
)

var (
	addr    = flag.String("addr", ":11435", "listen address")
	verbose = flag.Bool("v", false, "verbose logging")
)

func main() {
	flag.Parse()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", handleHealth)
	mux.HandleFunc("/v1/embeddings", handleEmbeddings)
	mux.HandleFunc("/v1/chat/completions", handleChatCompletions)
	mux.HandleFunc("/v1/models", handleModels)

	srv := &http.Server{
		Addr:         *addr,
		Handler:      withLogging(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	log.Printf("Mock AI server listening on %s", *addr)
	log.Printf("  POST /v1/embeddings")
	log.Printf("  POST /v1/chat/completions")
	log.Printf("  GET  /v1/models")
	log.Printf("  GET  /health")
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("server: %v", err)
	}
}

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		if *verbose {
			log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
		}
	})
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

type embeddingRequest struct {
	Input any    `json:"input"`
	Model string `json:"model"`
}

type embeddingResponse struct {
	Object string          `json:"object"`
	Data   []embeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  usageInfo       `json:"usage"`
}

type embeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type usageInfo struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
		return
	}

	var req embeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	// Determine how many inputs
	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []any:
		for _, item := range v {
			if s, ok := item.(string); ok {
				inputs = append(inputs, s)
			}
		}
	default:
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid input type"})
		return
	}

	data := make([]embeddingData, len(inputs))
	for i, text := range inputs {
		data[i] = embeddingData{
			Object:    "embedding",
			Embedding: deterministicEmbedding(text, 1024),
			Index:     i,
		}
	}

	resp := embeddingResponse{
		Object: "list",
		Data:   data,
		Model:  "mock-embedding",
		Usage: usageInfo{
			PromptTokens: len(inputs) * 10,
			TotalTokens:  len(inputs) * 10,
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

// deterministicEmbedding generates a repeatable pseudo-random vector from text.
func deterministicEmbedding(text string, dims int) []float64 {
	vec := make([]float64, dims)
	hash := md5.Sum([]byte(text))
	for i := range vec {
		// Mix hash bytes to produce deterministic values in [-1, 1]
		b := hash[i%len(hash)]
		phase := float64(b) / 255.0 * 2.0 * math.Pi
		vec[i] = math.Sin(phase*float64(i+1)) * 0.5
	}
	// Normalize
	var norm float64
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
	return vec
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   usageInfo    `json:"usage"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
		return
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	// Extract user query from messages
	userQuery := ""
	for _, msg := range req.Messages {
		if msg.Role == "user" {
			userQuery = msg.Content
		}
	}

	// Build a mock response that references the query
	answer := fmt.Sprintf(
		"这是基于知识库的模拟回答。根据您的问题「%s」，相关内容显示该技术的核心特点是高性能、易用性和可扩展性。",
		truncate(userQuery, 50),
	)

	resp := chatResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []chatChoice{
			{
				Index: 0,
				Message: chatMessage{
					Role:    "assistant",
					Content: answer,
				},
				FinishReason: "stop",
			},
		},
		Usage: usageInfo{
			PromptTokens: len(userQuery) / 4,
			TotalTokens:  len(userQuery)/4 + len(answer)/4,
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

type modelsResponse struct {
	Object string       `json:"object"`
	Data   []modelEntry `json:"data"`
}

type modelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	resp := modelsResponse{
		Object: "list",
		Data: []modelEntry{
			{ID: "mock-embedding", Object: "model", Created: 1700000000, OwnedBy: "mock"},
			{ID: "mock-chat", Object: "model", Created: 1700000000, OwnedBy: "mock"},
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func truncate(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen]) + "..."
}
