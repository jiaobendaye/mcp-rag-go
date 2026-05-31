.PHONY: run build test test-integration lint cover clean

# Build
build:
	go build -o bin/mcp-rag ./cmd/mcp-rag

# Run
run:
	go run ./cmd/mcp-rag serve

# Unit tests
test:
	go test ./internal/... -v -count=1 -short

# Unit tests with coverage
cover:
	go test ./internal/... -coverprofile=coverage.out -count=1
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

# Integration tests (requires running ES)
test-integration:
	@echo "Starting ES for integration tests..."
	docker-compose up -d elasticsearch
	@sleep 5
	go test ./... -tags=integration -v -count=1 || true
	docker-compose down

# Lint
lint:
	golangci-lint run ./...

# Clean
clean:
	rm -rf bin/ coverage.out coverage.html
