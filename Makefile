# mcp-rag Makefile
# 用法:
#   make help        - 列出所有可用目标
#   make serve       - 后台启动服务(写 PID 到 .run/mcp-rag.pid,日志到 .run/mcp-rag.log)
#   make stop        - 停止后台服务
#   make restart     - 重启服务
#   make status      - 查看服务是否在跑
#   make logs        - tail 服务日志
#   make build       - 编译二进制
#   make test        - 跑单元测试
#   make cover       - 跑单元测试并产出 HTML coverage
#   make test-integration - 跑集成测试(需要 ES)
#   make lint        - 跑 golangci-lint
#   clean-*          - 各级清理(详见 help)

BIN_DIR     := bin
BIN         := $(BIN_DIR)/mcp-rag
PID_FILE    := .run/mcp-rag.pid
LOG_FILE    := .run/mcp-rag.log
CONFIG      := config.yaml
SQLITE_DB   := data/knowledge_bases.sqlite3
ES_URL      ?= http://localhost:9200

# 把所有 *.pid / *.log 集中在 .run/,目录不存在时 serve 目标会创建。
# 故意不用 nohup / disown: 进程从 make spawn 的子 shell 中 fork 出来,
# shell 退出后会被 init 收养,效果一致但语法更简单。

.DEFAULT_GOAL := help

.PHONY: help build run serve stop restart status logs
.PHONY: test cover test-integration lint
.PHONY: clean clean-state clean-es clean-all reset

help: ## 列出所有可用目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# 构建
# -----------------------------------------------------------------------------

build: ## 编译二进制到 bin/mcp-rag
	@mkdir -p $(BIN_DIR)
	go build -o $(BIN) ./cmd/mcp-rag

run: build ## 前台运行(不后台,Ctrl-C 退出)
	@set -a; . ./.env; set +a; ./$(BIN) serve --config $(CONFIG)

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

DOCKER_IMAGE ?= mcp-rag
DOCKER_TAG   ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

docker-build: ## 构建 Docker 镜像
	docker build \
		--build-arg VERSION=$(DOCKER_TAG) \
		--build-arg COMMIT=$(shell git rev-parse --short HEAD 2>/dev/null || echo unknown) \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG) \
		-t $(DOCKER_IMAGE):latest \
		.

docker-run: docker-build ## 前台运行 Docker 容器(映射端口 8060)
	docker run --rm -p 8060:8060 \
		-e LANGFUSE_BASE_URL \
		-e LANGFUSE_PUBLIC_KEY \
		-e LANGFUSE_SECRET_KEY \
		-e MCP_RAG_LLM_API_KEY \
		$(DOCKER_IMAGE):latest

docker-push: ## 推送 Docker 镜像到 ghcr.io
	docker tag $(DOCKER_IMAGE):latest ghcr.io/jiaobendaye/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push ghcr.io/jiaobendaye/$(DOCKER_IMAGE):$(DOCKER_TAG)

# -----------------------------------------------------------------------------
# 服务生命周期
# -----------------------------------------------------------------------------

serve: build ## 后台启动服务(写 PID、写日志,立即返回)
	@mkdir -p .run data
	@if [ -f $(PID_FILE) ] && kill -0 "$$(cat $(PID_FILE))" 2>/dev/null; then \
		echo "mcp-rag already running (PID $$(cat $(PID_FILE)))"; \
		echo "use: make stop   to stop, make logs to tail"; \
		exit 1; \
	fi
	@rm -f $(PID_FILE) $(LOG_FILE)
	@set -a; . ./.env; set +a; ./$(BIN) serve --config $(CONFIG) > $(LOG_FILE) 2>&1 & echo $$! > $(PID_FILE)
	@echo "mcp-rag started (PID $$(cat $(PID_FILE)))"
	@echo "  logs:  make logs"
	@echo "  status: make status"
	@echo "  stop:  make stop"

stop: ## 停止后台服务(根据 PID 文件)
	@if [ ! -f $(PID_FILE) ]; then \
		echo "not running (no $(PID_FILE))"; \
	else \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			kill $$PID && echo "stopped mcp-rag (PID $$PID)"; \
		else \
			echo "PID $$PID not alive (stale $(PID_FILE) removed)"; \
		fi; \
		rm -f $(PID_FILE); \
	fi

restart: stop serve ## 重启服务

status: ## 显示后台服务状态
	@if [ -f $(PID_FILE) ] && kill -0 "$$(cat $(PID_FILE))" 2>/dev/null; then \
		echo "mcp-rag running (PID $$(cat $(PID_FILE)))"; \
		ps -p "$$(cat $(PID_FILE))" -o pid,etime,rss,cmd 2>/dev/null || true; \
	else \
		echo "mcp-rag not running"; \
		[ -f $(PID_FILE) ] && echo "  (stale $(PID_FILE) present)"; \
		true; \
	fi

logs: ## tail 服务日志(Ctrl-C 退出)
	@if [ ! -f $(LOG_FILE) ]; then \
		echo "no log file at $(LOG_FILE) - did you 'make serve'?"; \
		exit 1; \
	fi
	tail -f $(LOG_FILE)

# -----------------------------------------------------------------------------
# 测试
# -----------------------------------------------------------------------------

test: ## 跑单元测试
	go test ./internal/... -count=1 -short

cover: ## 跑单元测试并产出 HTML coverage
	go test ./internal/... -coverprofile=coverage.out -count=1
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

test-integration: ## 跑集成测试(自动启动临时 ES 容器，结束后清理)
	@docker rm -f mcp-rag-itest-es 2>/dev/null || true
	go test -p 1 ./... -tags=integration -v -count=1
	@docker rm -f mcp-rag-itest-es 2>/dev/null || true

lint: ## 跑 golangci-lint
	golangci-lint run ./...

# -----------------------------------------------------------------------------
# 清理(分层)
# -----------------------------------------------------------------------------

clean: ## 删除构建产物
	rm -rf $(BIN_DIR) coverage.out coverage.html

clean-state: stop ## 停止服务并删除运行时状态(PID/日志/SQLite DB)
	rm -rf .run $(SQLITE_DB)

clean-es: ## 删除 mcp-rag 创建的所有 ES 索引(kb_*, rag_placeholder_index)
	@curl -sS -XDELETE "$(ES_URL)/kb_*,rag_placeholder_index" \
		| head -c 200 ; echo
	@echo "ES indices cleared."

clean-all: clean clean-state clean-es ## 清理构建 + 状态 + ES(完整重置前的最后一步)

reset: clean-all build serve ## 一键完整重置:清掉一切 → 重新构建 → 后台启动
	@echo "---"
	@make status

# -----------------------------------------------------------------------------
# Idempotency
# -----------------------------------------------------------------------------

idem-cleanup: ## 清理过期的幂等性缓存记录(24h TTL)
	@curl -sS -XPOST "http://localhost:8060/admin/idempotency-cleanup" \
		| head -c 500 ; echo
