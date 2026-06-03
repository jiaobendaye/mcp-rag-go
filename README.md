# mcp-rag-go

基于 [CloudWeGo eino](https://github.com/cloudwego/eino) 框架的 RAG（检索增强生成）服务，支持 MCP 协议。Elasticsearch 作为向量/混合检索引擎，SQLite 管理知识库元数据，可接入任意 OpenAI 兼容的 Embedding 和 LLM 提供商。

## 架构

```
┌──────────────────────────────────────────────────────────┐
│                        HTTP API                          │
│  /chat  /search  /add-document  /upload-files  /mcp ... │
├──────────────────────────────────────────────────────────┤
│  Gin Middleware Pipeline                                 │
│  RequestID → SlogAccessLog → Recovery → Tracing →        │
│  Security → Idempotency → Handler                        │
├──────────────────────────────────────────────────────────┤
│                      RAG Pipeline (eino)                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │ parse    │──▶│ retrieve │──▶│ assemble │──▶ LLM      │
│  │ input    │   │ (hybrid) │   │ (dedup)  │             │
│  └──────────┘   └────┬─────┘   └──────────┘             │
│                      │                                    │
│            ┌─────────┼─────────┐                          │
│            ▼         ▼         ▼                          │
│         ES Index  ES Index  ES Index  (per-KB index)     │
├──────────────────────────────────────────────────────────┤
│                      Infrastructure                       │
│  SQLite (KB metadata, migrations, idempotency cache)     │
│  Elasticsearch (document vectors, BM25 + KNN retrieval)  │
│  Langfuse (LLM observability — token usage, cost, trace) │
│  slog (structured JSON logging)                           │
└──────────────────────────────────────────────────────────┘
```

## 功能特性

| 模块 | 说明 |
|------|------|
| **RAG 引擎** | 混合检索（BM25 + KNN 手动融合） / RRF / 纯向量，基于 eino Graph 每请求编译 |
| **多知识库** | 单 KB 检索 + 跨 KB 并行检索 + 结果合并排序 |
| **知识库管理** | SQLite 存储 KB 元数据（名称/作用域/嵌入模型绑定），1 KB = 1 ES Index |
| **MCP 协议** | 实现 `rag_ask` 工具，支持 Streamable HTTP 传输 |
| **文档处理** | 支持文本/PDF/Word/Markdown，自动分块（可配 chunk_size/chunk_overlap） |
| **嵌入模型** | SwappableEmbedder 代理模式，支持未来热切换；兼容 OpenAI/Ark/Ollama |
| **LLM 提供商** | 任意 OpenAI 兼容 API（DeepSeek / GPT / Qwen / Ollama 等） |
| **Langfuse 追踪** | 自动记录每次 ChatModel 调用的 token 用量、延迟、输入输出 |
| **结构化日志** | slog JSON 格式，每请求带 `request_id` 贯穿全链路 |
| **幂等性** | `Idempotency-Key` 头，24h TTL，防重复写入 |
| **安全** | API Key 认证 / 租户隔离 / 速率限制 / 上传配额 |
| **容器化** | 多阶段构建，distroless 运行时，非 root 用户，镜像 < 50MB |
| **配置热重载** | 运行时动态修改配置（chunk_size / top_k / search_mode 等），无需重启 |
| **SPA 前端** | 内置 Vue 前端（/app），文档/知识库/配置管理界面 |
| **SQLite 迁移** | 嵌入式 migration 系统（checksum 校验 + 事务保护） |

## 快速开始

### 前置条件

- Go 1.22+
- Elasticsearch 8.x（本地或 Docker）
- LLM API Key（DeepSeek / OpenAI 等）

### 1. 启动 Elasticsearch

```bash
docker compose up -d
# 只启动 ES，不启动 mcp-rag
```

### 2. 配置

复制并编辑环境变量文件：

```bash
cp .env.example .env   # 按需创建
```

`.env` 示例：

```env
# LLM API Key（必填）
MCP_RAG_LLM_API_KEY=sk-your-api-key
```

`config.yaml` 关键配置：

```yaml
http_port: 8060
es_url: "http://localhost:9200"

# Embedding 提供商
embedding_provider: "ollama"
embedding_model: "mxbai-embed-large"
embedding_base_url: "http://localhost:11434/v1"

# LLM 提供商
llm_provider: "openai"
llm_model: "deepseek-v4-flash"
llm_base_url: "https://your-llm-proxy/v1"

# RAG 参数
chunk_size: 4000
chunk_overlap: 200
top_k: 5
min_score: 0.2
search_mode: "hybrid"   # hybrid | rrf | knn
```

### 3. 运行

```bash
# 前台运行
make run

# 或后台运行
make serve
make logs    # 查看日志
make status  # 查看状态
make stop    # 停止
```

访问：
- API: `http://localhost:8060`
- 前端: `http://localhost:8060/app`
- 健康检查: `http://localhost:8060/health`
- 版本信息: `http://localhost:8060/version`

### 4. 快速验证

```bash
# 创建知识库
curl -XPOST http://localhost:8060/knowledge-bases \
  -H 'Content-Type: application/json' \
  -d '{"name":"我的知识库"}'

# 添加文档
curl -XPOST http://localhost:8060/add-document?kb_id=1 \
  -H 'Content-Type: application/json' \
  -d '{"content":"Elasticsearch 是一个基于 Lucene 的分布式搜索和分析引擎。"}'

# 搜索
curl 'http://localhost:8060/search?kb_id=1&query=什么是Elasticsearch' | jq .

# RAG 问答
curl -XPOST http://localhost:8060/chat \
  -H 'Content-Type: application/json' \
  -d '{"kb_id":1,"query":"Elasticsearch是什么？"}' | jq .
```

## Docker 部署

### 构建镜像

```bash
make docker-build
# 或手动：
docker build -t mcp-rag:latest .
```

### docker compose 全栈

```bash
# ES + mcp-rag 一起启动
docker compose --profile full up -d

# 查看日志
docker compose logs -f mcp-rag

# 停止
docker compose --profile full down
```

### 单独运行容器

```bash
docker run --rm -p 8060:8060 \
  -e MCP_RAG_ES_URL=http://host.docker.internal:9200 \
  -e MCP_RAG_LLM_API_KEY=sk-xxx \
  -e MCP_RAG_LLM_BASE_URL=https://your-llm-proxy/v1 \
  -e MCP_RAG_EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1 \
  -v mcp-rag-data:/data \
  mcp-rag:latest
```

## 配置说明

### 配置文件

默认读取 `config.yaml`（可通过 `--config` 指定路径）。所有配置项都可通过环境变量 `MCP_RAG_*` 覆盖，优先级：**环境变量 > YAML 文件 > 默认值**。

### 核心配置项

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|---------|--------|------|
| `http_port` | `MCP_RAG_HTTP_PORT` | `8060` | HTTP 监听端口 |
| `log_level` | `MCP_RAG_LOG_LEVEL` | `info` | 日志级别: debug/info/warn/error |
| `es_url` | `MCP_RAG_ES_URL` | `http://localhost:9200` | Elasticsearch 地址 |
| `knowledge_base_db_path` | `MCP_RAG_KNOWLEDGE_BASE_DB_PATH` | `./data/knowledge_bases.sqlite3` | SQLite 数据库路径 |
| `static_dir` | `MCP_RAG_STATIC_DIR` | `./static` | 前端静态文件目录 |
| `embedding_provider` | `MCP_RAG_EMBEDDING_PROVIDER` | `openai` | 嵌入提供商: openai/ark/ollama |
| `embedding_model` | `MCP_RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | 嵌入模型名称 |
| `embedding_base_url` | `MCP_RAG_EMBEDDING_BASE_URL` | - | 嵌入 API 地址 |
| `embedding_api_key` | `MCP_RAG_EMBEDDING_API_KEY` | - | 嵌入 API Key（空则复用 OPENAI_API_KEY） |
| `llm_provider` | `MCP_RAG_LLM_PROVIDER` | `openai` | LLM 提供商 |
| `llm_model` | `MCP_RAG_LLM_MODEL` | `gpt-4o-mini` | LLM 模型名称 |
| `llm_base_url` | `MCP_RAG_LLM_BASE_URL` | - | LLM API 地址 |
| `llm_api_key` | `MCP_RAG_LLM_API_KEY` | - | LLM API Key |
| `chunk_size` | `MCP_RAG_CHUNK_SIZE` | `4000` | 文本分块大小（字符） |
| `chunk_overlap` | `MCP_RAG_CHUNK_OVERLAP` | `200` | 分块重叠大小 |
| `top_k` | `MCP_RAG_TOP_K` | `5` | 检索返回数量 |
| `min_score` | `MCP_RAG_MIN_SCORE` | `0.7` | 最小相似度阈值 |
| `search_mode` | `MCP_RAG_SEARCH_MODE` | `hybrid` | 检索模式: hybrid/rrf/knn |

### 安全配置

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|---------|--------|------|
| `security_enabled` | `MCP_RAG_SECURITY_ENABLED` | `false` | 启用 API Key 认证 |
| `security_allow_anonymous` | `MCP_RAG_SECURITY_ALLOW_ANON` | `true` | 允许匿名访问 |
| `security_api_keys` | - | `[]` | 全局 API Key 列表 |
| `security_tenant_api_keys` | - | `{}` | 租户级 API Key 映射 |

### 速率限制与配额

| 配置项 | 环境变量 | 默认值 |
|--------|---------|--------|
| `rate_limit_requests_per_window` | `MCP_RAG_RATE_LIMIT_RPW` | `120` |
| `rate_limit_window_seconds` | `MCP_RAG_RATE_LIMIT_WINDOW` | `60` |
| `rate_limit_burst` | `MCP_RAG_RATE_LIMIT_BURST` | `30` |
| `quota_max_upload_files` | `MCP_RAG_QUOTA_UPLOAD_FILES` | `20` |
| `quota_max_upload_bytes` | `MCP_RAG_QUOTA_UPLOAD_BYTES` | `52428800` (50MB) |
| `quota_max_index_documents` | `MCP_RAG_QUOTA_INDEX_DOCS` | `500` |

### Langfuse 可观测性

设置以下环境变量即可启用：

```env
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
LANGFUSE_PUBLIC_KEY="pk-lf-xxx"
LANGFUSE_SECRET_KEY="sk-lf-xxx"
```

## API 概览

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查（含 ES 连通性） |
| `GET` | `/ready` | 就绪检查（503 if ES 不通） |
| `GET` | `/metrics` | 运行时指标 |
| `GET` | `/version` | 版本信息 |

### 知识库

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/knowledge-bases?user_id=N` | 列出可访问的知识库 |
| `POST` | `/knowledge-bases` | 创建知识库 `{"name":"...","scope":"public"}` |
| `GET` | `/collections` | 列出所有知识库名称 |

### 文档

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/add-document?kb_id=1` | 添加文本文档 `{"content":"..."}` |
| `POST` | `/upload-files?kb_id=1` | 上传文件（multipart，`files` 字段） |
| `GET` | `/list-documents?kb_id=1&limit=100&offset=0` | 列出文档 |
| `DELETE` | `/delete-document` | 删除文档 `{"document_id":"..."}` |
| `GET` | `/list-files?kb_id=1` | 按文件名聚合的文件列表 |
| `DELETE` | `/delete-file` | 按文件名删除 `{"filename":"..."}` |

### 检索与问答

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/search?query=...&kb_id=1&limit=5` | 纯检索（不经过 LLM） |
| `POST` | `/chat` | RAG 问答 `{"query":"...","kb_id":1,"limit":5}` |

多知识库检索：`GET /search?query=...&kb_ids=1,2,3` 或 `POST /chat` 带 `"kb_ids":[1,2,3]`。

### MCP 协议

| 方法 | 路径 | 说明 |
|------|------|------|
| `ANY` | `/mcp` | Streamable HTTP 端点 |
| `ANY` | `/mcp/*path` | MCP 子路径 |

MCP 工具 `rag_ask` 参数：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 查询问题 |
| `mode` | string | 否 | `raw`（仅检索）或 `summary`（检索+LLM），默认 `raw` |
| `kb_id` | int | 否 | 知识库 ID |
| `collection` | string | 否 | 知识库名称 |
| `scope` | string | 否 | 作用域: `public` / `agent_private` |
| `limit` | int | 否 | 返回数量 1-20，默认 5 |
| `threshold` | float | 否 | 相似度阈值 0-1 |

### 配置管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/config` | 获取全部配置 |
| `POST` | `/config` | 修改单个配置 `{"key":"top_k","value":10}` |
| `POST` | `/config/bulk` | 批量修改 `{"updates":{"top_k":10,"chunk_size":2000}}` |
| `POST` | `/config/reset` | 重置为默认值 |
| `POST` | `/config/reload` | 从磁盘重新加载 |

### 管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/admin/idempotency-cleanup` | 清理过期幂等性缓存 |

## MCP 客户端集成

```json
{
  "mcpServers": {
    "mcp-rag": {
      "url": "http://localhost:8060/mcp"
    }
  }
}
```

在支持 MCP 的客户端（如 Claude Desktop、Cursor 等）中配置后，即可通过 `rag_ask` 工具查询知识库。

## Observability（可观测性）

### 结构化日志 (slog)

所有日志以 JSON 格式输出到 stdout：

```json
{"time":"2026-06-03T10:00:00Z","level":"INFO","msg":"MCP-RAG listening","port":8060}
```

每条 HTTP 请求自动附带 `request_id`、`method`、`path`、`status`、`latency_ms`、`client_ip`：

```json
{"time":"...","level":"INFO","msg":"access","request_id":"abc123","method":"POST","path":"/chat","status":200,"latency_ms":342,"client_ip":"127.0.0.1"}
```

前端请求会通过 `X-Request-Id` 响应头返回 `request_id`，方便前后端日志关联。

### Langfuse 追踪

设置 `LANGFUSE_*` 环境变量后，每次 RAG 调用自动记录：

- **ChatModel Generation**：模型名、输入 messages、输出内容、token 用量
- **检索 Span**：查询文本、ES 检索耗时
- **Trace Metadata**：user_id、session_id（agent_xxx）、request_id、query 原文

在 Langfuse Dashboard 中可按用户/会话/请求维度查看完整的 RAG 调用链路和成本。

### 监控端点

- `/health` — 健康检查（含 ES 连通性、组件状态、运行时间、请求/错误计数）
- `/metrics` — 运行时指标快照（各操作计数、平均延迟、错误率）
- `/ready` — Kubernetes readiness probe（ES 不通返回 503）

## 开发指南

```bash
# 克隆
git clone git@github.com:jiaobendaye/mcp-rag-go.git
cd mcp-rag-go

# 安装依赖
go mod download

# 运行单元测试
make test

# HTML 覆盖率报告
make cover

# 代码检查
make lint

# 集成测试（需要 Docker，自动启动临时 ES 容器）
make test-integration

# 编译
make build
```

### 常用命令

| 命令 | 说明 |
|------|------|
| `make help` | 列出所有可用目标 |
| `make run` | 前台运行服务 |
| `make serve` | 后台启动（写 PID 到 `.run/`） |
| `make stop` | 停止后台服务 |
| `make logs` | tail 服务日志 |
| `make restart` | 重启服务 |
| `make reset` | 清理全部状态 + 重建 + 重启 |
| `make clean` | 删除构建产物 |
| `make clean-state` | 停止服务 + 删除 SQLite DB |
| `make clean-es` | 删除所有 ES 索引 |
| `make clean-all` | 以上三项全部 |
| `make idem-cleanup` | 清理过期幂等性缓存 |
| `make docker-build` | 构建 Docker 镜像 |
| `make docker-run` | 构建并前台运行容器 |

### 项目结构

```
mcp-rag-go/
├── cmd/
│   ├── mcp-rag/          # 主服务入口
│   └── mock-ai-server/   # 测试用 mock LLM/Embedding 服务器
├── internal/
│   ├── config/           # 配置管理（YAML + env + 热重载）
│   ├── embedder/          # SwappableEmbedder 代理（支持未来热切换）
│   ├── knowledgebase/    # 知识库 CRUD（SQLite Store + Service）
│   ├── migrations/       # SQLite 迁移脚本（embed FS）
│   ├── observability/    # slog Logger + 指标收集器
│   ├── rag/              # RAG 核心（检索 Graph、索引链、缓存）
│   ├── security/         # 认证中间件 + 速率限制 + 配额策略
│   └── server/           # HTTP 服务（路由、中间件、MCP 端点、幂等性）
├── frontend/             # Vue SPA 前端
├── static/               # 编译后的前端静态文件
├── data/                 # 运行时数据（SQLite DB，gitignore）
├── config.yaml           # 默认配置文件
├── config.docker.yaml    # 容器内配置文件（baked into image）
├── docker-compose.yml    # 开发环境编排（ES + mcp-rag）
├── Dockerfile            # 多阶段构建（distroless）
├── Makefile              # 构建/测试/部署命令
└── docs/                 # 技术文档
```

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 语言 | Go 1.23+ |
| RAG 框架 | CloudWeGo eino v0.9 |
| HTTP 框架 | Gin v1.12 |
| 搜索引擎 | Elasticsearch 8.x |
| 元数据存储 | SQLite3 (go-sqlite3) |
| MCP 协议 | mark3labs/mcp-go v0.54 |
| LLM 调用 | eino-ext openai adapter |
| 可观测性 | slog + Langfuse (eino callbacks) |
| 容器 | distroless/static-debian12 (non-root) |

## License

MIT
