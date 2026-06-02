# Embedding 模型热切换设计

## 动机

当前 embedder 在启动时创建一次，存为 `Server.embedder` 字段，所有请求共用。如需更换 embedding 模型，必须重启服务。需要支持不重启切换 embedding provider/model/base_url。

## 核心约束

- **不同模型的向量语义空间互不兼容**（同维度也不行），不能混用
- 切换 **必须全部迁移完成后才原子切换**，旧 embedder 持续服务直到切换完成
- 切换中间任何一步失败都能回滚，不影响线上服务

## 设计

### 1. SwappableEmbedder — 无锁热路径代理

```go
// internal/embedder/swappable.go

type SwappableEmbedder struct {
    current atomic.Pointer[embedding.Embedder]
}

func (s *SwappableEmbedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
    e := s.current.Load()
    return (*e).EmbedStrings(ctx, texts, opts...)
}

func (s *SwappableEmbedder) Swap(e embedding.Embedder) {
    s.current.Store(&e)
}
```

热路径仅一次 `atomic.Load` + 委托，零锁开销。`Server.embedder` 替换为此类型，`indexerConf.Embedding` 和 `kbRetriever` 内部引用也在启动时指向它。

### 2. 去掉预编译 graph，每请求编译

`Server.preCompiledGraph` 删除。`chat`、`chatMultiKB`、`handleSummaryMode` 三处改为每请求调用 `rag.BuildRetrievalGraph()`。

理由：
- `BuildRetrievalGraph` 编译开销 ~50μs，LLM 调用 1-10s，占比 <0.01%
- 消除 swap 时需要重编译 graph 的并发协调逻辑
- `BuildIndexChain` 已是同样模式

### 3. ConfigManager 回调

```go
// config/manager.go 新增
type ConfigChangeCallback func(oldCfg, newCfg *Config) error

func (cm *ConfigManager) OnConfigChange(cb ConfigChangeCallback) { ... }
```

`Set()` 和 `Load()` 成功后触发回调。回调在 HTTP handler goroutine 中同步执行，错误直接返回给 `/config` 调用方。

### 4. KB 绑定模型

`knowledge_bases` 表新增字段：

```sql
ALTER TABLE knowledge_bases ADD COLUMN embedding_model TEXT NOT NULL DEFAULT '';
ALTER TABLE knowledge_bases ADD COLUMN embedding_dims  INTEGER NOT NULL DEFAULT 0;
```

- `embedding_model`：语义标识（如 `"mxbai-embed-large"`），同维度不同模型向量空间不兼容
- `embedding_dims`：快速数值校验，维度不同一定不兼容；来自 probe `len(vecs[0])`

**Service 新增方法**：

```go
// 启动时注入，热切换后更新
func (s *Service) SetEmbeddingInfo(model string, dims int)

// 校验 KB 的模型/维度是否与当前 embedder 匹配
// 空值（"" / 0）视为未初始化，跳过校验
func (s *Service) CheckEmbeddingMatch(kb *KnowledgeBase) error
```

**创建 KB 时**：`Store.Create` 接收 `embeddingModel` 和 `embeddingDims` 参数（Service/Store 内部调用，外部通过 `SetEmbeddingInfo` 注入，Create 自动带出）。

**校验点**（resolve 后、embedder 使用前）：

| 操作 | 校验 | 理由 |
|------|------|------|
| `addDocument` | ✓ | 索引写入向量，必须匹配 |
| `search` / `chat` | ✓ | 检索用当前 embedder 编码 query，必须与索引一致 |
| `deleteDocument` | ✗ | 不需要 embedding |
| `Resolve`（自动创建） | ✗ | 新建 KB 自动记录当前模型，无需校验 |

切换前 resolve 时 `CheckEmbeddingMatch` 失败 → 说明该 KB 用的是旧模型，在迁移完成前仍用旧 embedder 服务。切换后旧 embedder 丢弃，所有 KB 指向新模型。

### 5. KB 迁移状态机

```
active → migrating → migrated → active（切换后）
  ↑                             |
  └──────── 失败回滚 ──────────┘
```

`knowledge_bases` 表新增 status 值：

| status | 含义 |
|--------|------|
| `active` | 正常服务（旧 embedder） |
| `migrating` | 正在重新索引 |
| `migrated` | 新 index 已就绪，等待全局切换 |

### 6. 迁移流程

```
1. POST /config/bulk
   {"embedding_model": "bge-m3", "embedding_base_url": "...", "embedding_api_key": "..."}
   → ConfigManager.Set() 写入配置
   → 回调：创建新 embedder → probe 验证连通性 → 存为 Server.pendingEmbedder
   → 返回 200 {"status": "pending", "kb_count": 5}

2. POST /admin/embedding/start-migration
   → 遍历所有 active KB，逐一迁移：
     a. UPDATE KB SET status='migrating'
     b. 从旧 ES index 读所有文档
     c. 用 pendingEmbedder 编码向量 → 写入新 ES index kb_N_v2
     d. 校验文档数一致
     e. UPDATE KB SET status='migrated'
        （collection_name 保持 kb_N，不切换引用 — 因为 active embedder 还是旧的）
     f. 失败则回滚 status='active'，旧 index 完好
   → 异步执行，返回 migration_id

3. GET /admin/embedding/migration-status
   → {
       "total": 5,
       "pending": 0,
       "migrating": 1,
       "migrated": 3,
       "failed": 0,
       "old_model": "mxbai-embed-large",
       "new_model": "bge-m3"
     }

4. POST /admin/embedding/complete-switch
   → 校验全部 migrated（无 pending/migrating）
   → SwappableEmbedder.Swap(pendingEmbedder)          // 原子切换 active embedder
   → 批量 UPDATE collection_name=kb_N_v2,
             embedding_model=bge-m3
     WHERE status='migrated'                           // 统一切换 ES index 引用
   → 删除旧 ES indices（kb_N 等）
   → 清理 pendingEmbedder
   → 返回 200
```

**迁移中 KB**：`add-document`/`delete-document` 返回 503；`search`/`chat` 仍用旧 index + 旧 embedder 正常服务。

### 7. 错误处理

| 场景 | 行为 |
|------|------|
| 新 embedder 创建失败（坏 URL/auth） | 回调返回 error，HTTP 400，旧 embedder 不变 |
| probe 失败（网络不通） | 同上 |
| 某个 KB 迁移失败 | 该 KB 回滚到 active，其他继续；status 显示 failed_count |
| 全部迁移前取消 | `POST /admin/embedding/cancel` 删除 pending embedder + 清理 kb_*_v2 indices |
| complete-switch 时还有未迁移 KB | 返回 409 Conflict，列出未迁移的 KB IDs |

### 8. 文件变更

| 文件 | 变更 |
|------|------|
| `internal/embedder/swappable.go` | 新增 SwappableEmbedder |
| `internal/config/manager.go` | 加 OnConfigChange callback |
| `internal/knowledgebase/models.go` | KB 加 `EmbeddingModel`、`EmbeddingDims` 字段；status 加 migrating/migrated |
| `internal/knowledgebase/store.go` | Create 加 embeddingModel/embeddingDims 参数；scanKB/scanKBs 读新列；支持 status 更新、collection_name 更新 |
| `internal/knowledgebase/service.go` | 新增 `SetEmbeddingInfo`、`CheckEmbeddingMatch`、迁移相关方法 |
| `internal/server/server.go` | embedder 改 SwappableEmbedder，去 preCompiledGraph，每请求编译 graph；addDocument/retrieveAt 调用 CheckEmbeddingMatch；加 /admin/embedding/* 端点 |
| `internal/server/mcp.go` | handleSummaryMode 改为每请求编译 graph |
| `cmd/mcp-rag/main.go` | 创建 SwappableEmbedder，注册 callback；启动时 `kbService.SetEmbeddingInfo()` |

## 验证

1. `go build ./...` + `go vet ./...`
2. `go test ./internal/... -count=1 -short` 全部通过
3. 集成测试：启动服务 → 添加文档 → 切换 embedding → 验证旧 KB 可用 → 迁移 → 切换 → 验证新旧 KB 都正常
4. 失败场景：坏 URL → 报错不变；迁移中取消 → 恢复
