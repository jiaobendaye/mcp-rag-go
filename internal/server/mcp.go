package server

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/mark3labs/mcp-go/mcp"
	mcpserver "github.com/mark3labs/mcp-go/server"

	"github.com/jiaobendaye/mcp-rag-go/internal/knowledgebase"
	"github.com/jiaobendaye/mcp-rag-go/internal/rag"
)

// InitMCP initializes the MCP server and returns both the MCPServer and StreamableHTTPServer.
func (s *Server) InitMCP() (*mcpserver.MCPServer, *mcpserver.StreamableHTTPServer) {
	mcpSrv := mcpserver.NewMCPServer("mcp-rag", "1.0.0",
		mcpserver.WithToolCapabilities(false),
	)

	// Register rag_ask tool
	tool := mcp.NewTool("rag_ask",
		mcp.WithDescription("查询知识库，基于检索到的相关内容回答问题。支持两种模式: raw(仅检索)和summary(检索+LLM总结)"),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("查询问题"),
		),
		mcp.WithString("mode",
			mcp.Description("模式: raw (仅检索) 或 summary (检索+LLM总结)，默认raw"),
		),
		mcp.WithString("collection",
			mcp.Description("知识库名称 (向后兼容)"),
		),
		mcp.WithInteger("kb_id",
			mcp.Description("知识库ID"),
		),
		mcp.WithString("scope",
			mcp.Description("知识库作用域: public 或 agent_private"),
		),
		mcp.WithInteger("limit",
			mcp.Description("返回结果数量 (1-20，默认5)"),
		),
		mcp.WithNumber("threshold",
			mcp.Description("相似度阈值 (0-1，默认0.7)"),
		),
		mcp.WithString("tenant",
			mcp.Description("租户标识"),
		),
		mcp.WithInteger("user_id",
			mcp.Description("用户ID"),
		),
		mcp.WithInteger("agent_id",
			mcp.Description("AI代理ID"),
		),
		mcp.WithString("api_key",
			mcp.Description("API密钥"),
		),
		mcp.WithString("request_id",
			mcp.Description("请求ID (用于追踪)"),
		),
		mcp.WithString("trace_id",
			mcp.Description("追踪ID (用于分布式追踪)"),
		),
		mcp.WithArray("kb_ids",
			mcp.Description("多知识库ID列表，用于跨知识库聚合检索"),
		),
	)

	mcpSrv.AddTool(tool, s.handleRagAsk)

	return mcpSrv, mcpserver.NewStreamableHTTPServer(mcpSrv)
}

// handleRagAsk handles the rag_ask MCP tool call.
func (s *Server) handleRagAsk(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	// Extract query (required)
	query, err := request.RequireString("query")
	if err != nil {
		return mcp.NewToolResultError("query is required"), nil
	}

	// Extract mode (raw or summary)
	mode := request.GetString("mode", "raw")

	// Extract collection (for legacy KB resolution)
	collection := request.GetString("collection", "default")

	// Extract KB parameters
	kbID := request.GetInt("kb_id", 0)
	kbIDsRaw := extractKBIDsArg(request.Params.Arguments)
	kbIDs := mcpExtractKBIDs(kbIDsRaw)
	scope := request.GetString("scope", "")
	userID := request.GetInt("user_id", 0)
	agentID := request.GetInt("agent_id", 0)

	// Extract search parameters
	limit := request.GetInt("limit", 5)
	if limit < 1 {
		limit = 1
	}
	if limit > 20 {
		limit = 20
	}

	threshold := request.GetFloat("threshold", 0.7)
	if threshold < 0 {
		threshold = 0
	}
	if threshold > 1 {
		threshold = 1
	}

	// Multi-KB mode
	if len(kbIDs) > 0 {
		return s.handleRagAskMultiKB(ctx, query, mode, kbIDs, limit, threshold)
	}

	// Resolve KB to get index name
	_, indexName, err := s.resolveMCPKB(kbID, scope, userID, agentID, collection)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("知识库解析失败: %v", err)), nil
	}

	switch mode {
	case "summary":
		return s.handleSummaryMode(ctx, query, limit, threshold, indexName)
	default:
		return s.handleRawMode(ctx, query, limit, threshold, indexName)
	}
}

// resolveMCPKB resolves KB from MCP parameters.
func (s *Server) resolveMCPKB(kbID int, scope string, userID, agentID int, collection string) (*knowledgebase.Resolution, string, error) {
	var kbIDPtr *int64
	if kbID > 0 {
		id := int64(kbID)
		kbIDPtr = &id
	}

	var scopePtr *string
	if scope != "" {
		scopePtr = &scope
	}

	var userIDPtr *int64
	if userID > 0 {
		uid := int64(userID)
		userIDPtr = &uid
	}

	var agentIDPtr *int64
	if agentID > 0 {
		aid := int64(agentID)
		agentIDPtr = &aid
	}

	legacyKey := ""
	if kbIDPtr == nil && scopePtr == nil {
		legacyKey = "legacy:public:" + collection
	}

	resolution, err := s.kbs.Resolve(knowledgebase.ResolveRequest{
		KBID:              kbIDPtr,
		Scope:             scopePtr,
		UserID:            userIDPtr,
		AgentID:           agentIDPtr,
		LegacyCollection:  &collection,
		LegacyCollectionKey: &legacyKey,
	})
	if err != nil {
		return nil, "", err
	}
	return resolution, resolution.KnowledgeBase.IndexName(), nil
}

// handleRawMode performs search-only and returns formatted text.
func (s *Server) handleRawMode(ctx context.Context, query string, limit int, threshold float64, indexName string) (*mcp.CallToolResult, error) {
	// Embed query
	vecs, err := s.embedder.EmbedStrings(ctx, []string{query})
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("向量化失败: %v", err)), nil
	}

	// Search with configured mode
	hits, err := s.searcher.SearchWithMode(ctx, query, toFloat32(vecs[0]), limit, threshold, s.cfg.SearchMode)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("检索失败: %v", err)), nil
	}

	if len(hits) == 0 {
		return mcp.NewToolResultText(fmt.Sprintf("未找到与 \"%s\" 相关的信息。", query)), nil
	}

	// Format results as text
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("查询: %s\n", query))
	sb.WriteString(fmt.Sprintf("模式: 检索 (共%d条结果)\n\n", len(hits)))
	for i, h := range hits {
		sb.WriteString(fmt.Sprintf("--- 结果 %d (相似度: %.3f) ---\n", i+1, h.Score))
		if h.Filename != "" {
			sb.WriteString(fmt.Sprintf("文件: %s\n", h.Filename))
		}
		if h.Source != "" {
			sb.WriteString(fmt.Sprintf("来源: %s\n", h.Source))
		}
		sb.WriteString(fmt.Sprintf("\n%s\n\n", h.Content))
	}

	return mcp.NewToolResultText(sb.String()), nil
}

// handleSummaryMode uses ChatService for search + LLM summary.
func (s *Server) handleSummaryMode(ctx context.Context, query string, limit int, threshold float64, indexName string) (*mcp.CallToolResult, error) {
	resp, err := s.chatSvc.Chat(ctx, &rag.ChatRequest{Query: query})
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("对话生成失败: %v", err)), nil
	}

	// Format response with sources
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("查询: %s\n", resp.Query))
	sb.WriteString(fmt.Sprintf("模式: 总结\n\n"))
	sb.WriteString(resp.Response)
	if len(resp.Sources) > 0 {
		sb.WriteString("\n\n--- 参考来源 ---\n")
		for i, src := range resp.Sources {
			sb.WriteString(fmt.Sprintf("%d. [相似度: %.3f]", i+1, src.Score))
			if src.Filename != "" {
				sb.WriteString(fmt.Sprintf(" 文件: %s", src.Filename))
			}
			sb.WriteString("\n")
		}
	}

	return mcp.NewToolResultText(sb.String()), nil
}

// mcpListTools handles GET /debug/mcp/tools - lists all registered MCP tools.
func (s *Server) mcpListTools(c *gin.Context) {
	tools := s.mcpSrv.ListTools()
	result := make([]gin.H, 0, len(tools))
	for _, st := range tools {
		result = append(result, gin.H{
			"name":         st.Tool.Name,
			"description":  st.Tool.Description,
			"input_schema": st.Tool.InputSchema,
		})
	}
	c.JSON(http.StatusOK, gin.H{"tools": result})
}

// mcpDebugCall handles POST /debug/mcp/call - debug invoke an MCP tool.
func (s *Server) mcpDebugCall(c *gin.Context) {
	var req struct {
		Tool      string         `json:"tool"`
		Arguments map[string]any `json:"arguments"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid request body"})
		return
	}
	if req.Tool == "" {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "tool name is required"})
		return
	}

	serverTool := s.mcpSrv.GetTool(req.Tool)
	if serverTool == nil {
		c.JSON(http.StatusNotFound, gin.H{"detail": fmt.Sprintf("tool not found: %s", req.Tool)})
		return
	}

	// Build CallToolRequest
	callReq := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      req.Tool,
			Arguments: req.Arguments,
		},
	}

	result, err := serverTool.Handler(c.Request.Context(), callReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	// Extract text content from result
	var texts []string
	for _, content := range result.Content {
		if textContent, ok := content.(mcp.TextContent); ok {
			texts = append(texts, textContent.Text)
		}
	}

	if result.IsError {
		c.JSON(http.StatusOK, gin.H{
			"tool":     req.Tool,
			"is_error": true,
			"content":  strings.Join(texts, "\n"),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"tool":     req.Tool,
		"is_error": false,
		"content":  strings.Join(texts, "\n"),
	})
}

// extractKBIDsArg extracts kb_ids from the raw MCP arguments (which is typed as `any`).
func extractKBIDsArg(args any) any {
	m, ok := args.(map[string]any)
	if !ok {
		return nil
	}
	return m["kb_ids"]
}

// mcpExtractKBIDs extracts kb_ids from various MCP argument formats.
func mcpExtractKBIDs(raw any) []int64 {
	if raw == nil {
		return nil
	}
	switch v := raw.(type) {
	case []any:
		var ids []int64
		for _, item := range v {
			switch n := item.(type) {
			case float64:
				ids = append(ids, int64(n))
			case int64:
				ids = append(ids, n)
			case int:
				ids = append(ids, int64(n))
			}
		}
		return ids
	case string:
		return parseKBIDs(v)
	}
	return nil
}

// handleRagAskMultiKB performs multi-KB retrieval for MCP rag_ask.
func (s *Server) handleRagAskMultiKB(ctx context.Context, query string, mode string, kbIDs []int64, limit int, threshold float64) (*mcp.CallToolResult, error) {
	// Resolve KBs
	type kbInfo struct {
		kb        *knowledgebase.KnowledgeBase
		indexName string
	}
	seen := map[string]bool{}
	var kbs []kbInfo
	for _, id := range kbIDs {
		resolution, err := s.kbs.Resolve(knowledgebase.ResolveRequest{KBID: &id})
		if err != nil {
			continue
		}
		indexName := resolution.KnowledgeBase.IndexName()
		if seen[indexName] {
			continue
		}
		seen[indexName] = true
		kbs = append(kbs, kbInfo{resolution.KnowledgeBase, indexName})
	}

	if len(kbs) == 0 {
		return mcp.NewToolResultError("no valid knowledge bases found"), nil
	}

	// Embed once
	vecs, err := s.embedder.EmbedStrings(ctx, []string{query})
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("向量化失败: %v", err)), nil
	}
	queryVec := toFloat32(vecs[0])

	// Parallel search
	type searchResult struct {
		hits []rag.SearchHit
		err  error
	}
	results := make([]searchResult, len(kbs))
	var wg sync.WaitGroup
	for i, kb := range kbs {
		wg.Add(1)
		go func(idx int, info kbInfo) {
			defer wg.Done()
			hits, err := s.searcher.SearchWithMode(ctx, query, queryVec, limit, threshold, s.cfg.SearchMode)
			results[idx] = searchResult{hits: hits, err: err}
		}(i, kb)
	}
	wg.Wait()

	// Collect and aggregate
	var kbResults []rag.KBSearchResult
	for i, info := range kbs {
		r := results[i]
		if r.err != nil {
			continue
		}
		kbResults = append(kbResults, rag.KBSearchResult{
			KBID:         info.kb.ID,
			KBName:       info.kb.Name,
			KBScope:      string(info.kb.Scope),
			OwnerUserID:  info.kb.OwnerUserID,
			OwnerAgentID: info.kb.OwnerAgentID,
			Hits:         r.hits,
		})
	}

	aggregated := rag.AggregateResults(kbResults, limit)

	if len(aggregated) == 0 {
		return mcp.NewToolResultText(fmt.Sprintf("未找到与 \"%s\" 相关的信息。", query)), nil
	}

	// Format results
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("查询: %s\n", query))
	sb.WriteString(fmt.Sprintf("模式: 跨%d个知识库检索 (共%d条结果)\n\n", len(kbIDs), len(aggregated)))
	for i, ar := range aggregated {
		sb.WriteString(fmt.Sprintf("--- 结果 %d (相似度: %.3f, 知识库: %s) ---\n", i+1, ar.Score, ar.KBName))
		if ar.Filename != "" {
			sb.WriteString(fmt.Sprintf("文件: %s\n", ar.Filename))
		}
		if ar.Source != "" {
			sb.WriteString(fmt.Sprintf("来源: %s\n", ar.Source))
		}
		sb.WriteString(fmt.Sprintf("\n%s\n\n", ar.Content))
	}

	if mode == "summary" {
		sb.WriteString("--- 摘要模式 ---\n")
		sb.WriteString("跨知识库摘要生成功能暂未启用。")
	}

	return mcp.NewToolResultText(sb.String()), nil
}
