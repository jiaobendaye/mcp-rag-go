#!/bin/bash
# e2e test suite for MCP-RAG Go server.
#
# Usage:
#   ./run.sh              # Full flow: setup -> test
#   ./run.sh setup        # Only build and copy artifacts to run/
#   ./run.sh test         # Run tests (assumes setup already done)
#   ./run.sh playwright   # Only run Playwright browser tests
#   ./run.sh clean        # Stop services and remove run/
#
# Requires: go, docker, node (for playwright), python3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$SCRIPT_DIR/run"

MOCK_PORT=11435
SERVER_PORT=18060
ES_PORT=19200
ES_CONTAINER="mcp-rag-e2e-es"
ES_URL="http://localhost:$ES_PORT"
ES_INDEX="e2e_test_kb"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0

pass() { echo -e "  ${GREEN}PASS${NC} $*"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}FAIL${NC} $*"; FAIL=$((FAIL + 1)); }
info() { echo -e "${CYAN}INFO${NC}  $*"; }
warn() { echo -e "${YELLOW}WARN${NC}  $*"; }

# ──────────────────────────────────────────────────────────────
# Setup phase
# ──────────────────────────────────────────────────────────────
do_setup() {
  echo "══════════════════════════════════════════════"
  echo "  E2E Setup: Building & Copying Artifacts"
  echo "══════════════════════════════════════════════"
  echo ""

  # Create run directory structure
  mkdir -p "$RUN_DIR/bin" "$RUN_DIR/static/app"

  # 1. Build binaries
  echo "==> Building binaries..."
  cd "$PROJECT_DIR"
  go build -o "$RUN_DIR/bin/mcp-rag" ./cmd/mcp-rag/
  info "Built mcp-rag"
  go build -o "$RUN_DIR/bin/mock-ai-server" ./cmd/mock-ai-server/
  info "Built mock-ai-server"

  # 2. Copy static files
  echo "==> Copying static files..."
  if [ -d "$PROJECT_DIR/static/app" ]; then
    cp -r "$PROJECT_DIR/static/app" "$RUN_DIR/static/"
    info "Copied static/app/"
  else
    warn "static/app/ not found, SPA tests may fail"
  fi

  # 3. Generate e2e config
  echo "==> Generating e2e config..."
  cat > "$RUN_DIR/config.yaml" << EOF
http_port: $SERVER_PORT
es_url: "$ES_URL"
es_index: "$ES_INDEX"
knowledge_base_db_path: "$RUN_DIR/e2e_kb.db"
static_dir: "$RUN_DIR/static"

embedding_provider: "mock"
embedding_model: "mock-embedding"
embedding_base_url: "http://localhost:$MOCK_PORT/v1"
embedding_api_key: ""

llm_provider: "mock"
llm_model: "mock-chat"
llm_base_url: "http://localhost:$MOCK_PORT/v1"
llm_api_key: ""

chunk_size: 1000
chunk_overlap: 100
top_k: 5
min_score: 0.0
search_mode: "hybrid"

security_enabled: true
security_allow_anonymous: true
security_api_keys: ["sk-e2e-test-key-123"]
EOF
  info "Generated $RUN_DIR/config.yaml"
  echo ""
}

# ──────────────────────────────────────────────────────────────
# Test phase
# ──────────────────────────────────────────────────────────────
do_test() {
  echo "══════════════════════════════════════════════"
  echo "  E2E Tests: Full Stack"
  echo "══════════════════════════════════════════════"
  echo ""

  # Check setup
  if [ ! -f "$RUN_DIR/bin/mcp-rag" ] || [ ! -f "$RUN_DIR/config.yaml" ]; then
    info "Artifacts not found, running setup first..."
    do_setup
  fi

  # ── 1. Start ES container (ephemeral, no volumes) ─────────
  echo "==> Starting Elasticsearch container ($ES_CONTAINER)..."
  docker rm -f "$ES_CONTAINER" 2>/dev/null || true
  docker run -d --rm --name "$ES_CONTAINER" \
    -p "$ES_PORT:9200" \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
    docker.io/elasticsearch:8.16.1 2>&1 | sed 's/^/  /'
  info "Waiting for ES to become healthy..."

  for i in $(seq 1 30); do
    if curl -sf "$ES_URL" > /dev/null 2>&1; then
      pass "Elasticsearch reachable at $ES_URL"
      break
    fi
    if [ "$i" -eq 30 ]; then
      fail "Elasticsearch not reachable after 30s"
      exit 1
    fi
    sleep 2
  done
  echo ""

  # ── 2. Start mock AI server ─────────────────────────────────
  echo "==> Starting mock AI server on :$MOCK_PORT..."
  "$RUN_DIR/bin/mock-ai-server" -addr ":$MOCK_PORT" &
  MOCK_PID=$!

  sleep 1
  if curl -sf http://localhost:$MOCK_PORT/health > /dev/null 2>&1; then
    pass "Mock AI server healthy"
  else
    fail "Mock AI server not responding"
  fi

  # Quick validation of mock endpoints
  EMBED_DIMS=$(curl -sf -X POST http://localhost:$MOCK_PORT/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{"input":"test","model":"mock-embedding"}' | \
    python3 -c "import json,sys; print(len(json.load(sys.stdin)['data'][0]['embedding']))")
  if [ "$EMBED_DIMS" = "1024" ]; then
    pass "Mock embeddings: 1024-dim vectors"
  else
    fail "Mock embeddings: expected 1024, got $EMBED_DIMS"
  fi
  echo ""

  # ── 3. Start Go MCP-RAG server ──────────────────────────────
  echo "==> Starting Go MCP-RAG server on :$SERVER_PORT..."
  "$RUN_DIR/bin/mcp-rag" serve --config "$RUN_DIR/config.yaml" &
  SERVER_PID=$!

  info "Waiting for server to be ready..."
  for i in $(seq 1 30); do
    if curl -sf http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
      pass "Go server healthy"
      break
    fi
    if [ "$i" -eq 30 ]; then
      fail "Go server not ready after 30s"
    fi
    sleep 1
  done
  echo ""

  # ── 4. API Tests ────────────────────────────────────────────
  echo "==> Running API tests..."
  echo ""

  # 4a. Health format
  HEALTH=$(curl -sf http://localhost:$SERVER_PORT/health)
  if echo "$HEALTH" | python3 -c "
import json,sys
h=json.load(sys.stdin)
assert h['status']=='healthy', f'status={h[\"status\"]}'
assert 'healthy' in h and 'ready' in h and 'bootstrapped' in h
assert 'runtime' in h and 'config_revision' in h and 'reasons' in h
print('OK')
" 2>&1; then
    pass "GET /health (Python-compatible format)"
  else
    fail "GET /health format incorrect"
  fi

  # 4b. Ready
  READY_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$SERVER_PORT/ready)
  if [ "$READY_CODE" = "200" ]; then
    pass "GET /ready returns 200"
  else
    fail "GET /ready returned $READY_CODE, expected 200"
  fi

  # 4c. Config flat format
  CONFIG=$(curl -sf http://localhost:$SERVER_PORT/config)
  if echo "$CONFIG" | python3 -c "
import json,sys
c=json.load(sys.stdin)
assert 'http_port' in c and 'provider_configs' in c
assert 'security' in c and 'rate_limit' in c and 'quotas' in c
assert 'config_revision' in c
print('OK')
" 2>&1; then
    pass "GET /config (flat format + provider_configs)"
  else
    fail "GET /config format incorrect"
  fi

  # 4d. KB list
  KB=$(curl -sf http://localhost:$SERVER_PORT/knowledge-bases)
  if echo "$KB" | python3 -c "import json,sys; assert 'knowledge_bases' in json.load(sys.stdin); print('OK')" 2>&1; then
    pass "GET /knowledge-bases"
  else
    fail "GET /knowledge-bases failed"
  fi

  # 4e. Add document
  ADD_RESP=$(curl -sf -X POST http://localhost:$SERVER_PORT/add-document \
    -H 'Content-Type: application/json' \
    -d '{"content":"Go语言是Google开发的静态类型编译型编程语言，以简洁高效著称，内置并发支持。RAG是检索增强生成技术，结合了信息检索和文本生成。"}')
  DOC_ID=$(echo "$ADD_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('document_id',''))")
  CHUNKS=$(echo "$ADD_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('chunk_count',0))")
  if [ -n "$DOC_ID" ] && [ "$CHUNKS" -gt 0 ]; then
    pass "POST /add-document (doc_id=$DOC_ID, chunks=$CHUNKS)"
  else
    fail "POST /add-document failed: $ADD_RESP"
  fi

  # 4f. Search (wait for ES refresh)
  sleep 2
  SEARCH=$(curl -sf "http://localhost:$SERVER_PORT/search?query=Go语言特点&limit=3")
  if echo "$SEARCH" | python3 -c "
import json,sys
s=json.load(sys.stdin)
assert s['query']=='Go语言特点'
assert len(s['results'])>0, 'no results'
r=s['results'][0]
assert 'vector_score' in r and 'retrieval_method' in r and 'metadata' in r
print('OK')
" 2>&1; then
    pass "GET /search (enriched results)"
  else
    fail "GET /search failed"
  fi

  # 4g. Chat
  CHAT=$(curl -sf -X POST http://localhost:$SERVER_PORT/chat \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"什么是RAG\",\"collection\":\"$ES_INDEX\",\"limit\":3}")
  if echo "$CHAT" | python3 -c "
import json,sys
c=json.load(sys.stdin)
assert c['query']=='什么是RAG'
assert 'response' in c and 'sources' in c and 'collection' in c
assert len(c['response'])>0
print('OK')
" 2>&1; then
    pass "POST /chat (with collection + sources)"
  else
    fail "POST /chat failed"
  fi

  # 4h. Provider models (stub-based)
  MODELS=$(curl -sf http://localhost:$SERVER_PORT/providers/mock/models)
  if echo "$MODELS" | python3 -c "
import json,sys
m=json.load(sys.stdin)
assert m['provider']=='mock'
assert len(m['models'])>0
print('OK')
" 2>&1; then
    pass "GET /providers/mock/models"
  else
    fail "GET /providers/mock/models failed"
  fi

  # 4i. SPA serving
  SPA_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$SERVER_PORT/app)
  if [ "$SPA_CODE" = "200" ]; then
    pass "GET /app returns 200 (SPA)"
  else
    fail "GET /app returned $SPA_CODE, expected 200"
  fi

  # 4j. Static files (follow redirects)
  STATIC_CODE=$(curl -s -o /dev/null -w "%{http_code}" -L http://localhost:$SERVER_PORT/static/app/index.html)
  if [ "$STATIC_CODE" = "200" ]; then
    pass "GET /static/app/index.html returns 200"
  else
    fail "GET /static/app/index.html returned $STATIC_CODE, expected 200"
  fi

  # 4k. Docs page
  DOCS_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$SERVER_PORT/docs)
  if [ "$DOCS_CODE" = "200" ]; then
    pass "GET /docs returns 200"
  else
    fail "GET /docs returned $DOCS_CODE, expected 200"
  fi

  # 4l. OpenAPI spec
  OAPI_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$SERVER_PORT/openapi.json)
  if [ "$OAPI_CODE" = "200" ]; then
    pass "GET /openapi.json returns 200"
  else
    fail "GET /openapi.json returned $OAPI_CODE, expected 200"
  fi

  # 4m. Redirects
  for pair in "/:302" "/doc:302" "/documents-page:302" "/config-page:302"; do
    IFS=':' read -r path code <<< "$pair"
    actual_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$SERVER_PORT$path")
    if [ "$actual_code" = "$code" ]; then
      pass "GET $path -> $code redirect"
    else
      fail "GET $path returned $actual_code, expected $code"
    fi
  done

  # 4n. Upload files (multipart)
  echo "test content for e2e upload" > "$RUN_DIR/test-upload.txt"
  UPLOAD_RESP=$(curl -sf -X POST http://localhost:$SERVER_PORT/upload-files \
    -F "files=@$RUN_DIR/test-upload.txt")
  if echo "$UPLOAD_RESP" | python3 -c "
import json,sys
u=json.load(sys.stdin)
assert u['total_files']==1, f'total_files={u[\"total_files\"]}'
assert u['successful']==1, f'successful={u[\"successful\"]}'
assert len(u['results'])==1
assert u['results'][0]['processed']==True
assert u['results'][0]['filename']=='test-upload.txt'
print('OK')
" 2>&1; then
    pass "POST /upload-files (multipart file upload)"
  else
    fail "POST /upload-files failed: $UPLOAD_RESP"
  fi

  # 4o. Create knowledge base (KB CRUD)
  KB_CREATE=$(curl -sf -X POST http://localhost:$SERVER_PORT/knowledge-bases \
    -H 'Content-Type: application/json' \
    -d '{"name":"e2e_public_kb","scope":"public"}')
  if echo "$KB_CREATE" | python3 -c "
import json,sys
k=json.load(sys.stdin)
assert k['name']=='e2e_public_kb', f'name={k.get(\"name\")}'
assert k['scope']=='public', f'scope={k.get(\"scope\")}'
assert 'id' in k
assert 'collection_name' in k
print('OK')
" 2>&1; then
    pass "POST /knowledge-bases (create KB)"
  else
    fail "POST /knowledge-bases failed: $KB_CREATE"
  fi

  # 4p. MCP list tools
  MCP_TOOLS=$(curl -sf http://localhost:$SERVER_PORT/debug/mcp/tools)
  if echo "$MCP_TOOLS" | python3 -c "
import json,sys
m=json.load(sys.stdin)
assert 'tools' in m
tools=[t['name'] for t in m['tools']]
assert 'rag_ask' in tools, f'rag_ask not in {tools}'
print('OK')
" 2>&1; then
    pass "GET /debug/mcp/tools (rag_ask tool registered)"
  else
    fail "GET /debug/mcp/tools failed: $MCP_TOOLS"
  fi

  # 4q. MCP call tool (rag_ask raw mode)
  sleep 1  # wait for ES refresh after upload
  MCP_CALL=$(curl -sf -X POST http://localhost:$SERVER_PORT/debug/mcp/call \
    -H 'Content-Type: application/json' \
    -d '{"tool":"rag_ask","arguments":{"query":"Go语言","mode":"raw","limit":3}}')
  if echo "$MCP_CALL" | python3 -c "
import json,sys
m=json.load(sys.stdin)
assert m['tool']=='rag_ask'
assert m['is_error']==False
assert len(m['content'])>0
print('OK')
" 2>&1; then
    pass "POST /debug/mcp/call (rag_ask raw mode)"
  else
    fail "POST /debug/mcp/call failed: $MCP_CALL"
  fi

  # 4r. Config bulk update with SPA wrapper
  BULK_RESP=$(curl -sf -X POST http://localhost:$SERVER_PORT/config/bulk \
    -H 'Content-Type: application/json' \
    -d '{"updates":{"chunk_size":5000}}')
  if echo "$BULK_RESP" | python3 -c "import json,sys; assert json.load(sys.stdin)['status']=='updated'; print('OK')" 2>&1; then
    pass "POST /config/bulk (SPA {\"updates\":{...}} wrapper)"
  else
    fail "POST /config/bulk failed"
  fi

  # ── Auth tests (security_enabled, allow_anonymous) ──────
  AUTH_KEY="sk-e2e-test-key-123"
  AUTH_BAD="sk-bad-key"

  # 4s. Valid API key via X-API-Key header
  AUTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "X-API-Key: $AUTH_KEY" \
    http://localhost:$SERVER_PORT/search?query=test)
  if [ "$AUTH_CODE" = "200" ]; then
    pass "Auth: valid X-API-Key → 200"
  else
    fail "Auth: valid key returned $AUTH_CODE, expected 200"
  fi

  # 4t. Valid API key via Authorization: Bearer header
  AUTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $AUTH_KEY" \
    http://localhost:$SERVER_PORT/search?query=test)
  if [ "$AUTH_CODE" = "200" ]; then
    pass "Auth: valid Bearer token → 200"
  else
    fail "Auth: valid Bearer returned $AUTH_CODE, expected 200"
  fi

  # 4u. Valid API key via query param
  AUTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    "http://localhost:$SERVER_PORT/search?query=test&api_key=$AUTH_KEY")
  if [ "$AUTH_CODE" = "200" ]; then
    pass "Auth: valid api_key query param → 200"
  else
    fail "Auth: valid query param returned $AUTH_CODE, expected 200"
  fi

  # 4v. Invalid API key → 403
  AUTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "X-API-Key: $AUTH_BAD" \
    http://localhost:$SERVER_PORT/search?query=test)
  AUTH_BODY=$(curl -s -H "X-API-Key: $AUTH_BAD" http://localhost:$SERVER_PORT/search?query=test)
  if [ "$AUTH_CODE" = "403" ]; then
    pass "Auth: invalid API key → 403"
  else
    fail "Auth: invalid key returned $AUTH_CODE (body: $AUTH_BODY), expected 403"
  fi

  # 4w. No API key → 200 (anonymous allowed)
  AUTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    http://localhost:$SERVER_PORT/search?query=test)
  if [ "$AUTH_CODE" = "200" ]; then
    pass "Auth: no API key → 200 (anonymous allowed)"
  else
    fail "Auth: no key returned $AUTH_CODE, expected 200"
  fi

  # 4x. Rate limiter: burst requests within window should not 429
  RATE_OK=true
  for i in $(seq 1 5); do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$SERVER_PORT/health)
    if [ "$CODE" != "200" ]; then
      RATE_OK=false
      break
    fi
  done
  if $RATE_OK; then
    pass "Rate limiter: 5 rapid health checks → all 200"
  else
    fail "Rate limiter: rapid requests got throttled unexpectedly"
  fi

  echo ""
  echo -e "  ${GREEN}API tests: $PASS passed${NC}"

  # ── 5. Playwright browser tests ──────────────────────────────
  echo ""
  do_playwright

  # ── 6. Summary ───────────────────────────────────────────────
  echo ""
  echo "══════════════════════════════════════════════"
  if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${GREEN}All e2e tests passed! ($PASS tests)${NC}"
  else
    echo -e "  ${RED}$FAIL test(s) failed, $PASS passed${NC}"
  fi
  echo "══════════════════════════════════════════════"
}

# ──────────────────────────────────────────────────────────────
# Playwright phase
# ──────────────────────────────────────────────────────────────
do_playwright() {
  echo "==> Running Playwright browser tests..."
  echo ""

  # Check if server is running
  if ! curl -sf http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
    fail "Playwright: server not running, cannot test"
    return
  fi

  local PLAYWRIGHT_SCRIPT="$SCRIPT_DIR/playwright/spa-tests.js"
  if [ ! -f "$PLAYWRIGHT_SCRIPT" ]; then
    fail "Playwright: spa-tests.js not found at $PLAYWRIGHT_SCRIPT"
    return
  fi

  # Run playwright test script (use temp file to avoid subshell counter issue)
  local PW_OUT="$RUN_DIR/playwright-output.txt"
  NODE_PATH="$(npm root -g 2>/dev/null || echo '')" \
  BASE_URL="http://localhost:$SERVER_PORT" \
  node "$PLAYWRIGHT_SCRIPT" > "$PW_OUT" 2>&1 || true

  while IFS= read -r line; do
    if [[ "$line" == PASS:* ]]; then
      pass "${line#PASS: }"
    elif [[ "$line" == FAIL:* ]]; then
      fail "${line#FAIL: }"
    else
      echo "  $line"
    fi
  done < "$PW_OUT"
}

# ──────────────────────────────────────────────────────────────
# Cleanup phase
# ──────────────────────────────────────────────────────────────
do_clean() {
  echo "==> Cleaning up..."
  cd "$PROJECT_DIR"

  # Stop e2e ES container
  docker stop "$ES_CONTAINER" 2>/dev/null || true
  info "Stopped Elasticsearch ($ES_CONTAINER)"

  # Kill any running test processes
  pkill -f "mock-ai-server" 2>/dev/null || true
  pkill -f "mcp-rag serve" 2>/dev/null || true
  info "Stopped test processes"

  # Remove run directory
  if [ -d "$RUN_DIR" ]; then
    rm -rf "$RUN_DIR"
    info "Removed $RUN_DIR"
  fi
  echo "Done."
}

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
case "${1:-all}" in
  setup)
    do_setup
    ;;
  test)
    # Trap cleanup on exit
    trap 'kill $SERVER_PID 2>/dev/null; kill $MOCK_PID 2>/dev/null; docker stop $ES_CONTAINER 2>/dev/null; true' EXIT
    do_test
    ;;
  playwright)
    do_playwright
    ;;
  clean)
    do_clean
    ;;
  all)
    # Trap cleanup on exit
    MOCK_PID=""
    SERVER_PID=""
    trap 'kill $SERVER_PID 2>/dev/null; kill $MOCK_PID 2>/dev/null; docker stop $ES_CONTAINER 2>/dev/null; true' EXIT
    do_setup
    do_test
    ;;
  *)
    echo "Usage: $0 {setup|test|playwright|clean|all}"
    exit 1
    ;;
esac

exit $FAIL
