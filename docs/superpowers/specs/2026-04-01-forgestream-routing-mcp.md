# ForgeStream Routing MCP — Design Document

**Date:** 2026-04-01
**Status:** Draft — awaiting approval before implementation
**Author:** Claude (Phase 3 of sentinel security/optimization work)

## Problem

ForgeStream's MCP server (`forgestream/mcp_server.py`) exposes all 6 tools to every caller with no access control. Sentinel modes, ForgeStream dispatch agents, and external projects all get the same full tool set. This creates:

1. **Over-privileged access**: Sentinel read-only modes can call write tools
2. **No observability**: No Braintrust tracing on MCP tool calls (only shell hooks for Claude Code sessions)
3. **No caller identification**: Can't distinguish sentinel vs dispatcher vs external callers

## Design: Lightweight FastMCP Proxy

A thin proxy layer that sits between callers and the existing ForgeStream MCP server. NOT a full gateway — tool-based routing with env-var scope resolution.

### Architecture

```
Caller (sentinel mode / dispatcher agent / external project)
  │
  │  env: SENTINEL_MODE=orange_team
  │       FORGESTREAM_AGENT_TYPE=research
  │       FORGESTREAM_CALLER_PROJECT=/Users/.../college-baseball
  │
  ▼
┌──────────────────────────────────────┐
│  ForgeStream Routing MCP (proxy)     │
│                                      │
│  1. Read env vars → resolve scope    │
│  2. Filter tools by scope            │
│  3. Wrap each call with Braintrust   │
│     span (tool name, caller, scope)  │
│  4. Forward to real MCP server       │
└──────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────┐
│  ForgeStream MCP Server (existing)   │
│  - forgestream_query_knowledge       │
│  - forgestream_get_requirements      │
│  - forgestream_get_seeds             │
│  - forgestream_get_contradictions    │
│  - forgestream_search_claims         │
│  - forgestream_get_expert            │
└──────────────────────────────────────┘
```

### Scope Model

| Scope ID | Env Signal | Allowed Tools | Rationale |
|----------|-----------|---------------|-----------|
| `sentinel:read` | `SENTINEL_MODE` set, no `SENTINEL_INJECT=true` | `query_knowledge`, `search_claims`, `get_expert` | Sentinel modes doing read-only analysis |
| `sentinel:inject` | `SENTINEL_MODE` set + `SENTINEL_INJECT=true` | `query_knowledge`, `search_claims`, `get_expert` + future `store_insight` | Sentinel modes that write findings |
| `dispatcher:research` | `FORGESTREAM_AGENT_TYPE=research` | `query_knowledge`, `search_claims`, `get_expert` | Research agents — read-only |
| `dispatcher:scaffold` | `FORGESTREAM_AGENT_TYPE=scaffold` | ALL 6 tools | Scaffold agents — full access |
| `project:external` | `FORGESTREAM_CALLER_PROJECT` set, no other signals | `query_knowledge` only | External projects (stone-and-co, college-baseball) |

**Resolution order:** Check env vars in priority: `SENTINEL_MODE` > `FORGESTREAM_AGENT_TYPE` > `FORGESTREAM_CALLER_PROJECT` > default (full access for backward compat).

### Braintrust Span Injection

Every tool call gets wrapped in a Braintrust span:

```python
from braintrust import traced

@traced(name="forgestream_mcp_proxy")
def proxy_tool_call(tool_name: str, args: dict, scope: str, caller: str):
    span = braintrust.current_span()
    span.log(
        metadata={
            "tool": tool_name,
            "scope": scope,
            "caller": caller,
            "project": os.environ.get("FORGESTREAM_CALLER_PROJECT", ""),
        }
    )
    result = backend_server.call_tool(tool_name, args)
    span.log(output={"result_length": len(str(result))})
    return result
```

This gives us:
- Per-tool latency tracking
- Scope violation attempts (logged as errors)
- Cross-project usage patterns
- Token cost attribution per caller type

### ToolAggregator Pattern

```python
from mcp.server.fastmcp import FastMCP

SCOPE_TOOLS: dict[str, set[str]] = {
    "sentinel:read": {"forgestream_query_knowledge", "forgestream_search_claims", "forgestream_get_expert"},
    "sentinel:inject": {"forgestream_query_knowledge", "forgestream_search_claims", "forgestream_get_expert"},
    "dispatcher:research": {"forgestream_query_knowledge", "forgestream_search_claims", "forgestream_get_expert"},
    "dispatcher:scaffold": None,  # None = all tools
    "project:external": {"forgestream_query_knowledge"},
}

def resolve_scope() -> str:
    if os.environ.get("SENTINEL_MODE"):
        return "sentinel:inject" if os.environ.get("SENTINEL_INJECT") == "true" else "sentinel:read"
    if agent_type := os.environ.get("FORGESTREAM_AGENT_TYPE"):
        return f"dispatcher:{agent_type}"
    if os.environ.get("FORGESTREAM_CALLER_PROJECT"):
        return "project:external"
    return "dispatcher:scaffold"  # backward compat default

def create_proxy_server() -> FastMCP:
    proxy = FastMCP("forgestream-proxy")
    backend = ForgeStreamMCPServer()
    scope = resolve_scope()
    allowed = SCOPE_TOOLS.get(scope)

    # Register only allowed tools as proxy endpoints
    for tool_name in backend.list_tools():
        if allowed is not None and tool_name not in allowed:
            continue
        # Create proxy function that wraps with Braintrust span
        register_proxy_tool(proxy, backend, tool_name, scope)

    return proxy
```

### File Layout

```
forgestream/
  mcp_server.py          # Existing — untouched
  mcp_proxy.py           # NEW — routing proxy (< 150 lines)
  __main__.py            # Updated — `mcp` subcommand routes to proxy
```

### Configuration

In `settings.json` or MCP plugin config, callers point to the proxy:

```json
{
  "command": "python",
  "args": ["-m", "forgestream", "mcp"],
  "env": {
    "SENTINEL_MODE": "orange_team"
  }
}
```

The proxy reads env at startup, resolves scope once, and only exposes the filtered tool set.

### Testing Plan

1. **Unit tests**: Scope resolution for each env combination
2. **Integration test**: Call proxy with `SENTINEL_MODE=orange_team`, verify only 3 tools visible
3. **Braintrust verification**: Check spans appear in the project dashboard
4. **Backward compat**: No env vars set → all tools available (existing behavior preserved)

### Dependencies

- `mcp>=1.0` — already available system-wide, should be declared in `pyproject.toml`
- `braintrust` — for span injection (optional, graceful degradation if missing)

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Env vars not set → wrong scope | Default to full access (backward compat), log warning |
| Braintrust unavailable | Graceful degradation — proxy works without tracing |
| New tools added to backend | Proxy forwards unknown tools only if scope allows all |
| Performance overhead | Negligible — env var check + one function call wrapper |

## Decision Required

Approve this design to proceed with implementation, or request changes.
