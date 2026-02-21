# TRP1 Interim Report — Week 1

Author: Zumi
Branch: trp1/interim

## Executive Summary

This interim draft documents the work completed to implement an intent-driven Hook Engine for the Roo Code VS Code extension. It contains design notes, the code architecture, hook decisions, and diagrams. The code scaffolding lives in `src/hooks/` and runtime artifacts are placed under `.orchestration/`.

## How the VS Code extension works

- The extension runs in the VS Code Extension Host. Agent requests (tool calls) are proxied through a Hook Engine in the extension host.
- All tool invocations (read or mutating) are classified and first pass through registered Pre-Hooks. Pre-Hooks can validate, enrich, or block the call.
- After the tool runs, Post-Hooks run to process results (e.g., compute content hashes and append to `agent_trace.jsonl`).
- The LLM system prompt is modified to require agents to call `select_active_intent(intent_id)` as their first action.

## Code & Design Architecture (from `ARCHITECTURE_NOTES.md`)

- Files added:

    - `.orchestration/active_intents.yaml` — intent registry.
    - `.orchestration/agent_trace.jsonl` — append-only trace ledger.
    - `src/hooks/prepost_hooks.ts` — HookEngine, Pre/Post hook implementations, YAML parsing.
    - `src/hooks/tool_registry.ts` — tool definitions and helpers.

- HookEngine (core): a small, typed engine that allows registration of pre/post hooks by tool name. The host should instantiate `HookEngine`, call `registerDefaultTools(hooks)`, and proxy tool calls through `callTool(hooks, tool, workspaceRoot)` so hooks run consistently.

## Architectural Decisions for the Hook

- Intent-first policy: Enforce `select_active_intent` as mandatory. Pre-Hook validates intent and returns a structured `intent_context` object with `id`, `constraints`, and `owned_scope`.
- Strong typing & interfaces: `PreHook<Input,Output>` and `PostHook<Input,Output>` types formalize middleware behavior and make testing easier.
- Structured YAML parsing: prefer `yaml` package; implement a small fallback parser for constrained environments.
- Spatial independence & hashing: Post-Hook compute SHA-256 of mutated content blocks and record `sha256:` prefixed hashes in `agent_trace.jsonl`.
- Optimistic concurrency: before a write, compare the agent's read-hash to on-disk hash (left as next implementation step) and block stale writes.

Enhancements implemented to harden middleware and traceability:

- Safe hook execution: hooks execute inside try/catch; pre-hooks return structured error objects (not exceptions) so the host can respond programmatically. Post-hook failures are logged to `.orchestration/hook_errors.log`.
- Active Intent Gatekeeper: `select_active_intent` persists an active intent to `.orchestration/active_intent_current.json`. The Hook Engine enforces that all mutating tools (write/delete/exec) are blocked unless an active intent with `status: IN_PROGRESS` is present.
- Trace enrichment: `postHook_WriteFile` now records `mutation_class` (heuristic between `AST_REFACTOR` and `INTENT_EVOLUTION`) and captures the current git SHA for `vcs.revision_id`. Related fields are populated from the active intent selection when available.

## Hook System Diagram

```mermaid
flowchart LR
  A[Agent/LLM] -->|callTool(select_active_intent)| B(Hook Engine PreHook)
  B --> C{valid intent?}
  C -- yes --> D[Return intent_context]
  C -- no --> E[Block & error]
  D --> F[Agent does work (LLM) with injected context]
  F --> G[write_file tool call]
  G --> H(Hook Engine PreHook for write)
  H --> I[Host performs write]
  I --> J(PostHook_WriteFile computes hash & appends trace)
  J --> K[.orchestration/agent_trace.jsonl]
```

## Schemas

- `active_intents.yaml` (example):

```yaml
active_intents:
    - id: "INT-001"
      name: "Build Weather API"
      status: "IN_PROGRESS"
      owned_scope:
          - "src/weather/**"
      constraints:
          - "No external paid APIs"
      acceptance_criteria:
          - "Unit tests in tests/weather/ pass"
```

- `agent_trace.jsonl` (entry example):

```json
{
	"id": "trace-161803398",
	"timestamp": "2026-02-18T12:00:00Z",
	"vcs": { "revision_id": "git-sha" },
	"files": [
		{
			"relative_path": "src/weather/api.ts",
			"conversations": [
				{
					"contributor": { "entity_type": "AI", "model_identifier": "example-model" },
					"ranges": [{ "start_line": 1, "end_line": 42, "content_hash": "sha256:..." }],
					"related": [{ "type": "specification", "value": "REQ-001" }]
				}
			]
		}
	]
}
```

## Implementation Notes and Next Steps

- Completed: scaffolding for HookEngine, structured parsing, sample intent file, trace ledger placeholder, tool registry.
- Next (recommended priorities):
    1. Wire `HookEngine` into Roo Code's extension host: proxy actual `execute_command` and `write_file` handlers through `callTool`.
    2. Implement optimistic concurrency check (read-hash vs on-disk hash).
    3. Add unit tests for `loadActiveIntent` and `preHook_SelectActiveIntent`.
    4. Implement system prompt update inside the LLM prompt builder so agents are forced to call the intent tool.

## How to generate the PDF (local)

You can convert this markdown into a PDF using `pandoc`:

```bash
cd Roo-Code
scripts/build_interim_pdf.sh
```

`build_interim_pdf.sh` runs `pandoc INTERIM_REPORT.md -o INTERIM_REPORT.pdf --pdf-engine=xelatex` (requires `pandoc` and a TeX engine).

---

Appendix: `ARCHITECTURE_NOTES.md` was used as the source for the design section.
