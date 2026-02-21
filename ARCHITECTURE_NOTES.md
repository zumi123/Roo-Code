# ARCHITECTURE_NOTES

Phase 0 notes: locating hook points and intent flow for Roo Code.

1. Hook Engine placement
    - Implement middleware in the Extension Host where `execute_command` and `write_file` are proxied.
2. Tools to add
    - `select_active_intent(intent_id: string)` - required first action for any mutating workflow.
3. Pre-Hook responsibilities
    - Validate `intent_id` against `.orchestration/active_intents.yaml`.
    - Return an `<intent_context>` containing constraints and scope.
4. Post-Hook responsibilities
    - Compute SHA-256 of modified content ranges and append trace to `.orchestration/agent_trace.jsonl`.
5. Concurrency
    - Compare read-hash vs on-disk hash; block on mismatch.

Next: implement `src/hooks/` scaffolding and wire a minimal pre/post hook.

Hardenings to implement next (done in scaffolding):

- Safe hook execution: Pre/Post hooks are executed inside try/catch; pre-hooks return structured error objects instead of throwing; post-hook errors are logged to `.orchestration/hook_errors.log`.
- Gatekeeper: `select_active_intent` persists an active intent selection to `.orchestration/active_intent_current.json`. Mutating tools are blocked unless an active intent is present and `status` is `IN_PROGRESS`.
- Trace enrichments: `postHook_WriteFile` now records `mutation_class` (heuristic AST_REFACTOR vs INTENT_EVOLUTION) and captures current git SHA in the trace's `vcs.revision_id`.
