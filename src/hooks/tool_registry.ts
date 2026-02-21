import { HookEngine, preHook_SelectActiveIntent, getActiveIntent } from "./prepost_hooks"
import * as path from "path"

export const TOOL_SELECT_ACTIVE_INTENT = "select_active_intent"

export function registerDefaultTools(hooks: HookEngine) {
	hooks.registerPreHook<string, any>(TOOL_SELECT_ACTIVE_INTENT, async (intentId, workspaceRoot) => {
		return await preHook_SelectActiveIntent(intentId, workspaceRoot)
	})
}

export type ToolCall = { name: string; args: any[] }

function isMutatingTool(name: string) {
	const p = name.toLowerCase()
	return p.includes("write") || p.includes("delete") || p.includes("exec") || p.includes("run")
}

// callTool proxies tool invocations through HookEngine. It enforces the active-intent gatekeeper for mutating tools.
export async function callTool(hooks: HookEngine, tool: ToolCall, workspaceRoot: string) {
	// Run pre-hooks (safe)
	const preResult = await hooks.runPreHooks(tool.name, tool.args[0], workspaceRoot)
	if (preResult && preResult.__hook_error) {
		// Return structured error to caller
		return { ok: false, error: preResult }
	}

	// Gatekeeper: block mutating tools when no active intent is selected
	if (isMutatingTool(tool.name)) {
		const active = getActiveIntent(workspaceRoot)
		if (!active) {
			return {
				ok: false,
				error: {
					message: "Scope Violation: no active intent selected. Call select_active_intent(intent_id) first.",
				},
			}
		}
		// Respect intent status: block if intent is not IN_PROGRESS
		if (active.status && active.status !== "IN_PROGRESS") {
			return {
				ok: false,
				error: { message: `Intent status '${active.status}' does not allow mutating operations.` },
			}
		}
	}

	// NOTE: actual tool execution should occur here. For select_active_intent, the pre-hook result is the context.
	let toolResult = preResult

	// Run post-hooks if any, best-effort
	await hooks.runPostHooks(tool.name, tool.args[0], toolResult, workspaceRoot)
	return { ok: true, result: toolResult }
}
