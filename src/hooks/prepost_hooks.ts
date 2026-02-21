// Enhanced Pre/Post-Hook scaffolding for Roo Code (TypeScript)

import * as crypto from "crypto"
import * as fs from "fs"
import * as path from "path"
import { execSync } from "child_process"

// Prefer a proper YAML parser if available (used elsewhere); a simple fallback can be implemented by the host.
let yamlParse: (s: string) => any
try {
	const yaml = require("yaml")
	yamlParse = (s: string) => yaml.parse(s)
} catch (e) {
	yamlParse = (s: string) => ({})
}

export interface Intent {
	id: string
	name: string
	status: string
	owned_scope?: string[]
	constraints?: string[]
	acceptance_criteria?: string[]
}

export function computeSha256(content: string): string {
	return crypto.createHash("sha256").update(content, "utf8").digest("hex")
}

export function appendAgentTrace(workspaceRoot: string, traceObj: object) {
	const file = path.join(workspaceRoot, ".orchestration", "agent_trace.jsonl")
	try {
		fs.mkdirSync(path.dirname(file), { recursive: true })
		const line = JSON.stringify(traceObj) + "\n"
		fs.appendFileSync(file, line, "utf8")
	} catch (e) {
		// best effort; avoid crashing host
		console.error("appendAgentTrace failed", e)
	}
}

// Hook engine with safe execution (structured errors)
export type PreHook<Input, Output> = (input: Input, workspaceRoot: string) => Promise<Output> | Output
export type PostHook<Input, Output> = (input: Input, result: Output, workspaceRoot: string) => Promise<void> | void

export class HookEngine {
	private preHooks: Map<string, Array<PreHook<any, any>>> = new Map()
	private postHooks: Map<string, Array<PostHook<any, any>>> = new Map()

	registerPreHook<TIn, TOut>(toolName: string, fn: PreHook<TIn, TOut>) {
		this.preHooks.set(toolName, (this.preHooks.get(toolName) || []).concat(fn))
	}

	registerPostHook<TIn, TOut>(toolName: string, fn: PostHook<TIn, TOut>) {
		this.postHooks.set(toolName, (this.postHooks.get(toolName) || []).concat(fn))
	}

	// run pre-hooks and return either last hook value or a structured error object { __hook_error: true, message }
	async runPreHooks(toolName: string, input: any, workspaceRoot: string): Promise<any> {
		const hooks = this.preHooks.get(toolName) || []
		let last: any = null
		for (const h of hooks) {
			try {
				last = await h(input, workspaceRoot)
			} catch (err: any) {
				return { __hook_error: true, message: String(err?.message || err), tool: toolName }
			}
		}
		return last
	}

	// run post-hooks and log errors to .orchestration/hook_errors.log (best-effort)
	async runPostHooks(toolName: string, input: any, result: any, workspaceRoot: string) {
		const hooks = this.postHooks.get(toolName) || []
		for (const h of hooks) {
			try {
				await h(input, result, workspaceRoot)
			} catch (err: any) {
				try {
					const logFile = path.join(workspaceRoot, ".orchestration", "hook_errors.log")
					fs.mkdirSync(path.dirname(logFile), { recursive: true })
					const line =
						JSON.stringify({
							tool: toolName,
							error: String(err?.message || err),
							timestamp: new Date().toISOString(),
						}) + "\n"
					fs.appendFileSync(logFile, line, "utf8")
				} catch (e) {
					// ignore logging failures
				}
			}
		}
	}
}

// Active intent management (simple file-backed gatekeeper)
const ACTIVE_INTENT_FILE = ".orchestration/active_intent_current.json"

export function setActiveIntent(intent: Intent, workspaceRoot: string) {
	const file = path.join(workspaceRoot, ACTIVE_INTENT_FILE)
	try {
		fs.mkdirSync(path.dirname(file), { recursive: true })
		fs.writeFileSync(file, JSON.stringify({ selected: intent, timestamp: new Date().toISOString() }), "utf8")
		return true
	} catch (e) {
		return false
	}
}

export function getActiveIntent(workspaceRoot: string): Intent | null {
	const file = path.join(workspaceRoot, ACTIVE_INTENT_FILE)
	if (!fs.existsSync(file)) return null
	try {
		const content = fs.readFileSync(file, "utf8")
		const parsed = JSON.parse(content)
		return parsed?.selected || null
	} catch (e) {
		return null
	}
}

function computeMutationClass(oldContent: string | null, newContent: string): string {
	if (!oldContent) return "INTENT_EVOLUTION"
	const oldLines = oldContent.split(/\r?\n/).length
	const newLines = newContent.split(/\r?\n/).length
	const delta = Math.abs(newLines - oldLines)
	const ratio = delta / Math.max(1, oldLines)
	if (ratio < 0.2) return "AST_REFACTOR"
	return "INTENT_EVOLUTION"
}

function getGitSha(workspaceRoot: string): string | null {
	try {
		return String(execSync("git rev-parse HEAD", { cwd: workspaceRoot })).trim()
	} catch (e) {
		return null
	}
}

// Exported hooks (to be wired into the extension host)
export async function preHook_SelectActiveIntent(intentId: string, workspaceRoot: string) {
	// Load the declared intents and validate
	const intentsFile = path.join(workspaceRoot, ".orchestration", "active_intents.yaml")
	if (!fs.existsSync(intentsFile)) throw new Error("active_intents.yaml not found")
	const content = fs.readFileSync(intentsFile, "utf8")
	const parsed = yamlParse(content) || {}
	const list = parsed.active_intents || []
	const found = list.find((a: any) => String(a.id) === String(intentId))
	if (!found) throw new Error("You must cite a valid active Intent ID.")
	const intent: Intent = {
		id: String(found.id),
		name: found.name || "unknown",
		status: found.status || "UNKNOWN",
		owned_scope: found.owned_scope || [],
		constraints: found.constraints || [],
		acceptance_criteria: found.acceptance_criteria || [],
	}
	// Persist active intent selection (gatekeeper)
	setActiveIntent(intent, workspaceRoot)
	// Return a structured intent_context object for injection into prompts
	return {
		intent_context: {
			id: intent.id,
			name: intent.name,
			constraints: intent.constraints || [],
			owned_scope: intent.owned_scope || [],
		},
	}
}

export async function postHook_WriteFile(relativePath: string, newContent: string, workspaceRoot: string, meta: any) {
	const filePath = path.join(workspaceRoot, relativePath)
	let oldContent: string | null = null
	try {
		if (fs.existsSync(filePath)) oldContent = fs.readFileSync(filePath, "utf8")
	} catch (e) {
		oldContent = null
	}
	const contentHash = computeSha256(newContent)
	const mutation_class = computeMutationClass(oldContent, newContent)
	const vcs_sha = getGitSha(workspaceRoot)
	const active = getActiveIntent(workspaceRoot)
	const related = [] as any[]
	if (active) related.push({ type: "intent", value: active.id })

	const trace = {
		id: meta?.id || `trace-${Date.now()}`,
		timestamp: new Date().toISOString(),
		vcs: { revision_id: vcs_sha },
		mutation_class,
		files: [
			{
				relative_path: relativePath,
				conversations: [
					{
						url: meta?.url || null,
						contributor: meta?.contributor || { entity_type: "AI", model_identifier: "example-model" },
						ranges: [
							{
								start_line: meta?.start_line || 1,
								end_line: meta?.end_line || 1,
								content_hash: `sha256:${contentHash}`,
							},
						],
						related: meta?.related || related,
					},
				],
			},
		],
	}
	appendAgentTrace(workspaceRoot, trace)
	return trace
}
