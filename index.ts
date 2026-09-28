/**
 * /btw - ask a quick side question without interrupting the main conversation.
 *
 * Port of Claude Code's `src/commands/btw` + `src/utils/sideQuestion.ts`.
 *
 * How it works:
 * - Extension commands in pi are dispatched *before* the streaming check
 *   (`core/agent-session.js` `prompt()`), so `/btw` runs immediately even while
 *   the main agent is mid-turn. Nothing is queued, nothing is aborted.
 * - The side question reuses the exact system prompt, message prefix and tool
 *   declarations the main thread is already sending, so the request hits the
 *   provider's prompt cache instead of re-paying for the whole conversation.
 * - The answer is rendered in an overlay and is never appended to the session,
 *   so it costs zero main-context tokens.
 * - The answer is shown as it streams. Esc/Enter/Space dismiss it and abort an
 *   in-flight request; Up/Down (or Ctrl+P/Ctrl+N) scroll long answers.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { uuidv7 } from "@earendil-works/pi-ai";
import type { AssistantMessage, Message, ThinkingLevel, Tool } from "@earendil-works/pi-ai";
import {
	convertToLlm,
	type ExtensionAPI,
	type ExtensionCommandContext,
	getAgentDir,
	getMarkdownTheme,
} from "@earendil-works/pi-coding-agent";
import { Loader, Markdown, matchesKey, Text, visibleWidth } from "@earendil-works/pi-tui";
import type { Component, TUI } from "@earendil-works/pi-tui";
import type { Theme } from "@earendil-works/pi-coding-agent";

/** Lines the Up/Down keys move at once. */
const SCROLL_LINES = 3;

/** Keep the overlay readable without letting it swallow the whole screen. */
const MAX_OVERLAY_ROWS = 24;

/** Cap the answer so a runaway model cannot bill an unbounded side question. */
const MAX_TOKENS = 4096;

/** Number of previous questions offered as `/btw` argument completions. */
const HISTORY_LIMIT = 20;
const HISTORY_FILE = "btw-history.json";

const SIDE_QUESTION_PROMPT = `<system-reminder>This is a side question from the user. Answer it directly, in a single response.

CONTEXT:
- You are a short-lived agent spawned to answer exactly one question
- The main agent is NOT interrupted and is NOT waiting for you; it keeps working independently
- You share the conversation context above, but you are a separate instance
- Do not talk about being interrupted or about "what you were doing before"

HARD CONSTRAINTS:
- You have NO tools. You cannot read files, run commands, or take any action
- There is no follow-up turn, so never say "let me check", "I'll look", "one moment"
- Answer only from the conversation context you already have
- If you do not know, say so plainly instead of offering to find out
- Keep it short: a few sentences unless the question genuinely needs more</system-reminder>`;

export default function btwExtension(pi: ExtensionAPI) {
	const history = loadHistory();

	pi.registerCommand("btw", {
		description: "Ask a quick side question without interrupting the main conversation",
		getArgumentCompletions: (prefix) => {
			const lower = prefix.toLowerCase();
			return history
				.filter((q) => q.toLowerCase().startsWith(lower))
				.map((q) => ({ value: q, label: q }));
		},
		handler: async (args, ctx) => {
			const question = await resolveQuestion(args, ctx, history);
			if (!question) {
				if (ctx.hasUI) ctx.ui.notify("Usage: /btw <your question>", "warning");
				return;
			}

			remember(history, question);
			await runSideQuestion(pi, ctx, question);
		},
	});
}

/** Args win; otherwise ask. An empty answer from the dialog means "never mind". */
async function resolveQuestion(
	args: string,
	ctx: ExtensionCommandContext,
	history: string[],
): Promise<string | undefined> {
	const fromArgs = args.trim();
	if (fromArgs) return fromArgs;
	if (!ctx.hasUI) return undefined;
	const typed = await ctx.ui.input("Side question", "ask anything about the current conversation");
	const question = typed?.trim();
	if (question) return question;
	// Nothing typed: offer the last questions instead of doing nothing.
	if (history.length > 0) {
		const picked = await ctx.ui.select("Recent side questions", history.slice(0, 5));
		return picked?.trim() || undefined;
	}
	return undefined;
}

async function runSideQuestion(pi: ExtensionAPI, ctx: ExtensionCommandContext, question: string) {
	const model = ctx.model;
	if (!model) {
		report(ctx, "No model selected. Pick one with /model first.");
		return;
	}

	if (!ctx.hasUI || ctx.mode !== "tui") {
		await runHeadless(pi, ctx, question);
		return;
	}

	await ctx.ui.custom<undefined>(
		(tui, theme, _keybindings, done) => {
			const view = new BtwView(tui, theme, question, () => done(undefined));
			void streamAnswer(pi, ctx, view, question);
			return view;
		},
		{
			overlay: true,
			overlayOptions: () => ({ width: "80%", minWidth: 40, maxHeight: "80%", anchor: "center", margin: 1 }),
		},
	);
}

function report(ctx: ExtensionCommandContext, message: string) {
	if (ctx.hasUI) ctx.ui.notify(message, "error");
	else process.stdout.write(`${message}\n`);
}

/* ------------------------------------------------------------------ *
 * Request
 * ------------------------------------------------------------------ */

async function streamAnswer(
	pi: ExtensionAPI,
	ctx: ExtensionCommandContext,
	view: BtwView,
	question: string,
): Promise<void> {
	const model = ctx.model;
	if (!model) {
		view.fail("No model selected.");
		return;
	}

	try {
		const stream = ctx.modelRegistry.streamSimple(
			model,
			{
				systemPrompt: ctx.getSystemPrompt(),
				messages: buildMessages(ctx, question),
				// Same tool declarations as the main thread keep the cached prefix
				// intact. The model is told it has no tools; if it calls one anyway
				// we report it instead of executing anything.
				tools: activeToolSchemas(pi),
			},
			{
				signal: view.signal,
				maxTokens: MAX_TOKENS,
				sessionId: uuidv7(),
				// Keep the main thread's reasoning level so the cached prefix matches.
				reasoning: sideReasoning(ctx.thinkingLevel),
			},
		);

		let toolName: string | undefined;
		for await (const event of stream) {
			switch (event.type) {
				case "text_delta":
					view.append(event.delta);
					break;
				case "toolcall_start":
					toolName = toolNameFromPartial(event.partial);
					if (toolName) view.fail(`The model tried to call ${toolName} instead of answering. Ask that in the main conversation.`);
					break;
				case "error":
					view.fail(event.error?.errorMessage || describeError(event.error));
					return;
				default:
					break;
			}
		}

		const final = await stream.result();
		if (final.stopReason === "error" || final.errorMessage) {
			view.fail(final.errorMessage || "Request failed.");
			return;
		}
		if (final.stopReason === "toolUse" && !toolName) {
			view.fail("The model tried to use a tool instead of answering. Ask that in the main conversation.");
			return;
		}
		view.finish(final);
	} catch (error) {
		// streamSimple throws synchronously when the provider has no auth.
		view.fail(errorMessage(error));
	}
}

/** Headless fallback for print/json/rpc modes. */
async function runHeadless(pi: ExtensionAPI, ctx: ExtensionCommandContext, question: string) {
	const model = ctx.model;
	if (!model) {
		report(ctx, "No model selected.");
		return;
	}

	const out = process.stdout;
	out.write(`> ${question}\n\n`);
	try {
		const stream = ctx.modelRegistry.streamSimple(
			model,
			{ systemPrompt: ctx.getSystemPrompt(), messages: buildMessages(ctx, question), tools: activeToolSchemas(pi) },
			{ maxTokens: MAX_TOKENS, sessionId: uuidv7(), reasoning: sideReasoning(ctx.thinkingLevel) },
		);
		for await (const event of stream) {
			if (event.type === "text_delta") out.write(event.delta);
			if (event.type === "error") {
				out.write(`\n(error: ${event.error?.errorMessage || "request failed"})\n`);
				return;
			}
		}
		out.write("\n");
	} catch (error) {
		out.write(`\n(error: ${errorMessage(error)})\n`);
	}
}

/**
 * Parent conversation as a cache-identical prefix, plus the side question.
 * A trailing in-progress assistant message is dropped: it is not part of any
 * cached prefix yet and would only add a dangling tool call.
 */
function buildMessages(ctx: ExtensionCommandContext, question: string): Message[] {
	const projected = ctx.sessionManager.buildSessionProjection().messages;
	const stable = dropInProgressTail(projected);
	const parent = convertToLlm(stable);
	return [
		...parent,
		{
			role: "user",
			content: [{ type: "text", text: `${SIDE_QUESTION_PROMPT}\n\n${question}` }],
			timestamp: Date.now(),
		} as Message,
	];
}

function dropInProgressTail<T extends { role?: string; stopReason?: string }>(messages: T[]): T[] {
	const last = messages[messages.length - 1];
	if (last?.role === "assistant" && last.stopReason === "pending") return messages.slice(0, -1);
	return messages;
}

function activeToolSchemas(pi: ExtensionAPI): Tool[] {
	const active = new Set(pi.getActiveTools());
	return pi
		.getAllTools()
		.filter((tool) => active.has(tool.name))
		.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.parameters }));
}

function toolNameFromPartial(partial: AssistantMessage): string | undefined {
	for (const block of partial.content) {
		if (block.type === "toolCall") return block.name;
	}
	return undefined;
}

/* ------------------------------------------------------------------ *
 * Overlay
 * ------------------------------------------------------------------ */

class BtwView implements Component {
	readonly signal: AbortSignal;
	private readonly abort = new AbortController();
	private readonly tui: TUI;
	private readonly theme: Theme;
	private readonly question: string;
	private readonly done: () => void;

	private text = "";
	private error: string | undefined;
	private finished = false;
	private tokens: number | undefined;
	private scroll = 0;
	private disposed = false;

	private loader: Loader | undefined;
	private markdown: Markdown | undefined;
	private markdownText: string | undefined;
	private markdownWidth = -1;

	constructor(tui: TUI, theme: Theme, question: string, done: () => void) {
		this.tui = tui;
		this.theme = theme;
		this.question = question;
		this.done = done;
		this.signal = this.abort.signal;
		this.loader = new Loader(tui, (s) => theme.fg("accent", s), (s) => theme.fg("accent", s), "Answering...");
		this.loader.start();
	}

	append(delta: string) {
		this.text += delta;
		this.tui.requestRender();
	}

	finish(message: AssistantMessage) {
		if (this.disposed) return;
		this.stopLoader();
		this.finished = true;
		this.tokens = message.usage.totalTokens;
		if (!this.text.trim()) this.error = "No response received.";
		this.tui.requestRender();
	}

	fail(message: string) {
		if (this.disposed) return;
		this.stopLoader();
		this.error = this.text.trim() ? `${this.text.trimEnd()}\n\n(${message})` : message;
		this.finished = true;
		this.tui.requestRender();
	}

	private stopLoader() {
		this.loader?.stop();
	}

	dispose() {
		this.disposed = true;
		this.stopLoader();
		this.abort.abort();
	}

	invalidate() {
		this.markdown = undefined;
		this.markdownText = undefined;
		this.markdownWidth = -1;
		this.loader?.invalidate();
	}

	handleInput(data: string) {
		if (matchesKey(data, "up") || matchesKey(data, "ctrl+p")) return this.scrollBy(SCROLL_LINES);
		if (matchesKey(data, "down") || matchesKey(data, "ctrl+n")) return this.scrollBy(-SCROLL_LINES);
		if (matchesKey(data, "pageUp")) return this.scrollBy(SCROLL_LINES * 3);
		if (matchesKey(data, "pageDown")) return this.scrollBy(-SCROLL_LINES * 3);
		if (matchesKey(data, "escape") || matchesKey(data, "enter") || matchesKey(data, "return") || matchesKey(data, "space")) {
			this.dismiss();
			return;
		}
		if (matchesKey(data, "ctrl+c")) {
			this.abort.abort();
			this.dismiss();
		}
	}

	private dismiss() {
		if (this.disposed) return;
		this.disposed = true;
		this.stopLoader();
		this.abort.abort();
		this.done();
	}

	private scrollBy(delta: number) {
		this.scroll = Math.max(0, this.scroll + delta);
		this.tui.requestRender();
	}

	render(width: number): string[] {
		const theme = this.theme;
		const box = Math.max(24, width);
		// Two border columns plus one space of padding on each side.
		const content = Math.max(10, box - 4);
		const body = this.bodyLines(content);
		// The header wraps like the body so a long question cannot smear the border.
		const header = new Text(theme.fg("accent", theme.bold("/btw ")) + theme.fg("dim", this.question), 0, 0).render(content);
		// borders(2) + header + blank + body + blank + hint
		const chrome = header.length + 5;
		const maxBody = Math.max(3, Math.min(MAX_OVERLAY_ROWS, this.terminalRows() - chrome, body.length || 3));
		const maxScroll = Math.max(0, body.length - maxBody);
		if (this.scroll > maxScroll) this.scroll = maxScroll;
		const start = Math.max(0, body.length - maxBody - this.scroll);
		const visible = body.slice(start, start + maxBody);

		const edge = (s: string) => theme.fg("borderAccent", s);
		const row = (line: string) => `${edge("│")} ${pad(line, content)} ${edge("│")}`;
		const rule = () => edge(`┌${"─".repeat(content + 2)}┐`);

		const lines: string[] = [rule()];
		for (const line of header) lines.push(row(line));
		lines.push(row(""));
		for (const line of visible) lines.push(row(line));
		for (let i = visible.length; i < maxBody; i++) lines.push(row(""));
		lines.push(row(""));
		lines.push(row(theme.fg("dim", this.hintLine(body.length, maxBody))));
		lines.push(edge(`└${"─".repeat(content + 2)}┘`));
		return lines;
	}

	private terminalRows(): number {
		return this.tui.terminal.rows || MAX_OVERLAY_ROWS;
	}

	private bodyLines(available: number): string[] {
		const width = Math.max(10, available);
		if (this.error !== undefined) {
			return new Text(this.error, 0, 0).render(width);
		}
		if (this.text.trim()) {
			if (this.markdownText !== this.text || this.markdownWidth !== width) {
				this.markdown = new Markdown(this.text, 0, 0, getMarkdownTheme());
				this.markdownText = this.text;
				this.markdownWidth = width;
			}
			return this.markdown!.render(width);
		}
		if (this.finished) return [this.theme.fg("dim", "No response received.")];
		return this.loader ? this.loader.render(width) : [this.theme.fg("dim", "Answering...")];
	}

	private hintLine(total: number, visible: number): string {
		const base = this.finished ? "Enter, Space or Esc to dismiss" : "Esc to dismiss";
		const scroll = total > visible ? ` · Up/Down to scroll` : "";
		const tokens = this.tokens !== undefined ? ` · ${this.tokens.toLocaleString()} tokens` : "";
		return `${base}${scroll}${tokens}`;
	}
}

/* ------------------------------------------------------------------ *
 * Question history
 * ------------------------------------------------------------------ */

function historyPath(): string {
	return path.join(getAgentDir(), HISTORY_FILE);
}

function loadHistory(): string[] {
	try {
		const parsed = JSON.parse(fs.readFileSync(historyPath(), "utf8"));
		return Array.isArray(parsed) ? parsed.filter((q): q is string => typeof q === "string").slice(0, HISTORY_LIMIT) : [];
	} catch {
		return [];
	}
}

function remember(history: string[], question: string) {
	const next = [question, ...history.filter((q) => q !== question)].slice(0, HISTORY_LIMIT);
	history.length = 0;
	history.push(...next);
	try {
		fs.writeFileSync(historyPath(), `${JSON.stringify(next, null, 2)}\n`);
	} catch {
		// History is a convenience; never fail the command over it.
	}
}

/* ------------------------------------------------------------------ *
 * Errors
 * ------------------------------------------------------------------ */

/** Pad a rendered (possibly ANSI-styled) line out to the box's content width. */
function pad(line: string, width: number): string {
	const used = visibleWidth(line);
	return used >= width ? line : line + " ".repeat(width - used);
}

/** pi's session "off" level has no provider-side equivalent; anything else passes through. */
function sideReasoning(level: string | undefined): ThinkingLevel | undefined {
	if (level === undefined || level === "off") return undefined;
	return level as ThinkingLevel;
}

function errorMessage(error: unknown): string {
	if (error instanceof Error) return error.message || error.name;
	return String(error);
}

function describeError(error: unknown): string {
	if (error && typeof error === "object" && "message" in error) return String((error as { message: unknown }).message);
	return errorMessage(error);
}
