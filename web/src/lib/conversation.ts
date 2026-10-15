
import type { Json, StepResult, StoredEvent, StreamEvent } from "./api";

export type Part =
  | { kind: "text"; id: string; text: string; closing: boolean }
  | { kind: "steps"; id: string; items: { detail: string; done: boolean }[] }
  | { kind: "result"; id: string; result: StepResult; reveal: boolean }
  | { kind: "verification"; id: string; event: Json }
  | { kind: "question"; id: string; question: string; options: string[]; turnId: string; answered: boolean }
  | { kind: "error"; id: string; message: string; retryText: string | null }
  | { kind: "meta"; id: string; text: string };

export interface UserMessage { role: "user"; id: string; text: string }
export interface AssistantMessage {
  role: "assistant";
  id: string;
  parts: Part[];
  pending: boolean;
  live: boolean;
  toolsRan: boolean;
}
export type Message = UserMessage | AssistantMessage;

let counter = 0;
export const newId = (prefix = "p") => `${prefix}-${Date.now().toString(36)}-${(counter++).toString(36)}`;

export function assistant(live = true): AssistantMessage {
  return { role: "assistant", id: newId("m"), parts: [], pending: live, live, toolsRan: false };
}

function finishSteps(parts: Part[]): Part[] {
  return parts.map((part) => part.kind === "steps" && part.items.some((item) => !item.done)
    ? { ...part, items: part.items.map((item) => ({ ...item, done: true })) }
    : part);
}

export function applyEvent(message: AssistantMessage, event: StreamEvent, turnId: string): AssistantMessage {
  let parts = message.parts;
  let toolsRan = message.toolsRan;
  const last = parts[parts.length - 1];
  switch (event.type) {
    case "token": {
      const text = String(event.text ?? "");
      if (last?.kind === "text") {
        parts = [...parts.slice(0, -1), { ...last, text: last.text + text }];
      } else if (text.trim()) {
        parts = [...parts, { kind: "text", id: newId(), text, closing: toolsRan }];
      }
      break;
    }
    case "retract": {
      let remove = Number(event.chars) || 0;
      while (remove > 0 && parts.length) {
        const tail = parts[parts.length - 1];
        if (tail.kind !== "text") break;
        if (tail.text.length > remove) {
          parts = [...parts.slice(0, -1), { ...tail, text: tail.text.slice(0, tail.text.length - remove) }];
          remove = 0;
        } else {
          remove -= tail.text.length;
          parts = parts.slice(0, -1);
        }
      }
      break;
    }
    case "step": {
      parts = finishSteps(parts);
      toolsRan = true;
      const item = { detail: String(event.detail || event.tool), done: false };
      const tail = parts[parts.length - 1];
      parts = tail?.kind === "steps"
        ? [...parts.slice(0, -1), { ...tail, items: [...tail.items, item] }]
        : [...parts, { kind: "steps", id: newId(), items: [item] }];
      break;
    }
    case "artifact": {
      parts = finishSteps(parts);
      const card: Part = { kind: "result", id: newId(), result: event.result as StepResult, reveal: true };
      const closing = parts.findIndex((part) => part.kind === "text" && part.closing);
      parts = closing >= 0 ? [...parts.slice(0, closing), card, ...parts.slice(closing)] : [...parts, card];
      break;
    }
    case "verification":
      parts = [...finishSteps(parts), { kind: "verification", id: newId(), event }];
      break;
    case "interrupt":
      parts = [...finishSteps(parts), {
        kind: "question", id: newId(), question: String(event.question || "Which do you mean?"),
        options: (event.options as string[]) || [], turnId, answered: false,
      }];
      break;
    case "error":
      parts = [...finishSteps(parts), { kind: "error", id: newId(), message: String(event.message), retryText: null }];
      break;
    case "done":
      parts = [...finishSteps(parts), { kind: "meta", id: newId(), text: turnMeta(event) }];
      break;
    default:
      return message;
  }
  return { ...message, parts, toolsRan, pending: false };
}

export function turnMeta(event: Json): string {
  const parts: string[] = [];
  if (event.first_token_ms != null) parts.push(`first words in ${(event.first_token_ms / 1000).toFixed(1)} s`);
  if (event.latency_ms != null) parts.push(`done in ${(event.latency_ms / 1000).toFixed(1)} s`);
  if (event.model_calls) parts.push(`${event.model_calls} model call${event.model_calls === 1 ? "" : "s"}`);
  if (event.input_tokens) {
    parts.push(`${Number(event.input_tokens).toLocaleString()} tokens in, ${Number(event.output_tokens ?? 0).toLocaleString()} out`);
  }
  return parts.join(" · ");
}

export function replay(events: StoredEvent[]): Message[] {
  const out: Message[] = [];
  const reply = (payload: Json, extraText?: string): AssistantMessage => {
    const message = assistant(false);
    if (payload.message) message.parts.push({ kind: "text", id: newId(), text: payload.message, closing: false });
    (payload.results || []).forEach((result: StepResult) =>
      message.parts.push({ kind: "result", id: newId(), result, reveal: false }));
    if (payload.verification) message.parts.push({ kind: "verification", id: newId(), event: payload.verification });
    if (extraText) message.parts.push({ kind: "text", id: newId(), text: extraText, closing: false });
    if (payload.usage) message.parts.push({ kind: "meta", id: newId(), text: turnMeta(payload.usage) });
    return message;
  };
  [...events].sort((a, b) => a.sequence_no - b.sequence_no).forEach((event) => {
    if (event.role === "user" && event.event_type === "message") {
      out.push({ role: "user", id: newId("u"), text: event.content });
    } else if (event.event_type === "agent_interrupt") {
      out.push(reply({ message: event.payload?.message, results: event.payload?.results || [] }, event.content));
    } else if (event.event_type === "workflow_action" && event.payload?.kind === "agent_result") {
      out.push(reply(event.payload));
    } else if (event.event_type === "workflow_failed") {
      const message = assistant(false);
      message.parts.push({ kind: "error", id: newId(), message: event.content || "This request failed.", retryText: null });
      out.push(message);
    }
  });
  return out;
}

export function plainText(artifact: Json): string {
  return (artifact.blocks || []).map((block: Json) => [block.title, ...block.units.map((unit: Json) => unit.text)]
    .filter(Boolean).join("\n")).join("\n\n");
}
