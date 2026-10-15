
/* eslint-disable @typescript-eslint/no-explicit-any */
export type Json = Record<string, any>;

export interface Source {
  chunk_id: string;
  kind: string;
  slide_number: number | null;
  page_number: number | null;
  location: string | null;
  section: string;
  text: string;
  image_description: boolean;
}

export interface DocumentInfo {
  document_id: string;
  document_version: string;
  filename?: string;
  source_kind?: string;
  unit_count?: number;
  chunk_count?: number;
}

export interface StepResult {
  step_id: string;
  status: "succeeded" | "blocked" | "failed_system" | "skipped" | "supported" | "unsupported";
  value?: Json;
}

export interface StoredEvent {
  sequence_no: number;
  role: string;
  event_type: string;
  content: string;
  payload?: Json;
}

export interface StreamEvent extends Json {
  type: "start" | "token" | "retract" | "step" | "artifact" | "check" | "verification" | "interrupt" | "error" | "done";
}

async function json(response: Response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(response.status, data.detail || `The request failed (status ${response.status}).`);
  return data;
}

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

export async function createSession(): Promise<string> {
  return (await json(await fetch("/api/sessions", { method: "POST" }))).session_id;
}

export async function loadSession(sessionId: string) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (!response.ok) return null;
  return (await response.json()) as {
    session_id: string;
    active_document: DocumentInfo | null;
    events: StoredEvent[];
  };
}

export async function uploadDocument(file: File, sessionId: string): Promise<DocumentInfo> {
  const url = `/api/documents?filename=${encodeURIComponent(file.name)}&session_id=${encodeURIComponent(sessionId)}`;
  return json(await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: file,
  }));
}

export async function loadSources(document: DocumentInfo): Promise<Source[]> {
  const query = document.document_version ? `?document_version=${encodeURIComponent(document.document_version)}` : "";
  const response = await fetch(`/api/documents/${encodeURIComponent(document.document_id)}/sources${query}`);
  if (!response.ok) return [];
  return (await response.json()).sources as Source[];
}

export async function* streamTurn(
  body: Json, signal: AbortSignal,
): AsyncGenerator<StreamEvent, void, void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(response.status, friendlyFailure(response.status, data));
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const line = frame.split("\n").find((item) => item.startsWith("data: "));
      if (line) yield JSON.parse(line.slice(6)) as StreamEvent;
    }
  }
}

function friendlyFailure(status: number, data: Json) {
  if (status === 503) {
    return data.detail && !/^The agent/.test(data.detail)
      ? data.detail
      : "The AI model is busy or unavailable right now. Nothing was saved. Try again in a moment.";
  }
  return data.detail || `The request failed (status ${status}).`;
}

export function store(key: string, value: unknown) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch {}
}

export function recall<T>(key: string): T | null {
  try {
    return JSON.parse(localStorage.getItem(key) ?? "null") as T | null;
  } catch {
    return null;
  }
}
