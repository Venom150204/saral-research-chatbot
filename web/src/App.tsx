import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError, createSession, loadSession, loadSources, recall, store, streamTurn, uploadDocument,
  type DocumentInfo, type Source,
} from "@/lib/api";
import { applyEvent, assistant, newId, replay, type AssistantMessage, type Message } from "@/lib/conversation";
import { SourceContext } from "@/lib/sources";
import { cn } from "@/lib/utils";
import { useTheme } from "@/lib/theme";

import { Ambient } from "./components/Ambient";
import { AssistantReply } from "./components/AssistantReply";
import { Composer, type ComposerHandle } from "./components/Composer";
import { DeckPanel } from "./components/DeckPanel";
import { StartScreen } from "./components/StartScreen";
import { BrandAvatar } from "./components/AssistantReply";
import { TopBar } from "./components/TopBar";

const STARTERS: [string, string][] = [
  ["90-second policy script", "Write a 90-second script for policymakers explaining why this work matters."],
  ["Talk for graduate students", "Create a 7-slide talk for graduate students focusing on the method, with a speaker note for each slide."],
  ["Tweet-sized abstracts", "Create 5 tweet-sized abstracts for a technically curious general audience."],
  ["Explain the key idea", "Explain the key idea of this work in plain language."],
];
const ACCEPTED = /\.(pptx|pdf|tex|md|markdown|txt)$/i;

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [doc, setDoc] = useState<DocumentInfo | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [sourceIds, setSourceIds] = useState<Record<string, string>>({});
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [stoppable, setStoppable] = useState(false);
  const [panel, setPanel] = useState(false);
  const [highlight, setHighlight] = useState<string | null>(null);
  const [upload, setUpload] = useState({ text: "", error: false, busy: false });
  const [booting, setBooting] = useState(true);
  const controller = useRef<AbortController | null>(null);
  const composer = useRef<ComposerHandle>(null);
  const nearBottom = useRef(true);
  const theme = useTheme();
  const booted = useRef(false);

  const openSource = useCallback(async (document: DocumentInfo) => {
    const list = await loadSources(document);
    setSources(list);
    setSourceIds(Object.fromEntries(list.map((item, index) => [`S${index + 1}`, item.chunk_id])));
  }, []);

  const fresh = useCallback(async () => {
    setDoc(null); setSources([]); setSourceIds({}); setMessages([]); setPanel(false);
    store("saral.document", null);
    try {
      const id = await createSession();
      setSessionId(id);
      store("saral.session", id);
    } catch (error) {
      setUpload({ text: `Cannot start a session: ${(error as Error).message}`, error: true, busy: false });
    }
  }, []);

  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    (async () => {
      const remembered = recall<string>("saral.session");
      const session = remembered ? await loadSession(remembered).catch(() => null) : null;
      if (session?.active_document) {
        const saved = recall<DocumentInfo>("saral.document");
        const document = {
          ...session.active_document,
          ...(saved?.document_version === session.active_document.document_version ? saved : {}),
        };
        setSessionId(session.session_id);
        setDoc(document);
        await openSource(document);
        setMessages(replay(session.events || []));
      } else if (session) {
        setSessionId(session.session_id);
      } else {
        await fresh();
      }
      setBooting(false);
    })();
  }, [fresh, openSource]);

  const onUpload = async (file: File) => {
    if (!ACCEPTED.test(file.name)) {
      setUpload({ text: "Please choose a .pptx, .pdf, .tex, .md or .txt file.", error: true, busy: false });
      return;
    }
    if (!sessionId) {
      setUpload({ text: "The session is not ready yet. Reload the page and try again.", error: true, busy: false });
      return;
    }
    const started = Date.now();
    const tick = () => setUpload({
      text: `Reading ${file.name}: extracting text, indexing every point… ${Math.floor((Date.now() - started) / 1000)}s`,
      error: false, busy: true,
    });
    tick();
    const clock = setInterval(tick, 1000);
    try {
      const document = await uploadDocument(file, sessionId);
      store("saral.document", document);
      setDoc(document);
      setMessages([]);
      await openSource(document);
      setUpload({ text: "", error: false, busy: false });
    } catch (error) {
      setUpload({ text: (error as Error).message, error: true, busy: false });
    } finally {
      clearInterval(clock);
    }
  };

  const update = (id: string, change: (message: AssistantMessage) => AssistantMessage) =>
    setMessages((list) => list.map((item) => (item.id === id && item.role === "assistant" ? change(item) : item)));

  const runTurn = async ({ message = null, resume = null, turnId }: { message?: string | null; resume?: string | null; turnId: string }) => {
    if (!doc || !sessionId) return;
    const reply = assistant(true);
    setMessages((list) => [...list, reply]);
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setStoppable(true);
    nearBottom.current = true;
    try {
      const body = {
        session_id: sessionId, turn_id: turnId,
        document_id: doc.document_id, document_version: doc.document_version,
        ...(message !== null ? { message } : { resume }),
      };
      for await (const event of streamTurn(body, abort.signal)) {
        if (event.type === "start" && event.source_ids) setSourceIds(event.source_ids);
        update(reply.id, (current) => applyEvent(current, event, turnId));
      }
    } catch (error) {
      const err = error as Error;
      if (err.name === "AbortError") {
        update(reply.id, (current) => ({ ...current, parts: [...current.parts, { kind: "meta", id: newId(), text: "Stopped." }] }));
      } else {
        const text = error instanceof ApiError ? err.message : "Could not reach the SARAL server. Check that it is still running.";
        update(reply.id, (current) => ({
          ...current,
          parts: [...current.parts, { kind: "error", id: newId(), message: text, retryText: message }],
        }));
      }
    } finally {
      update(reply.id, (current) => ({
        ...current, live: false, pending: false,
        parts: current.parts.map((part) => part.kind === "steps"
          ? { ...part, items: part.items.map((item) => ({ ...item, done: true })) } : part),
      }));
      controller.current = null;
      setBusy(false);
      setStoppable(false);
    }
  };

  const send = (text: string) => {
    if (busy || !doc) return;
    setMessages((list) => [...list, { role: "user", id: newId("u"), text }]);
    void runTurn({ message: text, turnId: `turn-${crypto.randomUUID()}` });
  };

  const answer = (messageId: string, partId: string, option: string, turnId: string) => {
    if (busy) return;
    update(messageId, (current) => ({
      ...current,
      parts: current.parts.map((part) => (part.id === partId && part.kind === "question" ? { ...part, answered: true } : part)),
    }));
    setMessages((list) => [...list, { role: "user", id: newId("u"), text: option }]);
    void runTurn({ resume: option, turnId });
  };

  const retry = (text: string) => {
    setMessages((list) => list.filter((item, index) => !(index === list.length - 1 && item.role === "assistant")));
    send(text);
  };

  useEffect(() => {
    const onScroll = () => {
      nearBottom.current = window.innerHeight + window.scrollY >= document.body.scrollHeight - 160;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => {
    if (nearBottom.current) window.scrollTo({ top: document.body.scrollHeight, behavior: busy ? "auto" : "smooth" });
  }, [messages, busy]);

  const context = useMemo(() => ({
    sources: new Map(sources.map((item) => [item.chunk_id, item])),
    sourceIds,
    show: (chunkId: string) => { setHighlight(chunkId); setPanel(true); },
  }), [sources, sourceIds]);

  const deckLabel = doc ? [
    doc.filename || "Uploaded source",
    doc.unit_count ? `${doc.unit_count} ${doc.source_kind === "slide_deck" ? "slides" : doc.source_kind === "document" ? "pages" : "sections"}` : null,
  ].filter(Boolean).join(" · ") : null;

  return (
    <SourceContext.Provider value={context}>
      <Ambient intensity={doc ? 0.45 : 1} />
      <TopBar deckLabel={deckLabel} onDeck={() => setPanel((open) => !open)} onNewChat={() => { if (!busy) void fresh(); }} busy={busy}
        theme={theme.theme} onToggleTheme={theme.toggle} />
      <AnimatePresence mode="wait">
        {booting ? null : !doc ? (
          <motion.div key="start" exit={{ opacity: 0, y: -12, filter: "blur(6px)" }} transition={{ duration: 0.35 }}>
            <StartScreen onUpload={onUpload} status={upload.text} error={upload.error} busy={upload.busy} />
          </motion.div>
        ) : (
          <motion.div
            key="chat"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
            className={cn("flex min-h-[calc(100vh-56px)] flex-col transition-[padding] duration-300 ease-out", panel && "lg:pr-[448px]")}
          >
            <section aria-live="polite" aria-label="Conversation" className="mx-auto w-full max-w-3xl flex-1 space-y-8 px-4 pb-10 pt-8 sm:px-6">
              <Welcome name={doc.filename || "your source"} count={sources.length} onPick={send} />
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 14, filter: "blur(4px)" }}
                  animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                  transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                >
                  {message.role === "user" ? (
                    <div className="flex justify-end">
                      <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md border border-hairline-strong bg-surface-2 px-4 py-2.5 text-[15px] leading-6 text-ink">
                        {message.text}
                      </div>
                    </div>
                  ) : (
                    <AssistantReply
                      message={message}
                      onSubmit={send}
                      onDraft={(text) => composer.current?.setDraft(text)}
                      onAnswer={answer}
                      onRetry={retry}
                    />
                  )}
                </motion.div>
              ))}
            </section>
            <div className="sticky bottom-0 z-20 bg-gradient-to-t from-canvas via-canvas/90 to-transparent pt-6">
              <Composer
                ref={composer}
                busy={busy}
                stoppable={stoppable}
                onSend={send}
                onStop={() => controller.current?.abort()}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <DeckPanel
        open={panel && Boolean(doc)}
        title={doc?.filename || "Source"}
        sources={sources}
        highlight={highlight}
        onClose={() => setPanel(false)}
      />
    </SourceContext.Provider>
  );
}

function Welcome({ name, count, onPick }: { name: string; count: number; onPick: (text: string) => void }) {
  return (
    <div className="flex gap-3.5">
      <BrandAvatar />
      <div className="min-w-0 flex-1 space-y-4 pt-0.5">
        <p className="text-[15px] leading-7 text-ink-muted">
          I&apos;ve read <strong className="font-semibold text-ink">{name}</strong>
          {count ? <>: <span className="text-ink">{count}</span> points I can cite</> : null}. What would you like to make from it?
        </p>
        <div className="flex flex-wrap gap-2">
          {STARTERS.map(([label, text], index) => (
            <motion.button
              key={label}
              type="button"
              onClick={() => onPick(text)}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 + index * 0.07 }}
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.96 }}
              className="rounded-full border border-hairline-strong bg-surface-1/80 px-3.5 py-1.5 text-[13px] font-medium text-ink-muted backdrop-blur transition-colors hover:border-accent/50 hover:text-ink"
            >
              {label}
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
}
