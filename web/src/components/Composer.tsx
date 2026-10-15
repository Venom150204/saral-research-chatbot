import { AnimatePresence, motion } from "motion/react";
import { ArrowUp, Square } from "lucide-react";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

import { BorderBeam } from "@/components/ui/border-beam";
import { cn } from "@/lib/utils";

export interface ComposerHandle { setDraft: (text: string) => void }

export const Composer = forwardRef<ComposerHandle, {
  busy: boolean; stoppable: boolean; onSend: (text: string) => void; onStop: () => void;
}>(function Composer({ busy, stoppable, onSend, onStop }, ref) {
  const [text, setText] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);

  useImperativeHandle(ref, () => ({
    setDraft: (value: string) => { setText(value); requestAnimationFrame(() => area.current?.focus()); },
  }));

  useEffect(() => {
    const element = area.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 220)}px`;
  }, [text]);

  const send = () => {
    const value = text.trim();
    if (!value || busy) return;
    onSend(value);
    setText("");
  };

  return (
    <form
      onSubmit={(event) => { event.preventDefault(); if (stoppable) onStop(); else send(); }}
      className="mx-auto w-full max-w-3xl px-4 pb-4 sm:px-6"
    >
      <div
        className={cn(
          "relative flex items-end gap-2 rounded-2xl border bg-surface-1/90 p-2 pl-4 shadow-[0_20px_60px_-24px_var(--shadow)] backdrop-blur-xl transition-colors",
          busy ? "border-accent/30" : "border-hairline-strong focus-within:border-accent/50",
        )}
      >
        {busy && <BorderBeam size={90} duration={2.4} colorFrom="#d4a853" colorTo="#836879" borderWidth={1.5} />}
        <textarea
          ref={area}
          rows={1}
          name="message"
          autoComplete="off"
          value={text}
          aria-label="Message"
          placeholder="Ask for a script, slides, posts, or an explanation…"
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              send();
            }
          }}
          className="max-h-[220px] min-h-[40px] flex-1 resize-none bg-transparent py-2.5 text-[15px] leading-6 text-ink placeholder:text-ink-tertiary focus:outline-none"
        />
        <motion.button
          type="submit"
          aria-label={stoppable ? "Stop" : "Send"}
          disabled={!stoppable && (busy || !text.trim())}
          whileHover={{ scale: 1.06 }}
          whileTap={{ scale: 0.92 }}
          className={cn(
            "relative flex size-10 shrink-0 items-center justify-center rounded-xl transition-colors",
            stoppable ? "bg-ink text-canvas" : "bg-accent text-accent-ink disabled:bg-surface-3 disabled:text-ink-tertiary",
          )}
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={stoppable ? "stop" : "send"}
              initial={{ scale: 0.4, opacity: 0, rotate: -90 }}
              animate={{ scale: 1, opacity: 1, rotate: 0 }}
              exit={{ scale: 0.4, opacity: 0, rotate: 90 }}
              transition={{ duration: 0.18 }}
            >
              {stoppable ? <Square className="size-3.5 fill-current" /> : <ArrowUp className="size-4.5" strokeWidth={2.5} />}
            </motion.span>
          </AnimatePresence>
        </motion.button>
      </div>
      <p className="mt-2 text-center text-[11.5px] text-ink-tertiary">
        Enter to send · Shift+Enter for a new line · claims are checked against your source
      </p>
    </form>
  );
});
