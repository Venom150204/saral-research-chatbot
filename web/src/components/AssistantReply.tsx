import { AnimatePresence, motion } from "motion/react";
import { Check, LoaderCircle, RotateCcw } from "lucide-react";

import type { AssistantMessage, Part } from "@/lib/conversation";
import { cn } from "@/lib/utils";
import { TextShimmer } from "@/components/ui/text-shimmer";

import { AgentText } from "./AgentText";
import { Logo } from "./Logo";
import { answerChecks, Checks } from "./Checks";
import { Fold } from "./Fold";
import { ActionButton, Notice, ResultCard } from "./ResultCard";

interface Props {
  message: AssistantMessage;
  onSubmit: (text: string) => void;
  onDraft: (text: string) => void;
  onAnswer: (messageId: string, partId: string, option: string, turnId: string) => void;
  onRetry: (text: string) => void;
}

export function AssistantReply({ message, onSubmit, onDraft, onAnswer, onRetry }: Props) {
  const lastText = [...message.parts].reverse().find((part) => part.kind === "text");
  return (
    <div className="flex gap-3.5">
      <BrandAvatar live={message.live} />
      <div className="min-w-0 flex-1 space-y-4 pt-0.5">
        <AnimatePresence mode="popLayout">
          {message.pending && (
            <motion.div key="thinking" exit={{ opacity: 0, y: -4 }} className="pt-1">
              <TextShimmer className="text-[14px] font-medium [--base-color:var(--shimmer-base)] [--base-gradient-color:var(--shimmer-shine)]" duration={1.6}>
                Thinking…
              </TextShimmer>
            </motion.div>
          )}
        </AnimatePresence>
        {message.parts.map((part) => (
          <motion.div
            key={part.id}
            layout="position"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
          >
            <PartView
              part={part}
              live={message.live && part === lastText}
              onSubmit={onSubmit}
              onDraft={onDraft}
              onAnswer={(option, turnId) => onAnswer(message.id, part.id, option, turnId)}
              onRetry={onRetry}
            />
          </motion.div>
        ))}
      </div>
    </div>
  );
}

function PartView({ part, live, onSubmit, onDraft, onAnswer, onRetry }: {
  part: Part; live: boolean; onSubmit: (t: string) => void; onDraft: (t: string) => void;
  onAnswer: (option: string, turnId: string) => void; onRetry: (t: string) => void;
}) {
  switch (part.kind) {
    case "text":
      return <AgentText text={part.text} live={live} />;
    case "steps":
      return <Steps items={part.items} />;
    case "result":
      return <ResultCard result={part.result} reveal={part.reveal} onSubmit={onSubmit} onDraft={onDraft} />;
    case "verification": {
      const issues: string[] = part.event.issues || [];
      return (
        <div className="space-y-2">
          <Checks items={answerChecks(part.event)} />
          {issues.length > 0 && (
            <Fold title="What the check found">
              <ul className="list-disc space-y-1 pl-4">{issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>
            </Fold>
          )}
        </div>
      );
    }
    case "question":
      return (
        <div className="rounded-xl border border-accent/25 bg-accent/[0.06] p-4">
          <p className="mb-3 text-[15px] font-medium text-ink">{part.question}</p>
          <div className="flex flex-wrap gap-2">
            {part.options.map((option) => (
              <motion.button
                key={option}
                type="button"
                disabled={part.answered}
                whileHover={part.answered ? undefined : { y: -1 }}
                whileTap={part.answered ? undefined : { scale: 0.96 }}
                onClick={() => onAnswer(option, part.turnId)}
                className="rounded-full border border-hairline-strong bg-surface-2 px-3.5 py-1.5 text-[13px] font-medium text-ink-muted transition-colors enabled:hover:border-accent/60 enabled:hover:text-ink disabled:opacity-50"
              >
                {option}
              </motion.button>
            ))}
          </div>
        </div>
      );
    case "error":
      return (
        <div className="space-y-2">
          <Notice tone="error">{part.message}</Notice>
          {part.retryText && (
            <ActionButton icon={<RotateCcw className="size-3.5" />} label="Try again" onClick={() => onRetry(part.retryText!)} />
          )}
        </div>
      );
    case "meta":
      return part.text ? <p className="font-mono text-[11.5px] text-ink-tertiary">{part.text}</p> : null;
  }
}

function Steps({ items }: { items: { detail: string; done: boolean }[] }) {
  return (
    <ul className="space-y-1.5 rounded-xl border border-hairline bg-surface-1/60 px-4 py-3">
      <AnimatePresence initial={false}>
        {items.map((item, index) => (
          <motion.li
            key={`${index}-${item.detail}`}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.25 }}
            className={cn("flex items-center gap-2.5 text-[13px]", item.done ? "text-ink-subtle" : "text-ink")}
          >
            <span className="relative flex size-4 items-center justify-center">
              <AnimatePresence mode="wait" initial={false}>
                {item.done ? (
                  <motion.span key="done" initial={{ scale: 0, rotate: -45 }} animate={{ scale: 1, rotate: 0 }}
                    transition={{ type: "spring", stiffness: 500, damping: 22 }}
                    className="flex size-4 items-center justify-center rounded-full bg-ok/20 text-ok">
                    <Check className="size-2.5" strokeWidth={3} />
                  </motion.span>
                ) : (
                  <motion.span key="run" exit={{ scale: 0 }} className="text-accent-text">
                    <LoaderCircle className="size-4 animate-spin" />
                  </motion.span>
                )}
              </AnimatePresence>
            </span>
            {item.done ? item.detail : <TextShimmer duration={1.8} className="[--base-color:var(--ink-subtle)] [--base-gradient-color:var(--shimmer-shine)]">{item.detail}</TextShimmer>}
          </motion.li>
        ))}
      </AnimatePresence>
    </ul>
  );
}

export function BrandAvatar({ live = false }: { live?: boolean }) {
  return (
    <div className="relative mt-0.5 h-7 w-9 shrink-0">
      {live && (
        <motion.span
          aria-hidden
          className="absolute inset-0 rounded-[9px] bg-accent/45 blur-md"
          animate={{ opacity: [0.25, 0.85, 0.25] }}
          transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
        />
      )}
      <div className="relative flex h-7 w-9 items-center justify-center rounded-[9px] border border-hairline bg-surface-1">
        <Logo variant="mark" className="h-3.5 w-auto" />
      </div>
    </div>
  );
}
