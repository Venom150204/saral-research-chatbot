import { motion } from "motion/react";
import { Check, Copy, RotateCcw, TriangleAlert, Undo2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type { Json, StepResult } from "@/lib/api";
import { plainText } from "@/lib/conversation";
import { sourceLocation, useSources } from "@/lib/sources";
import { cn } from "@/lib/utils";
import { BorderBeam } from "@/components/ui/border-beam";

import { Checks, draftChecks } from "./Checks";
import { Citation } from "./Citation";
import { Fold } from "./Fold";

interface Props {
  result: StepResult;
  reveal: boolean;
  onSubmit: (text: string) => void;
  onDraft: (text: string) => void;
}

const EASE = [0.22, 1, 0.36, 1] as const;

export function ResultCard({ result, reveal, onSubmit, onDraft }: Props) {
  const value = result.value || {};
  if (result.status === "failed_system") {
    return <Notice tone="error">{value.message || "This step could not complete."}</Notice>;
  }
  if (result.status === "skipped") {
    return <p className="text-[13px] text-ink-tertiary">Skipped, because a step it depended on did not complete.</p>;
  }
  return (
    <div className="space-y-3">
      {result.status === "blocked" && <Blocked value={value} onDraft={onDraft} />}
      {value.artifact ? (
        <Draft value={value} draft={result.status === "blocked"} reveal={reveal} onSubmit={onSubmit} onDraft={onDraft} />
      ) : value.kind === "verification" ? (
        <VerificationResult value={value} />
      ) : value.reason && result.status !== "blocked" ? (
        <p className="text-[15px] leading-7 text-ink-muted">{value.reason}</p>
      ) : null}
    </div>
  );
}

function Draft({ value, draft, reveal, onSubmit, onDraft }: {
  value: Json; draft: boolean; reveal: boolean; onSubmit: (t: string) => void; onDraft: (t: string) => void;
}) {
  const { sources } = useSources();
  const artifact = value.artifact;
  const [beam, setBeam] = useState(reveal);
  useEffect(() => {
    if (!reveal) return;
    const timer = setTimeout(() => setBeam(false), 3200);
    return () => clearTimeout(timer);
  }, [reveal]);

  const numbers = useMemo(() => {
    const map = new Map<string, number>();
    (artifact.blocks || []).forEach((block: Json) => block.units.forEach((unit: Json) => {
      if (unit.factual) (unit.support || []).forEach((ref: Json) => {
        if (!map.has(ref.chunk_id)) map.set(ref.chunk_id, map.size + 1);
      });
    }));
    return map;
  }, [artifact]);

  const units: Json[] = (artifact.blocks || []).flatMap((block: Json) => block.units);
  const step = reveal ? Math.min(0.07, 1.6 / Math.max(units.length, 1)) : 0;
  let index = 0;
  const checksDelay = reveal ? units.length * step + 0.3 : 0;

  return (
    <motion.div
      initial={reveal ? { opacity: 0, y: 12, scale: 0.985 } : false}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, ease: EASE }}
      className={cn(
        "relative overflow-hidden rounded-xl border bg-surface-1 shadow-[0_1px_2px_var(--shadow)]",
        draft ? "border-dashed border-miss/40" : "border-hairline",
      )}
    >
      {beam && <BorderBeam size={120} duration={3} colorFrom="#d4a853" colorTo="#836879" borderWidth={1.5} />}
      <div className="space-y-5 p-5 sm:p-6">
        {draft && <p className="text-[12px] font-medium uppercase tracking-wide text-miss-text">Draft shown for review, not saved</p>}
        {(artifact.blocks || []).map((block: Json, blockIndex: number) => (
          <section key={block.block_id ?? blockIndex} className="space-y-2">
            {block.title && (
              <motion.h4
                initial={reveal ? { opacity: 0, x: -6 } : false}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * step, duration: 0.35, ease: EASE }}
                className="text-[15px] font-semibold tracking-[-0.01em] text-ink"
              >
                <span className="mr-2 text-ink-tertiary">#{blockIndex + 1}</span>{block.title}
              </motion.h4>
            )}
            {block.units.map((unit: Json) => {
              const delay = index++ * step;
              const note = /note/i.test(unit.role || "");
              return (
                <motion.p
                  key={unit.unit_id}
                  initial={reveal ? { opacity: 0, y: 8, filter: "blur(6px)" } : false}
                  animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                  transition={{ delay, duration: 0.45, ease: EASE }}
                  className={cn(
                    "text-[15px] leading-7",
                    note ? "rounded-lg border-l-2 border-glow/50 bg-surface-2/60 py-1.5 pl-3 pr-2 text-ink-subtle" : "text-ink-muted",
                  )}
                >
                  {note && <span className="mr-1.5 text-[11px] font-semibold uppercase tracking-wide text-glow/80">Speaker note</span>}
                  {unit.text}
                  {unit.factual && (unit.support || []).map((ref: Json) => (
                    <Citation key={ref.chunk_id} chunkId={ref.chunk_id} label={String(numbers.get(ref.chunk_id))} />
                  ))}
                </motion.p>
              );
            })}
          </section>
        ))}
      </div>
      <motion.div
        initial={reveal ? { opacity: 0 } : false}
        animate={{ opacity: 1 }}
        transition={{ delay: checksDelay, duration: 0.4 }}
        className="space-y-3 border-t border-hairline bg-surface-2/40 px-5 py-4 sm:px-6"
      >
        <Checks items={draftChecks(value)} delay={checksDelay} />
        <div className="space-y-1">
          {numbers.size > 0 && (
            <Fold title={`Sources (${numbers.size})`}>
              {[...numbers.entries()].map(([chunkId, number]) => {
                const source = sources.get(chunkId);
                return (
                  <p key={chunkId}>
                    <span className="mr-1.5 font-semibold text-ink">[{number}] {source ? sourceLocation(source) : "Source"}</span>
                    {source ? source.text : chunkId}
                  </p>
                );
              })}
              <p className="text-ink-tertiary">Sentences without a number are general explanation or framing, not claims taken from the source. They are reviewed separately.</p>
            </Fold>
          )}
          <Advice value={value} draft={draft} onDraft={onDraft} />
          <Changes value={value} />
          <Trace value={value} />
        </div>
        <Actions value={value} onSubmit={onSubmit} onDraft={onDraft} />
      </motion.div>
    </motion.div>
  );
}

function Advice({ value, draft, onDraft }: { value: Json; draft: boolean; onDraft: (t: string) => void }) {
  const review = value.verification;
  if (draft || !review || review.fulfills_request !== false) return null;
  const notes: string[] = review.quality_issues?.length ? review.quality_issues : [review.quality_reason].filter(Boolean);
  return (
    <Fold title="Reviewer's suggestions">
      <ul className="list-disc space-y-1 pl-4">{notes.map((note) => <li key={note}>{note}</li>)}</ul>
      <Chips actions={review.guidance?.suggested_actions} onDraft={onDraft} />
    </Fold>
  );
}

function Changes({ value }: { value: Json }) {
  const deltas: Json[] = value.deltas?.length ? value.deltas : value.repair?.deltas || [];
  if (!deltas.length) return null;
  return (
    <Fold title={value.deltas?.length ? "What changed" : "Automatically repaired before saving"} defaultOpen={Boolean(value.deltas?.length)}>
      {deltas.map((delta) => (
        <div key={delta.unit_id} className="space-y-1 rounded-lg bg-surface-2/70 p-3">
          <p className="text-del-text line-through decoration-danger/60">{delta.old_text}</p>
          <p className="text-ins-text">{delta.new_text}</p>
        </div>
      ))}
      {value.reason && <p><span className="font-semibold text-ink">Why:</span> {value.reason}</p>}
    </Fold>
  );
}

function Trace({ value }: { value: Json }) {
  const lines: string[] = [];
  if (value.selection_mode) lines.push(`Evidence: ${value.selection_mode}`);
  if (value.regeneration?.attempted) lines.push(`One rewrite was requested: ${(value.regeneration.feedback || []).join(" ")}`);
  (value.constraints?.ignored || []).forEach((item: Json) => lines.push(`Limit not enforced (${item.reason}): ${item.quote}`));
  if (value.artifact) lines.push(`Version ${value.artifact.version} · ${value.artifact.model_id} · ${value.artifact.prompt_version}`);
  if (!lines.length) return null;
  return <Fold title="How this was made">{lines.map((line) => <p key={line} className="font-mono text-[12px]">{line}</p>)}</Fold>;
}

function Actions({ value, onSubmit, onDraft }: { value: Json; onSubmit: (t: string) => void; onDraft: (t: string) => void }) {
  const [copied, setCopied] = useState<"idle" | "done" | "failed">("idle");
  const adjust = (value.constraints?.checked || []).filter((item: Json) => !item.met && item.enforced === false && item.fix);
  return (
    <div className="flex flex-wrap items-center gap-2">
      <ActionButton
        icon={copied === "done" ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
        label={copied === "done" ? "Copied" : copied === "failed" ? "Copy failed" : "Copy"}
        onClick={async () => {
          try { await navigator.clipboard.writeText(plainText(value.artifact)); setCopied("done"); }
          catch { setCopied("failed"); }
          setTimeout(() => setCopied("idle"), 1500);
        }}
      />
      {value.kind === "edited_artifact" && (
        <ActionButton icon={<Undo2 className="size-3.5" />} label="Undo this edit" onClick={() => onSubmit("Undo the last change.")} />
      )}
      {adjust.map((item: Json) => {
        const label = /^add/i.test(item.fix) ? "Make it longer" : "Make it shorter";
        return <ActionButton key={item.quote} icon={<RotateCcw className="size-3.5" />} label={label} onClick={() => onDraft(`${label}. ${item.fix}`)} />;
      })}
    </div>
  );
}

export function ActionButton({ icon, label, onClick }: { icon: React.ReactNode; label: string; onClick: () => void }) {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      whileHover={{ y: -1 }}
      whileTap={{ scale: 0.96 }}
      className="inline-flex items-center gap-1.5 rounded-md border border-hairline bg-surface-1 px-2.5 py-1.5 text-[12.5px] font-medium text-ink-subtle transition-colors hover:border-hairline-strong hover:text-ink"
    >
      {icon}{label}
    </motion.button>
  );
}

function Blocked({ value, onDraft }: { value: Json; onDraft: (t: string) => void }) {
  const issue = value.blocked || value;
  const guidance = issue.guidance || {};
  const points: string[] = guidance.observations?.length ? guidance.observations : issue.quality_issues || [];
  const claims: string[] = (issue.claim_issues || []).map((item: Json) => `${String(item.verdict).replaceAll("_", " ")}: ${item.reason}`);
  return (
    <Notice tone="warn">
      <p><strong className="text-ink">Not saved.</strong> {guidance.summary || issue.message || "This draft did not pass its checks."}</p>
      {[...points, ...claims].length > 0 && (
        <ul className="mt-2 list-disc space-y-1 pl-4">{[...points, ...claims].map((item) => <li key={item}>{item}</li>)}</ul>
      )}
      <Chips actions={guidance.suggested_actions} onDraft={onDraft} />
    </Notice>
  );
}

function Chips({ actions, onDraft }: { actions?: Json[]; onDraft: (t: string) => void }) {
  if (!actions?.length) return null;
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {actions.map((item) => (
        <button key={item.label} type="button" onClick={() => onDraft(item.instruction)}
          className="rounded-full border border-hairline-strong bg-surface-2 px-3 py-1 text-[12.5px] text-ink-muted transition-colors hover:border-accent/50 hover:text-ink">
          {item.label}
        </button>
      ))}
    </div>
  );
}

function VerificationResult({ value }: { value: Json }) {
  const verdicts: Json[] = value.report?.verdicts || [];
  const supported = verdicts.filter((item) => item.verdict === "supported").length;
  const failed = verdicts.filter((item) => item.verdict !== "supported");
  return (
    <div className="space-y-2">
      <Checks items={[{ text: `${supported} of ${verdicts.length} statements supported by the source`, ok: Boolean(value.accepted) }]} />
      {!value.accepted && value.reason && <p className="text-[14px] text-ink-subtle">{value.reason}</p>}
      {failed.length > 0 && (
        <Fold title="Details">
          {failed.map((item) => <p key={item.unit_id}><strong className="text-ink">{String(item.verdict).replaceAll("_", " ")}</strong>: {item.reason}</p>)}
        </Fold>
      )}
    </div>
  );
}

export function Notice({ tone, children }: { tone: "warn" | "error"; children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "flex gap-3 rounded-xl border p-4 text-[14px] leading-6",
        tone === "error" ? "border-danger/30 bg-danger/10 text-danger-text" : "border-miss/35 bg-miss/[0.08] text-miss-text",
      )}
    >
      <TriangleAlert className="mt-0.5 size-4 shrink-0" />
      <div className="min-w-0 flex-1">{children}</div>
    </motion.div>
  );
}
