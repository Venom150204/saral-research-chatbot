import { motion } from "motion/react";
import { Check, CircleAlert } from "lucide-react";

import type { Json } from "@/lib/api";
import { cn } from "@/lib/utils";

export interface CheckItem { text: string; ok: boolean; title?: string }

export function Checks({ items, delay = 0 }: { items: CheckItem[]; delay?: number }) {
  if (!items.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item, index) => (
        <motion.span
          key={`${item.text}-${index}`}
          title={item.title}
          initial={{ opacity: 0, scale: 0.85, y: 4 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ delay: delay + index * 0.06, type: "spring", stiffness: 420, damping: 26 }}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[12px] leading-4",
            item.ok ? "border-ok/30 bg-ok/10 text-ok-text" : "border-miss/35 bg-miss/10 text-miss-text",
          )}
        >
          {item.ok ? <Check className="size-3" /> : <CircleAlert className="size-3" />}
          {item.text}
        </motion.span>
      ))}
    </div>
  );
}

export function draftChecks(value: Json): CheckItem[] {
  const items: CheckItem[] = [];
  const review = value.verification || {};
  const verdicts: Json[] = review.verdicts || [];
  if (verdicts.length) {
    const supported = verdicts.filter((item) => item.verdict === "supported").length;
    items.push({ text: `${supported} of ${verdicts.length} research claims match the source`, ok: supported === verdicts.length });
  }
  const background = new Set(
    (value.artifact?.blocks || []).flatMap((block: Json) =>
      block.units.filter((unit: Json) => unit.basis === "background").map((unit: Json) => unit.unit_id)),
  );
  const explained = (review.uncited_verdicts || []).filter((item: Json) => background.has(item.unit_id));
  if (explained.length) {
    const fine = explained.filter((item: Json) => item.verdict === "acceptable").length;
    items.push({ text: `${fine} of ${explained.length} explanations checked as general knowledge`, ok: fine === explained.length });
  }
  (review.requirement_verdicts || []).forEach((item: Json) =>
    items.push({ text: item.requirement, ok: item.met, title: item.reason }));
  (value.constraints?.checked || []).forEach((item: Json) =>
    items.push({ text: `"${item.quote}": ${item.measured}${item.enforced === false ? " (target)" : ""}`, ok: item.met }));
  if (value.citation_completions?.length) {
    items.push({ text: `${value.citation_completions.length} citation(s) added for figures`, ok: true });
  }
  return items;
}

export function answerChecks(event: Json): CheckItem[] {
  const items: CheckItem[] = [];
  if (event.claims) {
    items.push({ text: `${event.claims_supported} of ${event.claims} research claims match the source`, ok: event.claims_supported === event.claims });
  }
  if (event.explanations) {
    items.push({ text: `${event.explanations_accepted} of ${event.explanations} explanations checked as general knowledge`, ok: event.explanations_accepted === event.explanations });
  }
  if (!event.claims && !event.explanations && event.passed === null) {
    items.push({ text: "This answer could not be checked right now", ok: false });
  }
  return items;
}
