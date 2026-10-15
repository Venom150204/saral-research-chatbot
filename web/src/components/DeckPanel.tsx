import { AnimatePresence, motion } from "motion/react";
import { X } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";

import type { Source } from "@/lib/api";
import { sourceGroup } from "@/lib/sources";
import { cn } from "@/lib/utils";

export function DeckPanel({ open, title, sources, highlight, onClose }: {
  open: boolean; title: string; sources: Source[]; highlight: string | null; onClose: () => void;
}) {
  const body = useRef<HTMLDivElement>(null);
  const groups = useMemo(() => {
    const map = new Map<string, Source[]>();
    sources.forEach((source) => {
      const key = sourceGroup(source);
      map.set(key, [...(map.get(key) || []), source]);
    });
    return [...map.entries()];
  }, [sources]);

  useEffect(() => {
    if (!open || !highlight) return;
    const timer = setTimeout(() => {
      body.current?.querySelector(`[data-chunk="${CSS.escape(highlight)}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 260);
    return () => clearTimeout(timer);
  }, [open, highlight]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="scrim"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px] lg:hidden"
          />
          <motion.aside
            key="panel"
            aria-label="Source contents"
            initial={{ x: "100%", opacity: 0.6 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: "100%", opacity: 0.6 }}
            transition={{ type: "spring", stiffness: 320, damping: 34 }}
            className="fixed bottom-0 right-0 top-14 z-50 flex w-full max-w-md flex-col border-l border-hairline bg-surface-1/95 backdrop-blur-xl"
          >
            <div className="flex items-start justify-between gap-4 border-b border-hairline px-5 py-4">
              <div className="min-w-0">
                <p className="truncate text-[15px] font-semibold text-ink">{title}</p>
                <p className="mt-0.5 text-[12.5px] text-ink-subtle">Everything SARAL can cite. Citations point here.</p>
              </div>
              <button type="button" onClick={onClose} aria-label="Close"
                className="rounded-md p-1.5 text-ink-subtle transition-colors hover:bg-surface-3 hover:text-ink">
                <X className="size-4" />
              </button>
            </div>
            <div ref={body} className="flex-1 space-y-5 overflow-y-auto overscroll-contain px-5 py-4">
              {groups.length === 0 && <p className="text-[13px] text-ink-tertiary">The source contents could not be loaded.</p>}
              {groups.map(([label, items], groupIndex) => (
                <motion.div key={label} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: Math.min(groupIndex * 0.04, 0.4) }}>
                  <p className="mb-2 text-[11.5px] font-semibold uppercase tracking-[0.08em] text-ink-tertiary">{label}</p>
                  <div className="space-y-1">
                    {items.map((source) => (
                      <motion.p
                        key={source.chunk_id}
                        data-chunk={source.chunk_id}
                        animate={highlight === source.chunk_id
                          ? { backgroundColor: ["rgba(212,168,83,0.38)", "rgba(212,168,83,0.14)"] }
                          : { backgroundColor: "rgba(212,168,83,0)" }}
                        transition={{ duration: 1.2 }}
                        className={cn(
                          "rounded-md px-2 py-1 text-[13px] leading-5",
                          source.kind === "title" ? "font-semibold text-ink" : "text-ink-muted",
                          source.image_description && "italic text-ink-subtle",
                          highlight === source.chunk_id && "ring-1 ring-accent/50",
                        )}
                      >
                        {source.text}
                      </motion.p>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
