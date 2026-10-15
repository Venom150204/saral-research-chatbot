import { AnimatePresence, motion } from "motion/react";
import { ChevronRight } from "lucide-react";
import { useState, type ReactNode } from "react";

export function Fold({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-t border-hairline/70 pt-2 first:border-t-0">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="group flex w-full items-center gap-1.5 py-1 text-left text-[13px] font-medium text-ink-subtle transition-colors hover:text-ink"
      >
        <motion.span animate={{ rotate: open ? 90 : 0 }} transition={{ type: "spring", stiffness: 400, damping: 30 }}>
          <ChevronRight className="size-3.5" />
        </motion.span>
        {title}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <div className="space-y-2 pb-2 pl-5 pt-1 text-[13px] leading-6 text-ink-subtle">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
