import { AnimatePresence, motion } from "motion/react";
import { BookOpen, Moon, Plus, Sun } from "lucide-react";

import type { Theme } from "@/lib/theme";

import { Logo } from "./Logo";

export function TopBar({ deckLabel, onDeck, onNewChat, busy, theme, onToggleTheme }: {
  deckLabel: string | null; onDeck: () => void; onNewChat: () => void; busy: boolean;
  theme: Theme; onToggleTheme: (origin: { x: number; y: number }) => void;
}) {
  const next = theme === "dark" ? "light" : "dark";
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center justify-between gap-3 border-b border-hairline/70 bg-canvas/75 px-4 backdrop-blur-xl sm:px-6">
      <a href="/" aria-label="SARAL home" className="flex items-center gap-2.5" translate="no">
        <Logo variant="full" className="h-5 w-auto" />
      </a>
      <div className="flex min-w-0 items-center gap-2">
        {deckLabel && (
          <motion.button
            type="button"
            onClick={onDeck}
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ y: -1 }}
            className="flex min-w-0 items-center gap-2 rounded-full border border-hairline bg-surface-1 px-3 py-1.5 text-[13px] text-ink-muted transition-colors hover:border-hairline-strong hover:text-ink"
          >
            <BookOpen className="size-3.5 shrink-0 text-accent-text" />
            <span className="truncate">{deckLabel}</span>
          </motion.button>
        )}
        <motion.button
          type="button"
          onClick={onNewChat}
          disabled={busy}
          whileHover={{ y: -1 }}
          whileTap={{ scale: 0.96 }}
          className="flex shrink-0 items-center gap-1.5 rounded-full border border-hairline bg-surface-1 px-3 py-1.5 text-[13px] font-semibold text-ink-muted transition-colors hover:border-hairline-strong hover:text-ink disabled:opacity-50"
        >
          <Plus className="size-3.5" /><span className="max-sm:hidden">New chat</span>
        </motion.button>
        <motion.button
          type="button"
          onClick={(event) => {
            const box = event.currentTarget.getBoundingClientRect();
            onToggleTheme({ x: box.left + box.width / 2, y: box.top + box.height / 2 });
          }}
          aria-label={`Switch to ${next} theme`}
          title={`Switch to ${next} theme`}
          whileHover={{ y: -1 }}
          whileTap={{ scale: 0.9 }}
          className="relative flex size-9 shrink-0 items-center justify-center overflow-hidden rounded-full border border-hairline bg-surface-1 text-ink-muted transition-colors hover:border-hairline-strong hover:text-ink"
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={theme}
              initial={{ y: 14, rotate: -90, opacity: 0 }}
              animate={{ y: 0, rotate: 0, opacity: 1 }}
              exit={{ y: -14, rotate: 90, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              {theme === "dark" ? <Moon className="size-4" /> : <Sun className="size-4" />}
            </motion.span>
          </AnimatePresence>
        </motion.button>
      </div>
    </header>
  );
}
