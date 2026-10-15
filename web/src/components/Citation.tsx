import { motion } from "motion/react";

import { sourceLocation, useSources } from "@/lib/sources";

export function Citation({ chunkId, label }: { chunkId: string | null; label: string }) {
  const { sources, show } = useSources();
  const source = chunkId ? sources.get(chunkId) : undefined;
  const title = source ? `${sourceLocation(source)}: ${source.text}` : `${label} (not in the source)`;
  return (
    <motion.button
      type="button"
      title={title}
      aria-label={`Source: ${title}`}
      onClick={() => chunkId && show(chunkId)}
      whileHover={{ y: -1 }}
      whileTap={{ scale: 0.94 }}
      className="mx-0.5 inline-flex translate-y-[-1px] items-center rounded-full border border-accent/30 bg-accent/10 px-2 py-px align-middle text-[11px] font-medium leading-5 text-accent-text transition-colors hover:border-accent/60 hover:bg-accent/20"
    >
      {label}
    </motion.button>
  );
}
