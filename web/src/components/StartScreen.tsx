import { AnimatePresence, motion } from "motion/react";
import { FileUp, LoaderCircle, Quote, ShieldCheck, Sparkles } from "lucide-react";
import { useRef, useState } from "react";

import { AnimatedShinyText } from "@/components/ui/animated-shiny-text";
import { BorderBeam } from "@/components/ui/border-beam";
import { TextEffect } from "@/components/ui/text-effect";
import { cn } from "@/lib/utils";

import { Spotlight } from "./Spotlight";

const PROMISES = [
  { icon: Quote, title: "Every claim cites its source", text: "Slide, page or line, one click away." },
  { icon: ShieldCheck, title: "Checked before it is saved", text: "An independent model verifies each claim." },
  { icon: Sparkles, title: "Edit by talking", text: "“Make #2 less technical”, with old vs new." },
];

export function StartScreen({ onUpload, status, error, busy }: {
  onUpload: (file: File) => void; status: string; error: boolean; busy: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const choose = (file?: File | null) => {
    if (!file || busy) return;
    onUpload(file);
  };

  return (
    <main className="relative mx-auto flex min-h-[calc(100vh-56px)] w-full max-w-5xl flex-col items-center justify-center px-4 py-16 sm:px-6">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="mb-6 rounded-full border border-hairline bg-surface-1/70 px-3.5 py-1 backdrop-blur"
      >
        <AnimatedShinyText className="text-[12.5px] font-medium text-ink-subtle [--shiny-width:90px]">
          Research communication, grounded in your source
        </AnimatedShinyText>
      </motion.div>

      <TextEffect
        as="h1"
        per="word"
        preset="fade-in-blur"
        speedReveal={1.4}
        className="max-w-3xl text-balance text-center text-[40px] font-semibold leading-[1.05] tracking-[-0.035em] text-ink sm:text-[64px]"
      >
        Turn research into talks, scripts and posts you can trust
      </TextEffect>

      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.55, duration: 0.6 }}
        className="mt-5 max-w-2xl text-balance text-center text-[17px] leading-7 text-ink-subtle"
      >
        Upload a deck, paper or brief. Ask for a 90-second script for policymakers, a 7-slide talk with speaker notes,
        or tweet-sized abstracts, then refine them in conversation.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 18, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ delay: 0.75, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        className="mt-10 w-full max-w-2xl"
      >
        <Spotlight className="rounded-2xl">
          <label
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files?.[0]); }}
            className={cn(
              "relative flex cursor-pointer flex-col items-center gap-3 rounded-2xl border border-dashed px-6 py-12 text-center transition-colors duration-300 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-accent/60",
              dragging ? "border-accent/70 bg-accent/[0.07]" : "border-hairline-strong bg-surface-1/70 hover:border-accent/40",
              busy && "cursor-progress",
            )}
          >
            <input
              ref={input}
              type="file"
              className="sr-only"
              accept=".pptx,.pdf,.tex,.md,.markdown,.txt"
              onChange={(event) => { choose(event.target.files?.[0]); event.target.value = ""; }}
            />
            {busy && <BorderBeam size={160} duration={2.6} colorFrom="#d4a853" colorTo="#836879" borderWidth={1.5} />}
            <motion.div
              animate={dragging ? { y: -6, scale: 1.08 } : busy ? { rotate: 0 } : { y: [0, -4, 0] }}
              transition={dragging || busy ? { type: "spring", stiffness: 300, damping: 18 } : { duration: 3, repeat: Infinity, ease: "easeInOut" }}
              className="flex size-14 items-center justify-center rounded-2xl border border-hairline-strong bg-surface-2 text-accent-text shadow-[0_10px_40px_-12px_rgba(212,168,83,0.55)]"
            >
              {busy ? <LoaderCircle className="size-6 animate-spin" /> : <FileUp className="size-6" />}
            </motion.div>
            <div>
              <p className="text-[16px] font-semibold text-ink">{dragging ? "Drop it to start" : "Drop a file here, or click to choose"}</p>
              <p className="mt-1 text-[13.5px] text-ink-subtle">Decks: .pptx, .pdf, Beamer .tex · Papers and briefs: .pdf, .tex, .md, .txt</p>
            </div>
          </label>
        </Spotlight>
        <AnimatePresence mode="wait">
          {status && (
            <motion.p
              key={status.replace(/\d+s$/, "")}
              role="status"
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className={cn("mt-4 text-center text-[13.5px]", error ? "text-danger" : "text-ink-subtle")}
            >
              {status}
            </motion.p>
          )}
        </AnimatePresence>
      </motion.div>

      <div className="mt-14 grid w-full max-w-4xl gap-3 sm:grid-cols-3">
        {PROMISES.map(({ icon: Icon, title, text }, index) => (
          <motion.div
            key={title}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.95 + index * 0.1, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            whileHover={{ y: -3 }}
            className="rounded-xl border border-hairline bg-surface-1/60 p-4 backdrop-blur-sm"
          >
            <Icon className="mb-3 size-4.5 text-accent-text" />
            <p className="text-[14px] font-semibold text-ink">{title}</p>
            <p className="mt-1 text-[13px] leading-5 text-ink-subtle">{text}</p>
          </motion.div>
        ))}
      </div>
    </main>
  );
}
