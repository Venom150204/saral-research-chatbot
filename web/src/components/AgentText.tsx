import { Fragment, type ReactNode } from "react";

import { useSources } from "@/lib/sources";

import { Citation } from "./Citation";

const CITATIONS = /\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]/g;

function Inline({ text }: { text: string }) {
  const { sources, sourceIds } = useSources();
  const nodes: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(CITATIONS)) {
    nodes.push(<Bold key={`t${last}`} text={text.slice(last, match.index)} />);
    match[1].split(/\s*[,;]\s*/).forEach((sid) => {
      const chunkId = sourceIds[sid] ?? null;
      const label = (chunkId && sources.get(chunkId)?.location) || sid;
      nodes.push(<Citation key={`c${match.index}-${sid}`} chunkId={chunkId} label={label} />);
    });
    last = (match.index ?? 0) + match[0].length;
  }
  nodes.push(<Bold key={`t${last}`} text={text.slice(last)} />);
  return <>{nodes}</>;
}

function Bold({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\*\*[^*]+\*\*)/).map((piece, index) =>
        piece.startsWith("**") && piece.endsWith("**") && piece.length > 4
          ? <strong key={index} className="font-semibold text-ink">{piece.slice(2, -2)}</strong>
          : <Fragment key={index}>{piece}</Fragment>)}
    </>
  );
}

export function AgentText({ text, live }: { text: string; live?: boolean }) {
  const blocks = text.trim().split(/\n{2,}/);
  return (
    <div className="space-y-3 text-[15px] leading-7 text-ink-muted">
      {blocks.map((block, index) => {
        const lines = block.split("\n");
        const caret = live && index === blocks.length - 1;
        if (lines.every((line) => /^\s*[-*•]\s+/.test(line))) {
          return (
            <ul key={index} className="list-disc space-y-1 pl-5 marker:text-ink-tertiary">
              {lines.map((line, i) => (
                <li key={i}><Inline text={line.replace(/^\s*[-*•]\s+/, "")} />{caret && i === lines.length - 1 && <Caret />}</li>
              ))}
            </ul>
          );
        }
        return (
          <p key={index}>
            {lines.map((line, i) => (
              <Fragment key={i}>{i > 0 && <br />}<Inline text={line} /></Fragment>
            ))}
            {caret && <Caret />}
          </p>
        );
      })}
    </div>
  );
}

function Caret() {
  return <span aria-hidden className="ml-0.5 inline-block h-4 w-[2px] translate-y-[3px] animate-pulse rounded bg-accent" />;
}
