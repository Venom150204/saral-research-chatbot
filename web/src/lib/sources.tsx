
import { createContext, useContext } from "react";

import type { Source } from "./api";

export interface SourceContextValue {
  sources: Map<string, Source>;
  sourceIds: Record<string, string>;
  show: (chunkId: string) => void;
}

export const SourceContext = createContext<SourceContextValue>({
  sources: new Map(), sourceIds: {}, show: () => {},
});

export const useSources = () => useContext(SourceContext);

export function sourceLocation(source: Source): string {
  const label = source.location ? source.location[0].toUpperCase() + source.location.slice(1) : "Document";
  return source.image_description ? `${label}, image description` : label;
}

export function sourceGroup(source: Source): string {
  if (source.slide_number) return `Slide ${source.slide_number}`;
  if (source.page_number) return `Page ${source.page_number}`;
  return "Document";
}
