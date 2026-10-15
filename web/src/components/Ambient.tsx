import { motion, useScroll, useTransform } from "motion/react";

import { DotPattern } from "@/components/ui/dot-pattern";
import { cn } from "@/lib/utils";

export function Ambient({ intensity = 1 }: { intensity?: number }) {
  const { scrollY } = useScroll();
  const drift = useTransform(scrollY, [0, 800], [0, -60]);
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <motion.div style={{ y: drift, opacity: intensity }} className="absolute inset-0">
        <div className="absolute -left-[15%] -top-[20%] h-[70vh] w-[70vw] animate-aurora rounded-full bg-[radial-gradient(closest-side,var(--aurora-a),transparent)] blur-3xl" />
        <div className="absolute -right-[20%] top-[10%] h-[65vh] w-[60vw] animate-aurora rounded-full bg-[radial-gradient(closest-side,var(--aurora-b),transparent)] blur-3xl [animation-delay:-6s] [animation-direction:alternate-reverse]" />
        <div className="absolute bottom-[-30%] left-[20%] h-[60vh] w-[60vw] animate-aurora rounded-full bg-[radial-gradient(closest-side,var(--aurora-a),transparent)] opacity-50 blur-3xl [animation-delay:-11s]" />
      </motion.div>
      <DotPattern
        width={22}
        height={22}
        cr={0.9}
        className={cn(
          "fill-[var(--dot)] [mask-image:radial-gradient(70%_60%_at_50%_35%,black,transparent)]",
        )}
      />
      <div className="absolute inset-0 bg-[radial-gradient(120%_80%_at_50%_0%,transparent,var(--vignette))]" />
    </div>
  );
}
