import { motion, useMotionTemplate, useMotionValue, useSpring } from "motion/react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export function Spotlight({ children, className, color = "rgba(212, 168, 83, 0.16)", size = 320 }: {
  children: ReactNode; className?: string; color?: string; size?: number;
}) {
  const x = useMotionValue(-size);
  const y = useMotionValue(-size);
  const sx = useSpring(x, { stiffness: 300, damping: 35 });
  const sy = useSpring(y, { stiffness: 300, damping: 35 });
  const background = useMotionTemplate`radial-gradient(${size}px circle at ${sx}px ${sy}px, ${color}, transparent 70%)`;
  return (
    <div
      className={cn("group relative overflow-hidden", className)}
      onPointerMove={(event) => {
        const box = event.currentTarget.getBoundingClientRect();
        x.set(event.clientX - box.left);
        y.set(event.clientY - box.top);
      }}
      onPointerLeave={() => { x.set(-size); y.set(-size); }}
    >
      <motion.div aria-hidden className="pointer-events-none absolute inset-0 rounded-[inherit]" style={{ background }} />
      {children}
    </div>
  );
}
