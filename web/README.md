# SARAL interface

React 19, TypeScript, Tailwind CSS 4 and Motion, built by Vite into static files that
FastAPI serves (`backend/app/api.py`), so production is still one process.

```powershell
npm install          # dependencies pinned to exact versions
npm run dev          # http://localhost:5173, proxies /api to uvicorn on :8000
npm run build        # web/dist, served by FastAPI at /
```

Animated components come from Magic UI (border beam, shiny text, dot pattern, number
ticker) and Motion Primitives (text effect, text shimmer), copied into
`src/components/ui` through their shadcn registries (MIT). Every animation respects
the reader's reduced-motion setting.

| Folder | What it holds |
|---|---|
| `src/lib/api.ts` | API calls and the server-sent-event reader for agent turns |
| `src/lib/conversation.ts` | a reply as ordered parts (words, steps, draft, checks); replay after reload |
| `src/components/` | start screen, chat, draft card, citations, source drawer, composer |
