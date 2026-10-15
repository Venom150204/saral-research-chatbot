# SARAL Research Chatbot

Upload a research deck and talk to it. SARAL answers questions with citations to
the exact bullets, and writes deliverables (talk scripts, slides with speaker notes,
tweet-sized abstracts, LinkedIn posts, summaries) for a chosen audience, length and
style. Every factual sentence is checked against the deck by an independent model
before it is saved; drafts are versioned, with targeted edits, visible changes, undo
and redo.

## How it works

![SARAL architecture](docs/architecture.png)

- **One agent** (LangChain `create_agent` on LangGraph) owns the conversation and
  decides what to do; the **tools guarantee how**: citations to real bullets, a
  figure check, limits measured by code, an independent claim checker, versioned and
  idempotent saves. Hard limits on model and tool calls per turn.
- **Retrieval** is bullet-level BM25 plus local `bge-base-en-v1.5` embeddings;
  decks are small, so the writer reads the whole deck and retrieval is used for
  lookup and citation attribution.
- **Models**: Gemini (`gemini-3.5-flash-lite` agent and writer, a different
  model as checker), chosen by measurement.

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/DECISIONS.md](docs/DECISIONS.md).

## Evaluation

[eval/REPORT.md](eval/REPORT.md) holds every measured result: retrieval
Precision/Recall/F1/nDCG/MRR, answer citation precision and recall, RAGAS
faithfulness / context precision / context recall, draft claim support, citation
coverage, the source-overlap proxy, limits and requirements met, latency and tokens,
and the measured comparison that replaced the earlier plan-and-execute pipeline.
One command regenerates it (see [eval/README.md](eval/README.md)).

## Run it

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker (PostgreSQL with
pgvector), a Gemini API key.

```powershell
uv sync
Copy-Item .env.example .env        # then set GEMINI_API_KEY and SARAL_DATABASE_URL
cd web; npm install; npm run build; cd ..   # the interface, built once (Node 22+)
.\.venv\Scripts\python.exe -m uvicorn backend.app.api:app --loop backend.app.event_loop:selector_loop_factory
```

For interface work, run `npm run dev` in `web/` as well and open <http://localhost:5173>;
it reloads on every change and passes API calls to the server on port 8000.

Open <http://127.0.0.1:8000>. (`--loop` selects the selector event loop that the
async PostgreSQL driver needs on Windows; it is harmless elsewhere.)

Tests (unit tests run without a database; storage tests run when
`SARAL_TEST_DATABASE_URL` points at a disposable database):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Deploy

One container on Google Cloud Run with a Neon PostgreSQL database, both within their
free allowances for a demo, or GitHub Codespaces with no card:
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Repository

```text
backend/app/agent/       the agent, its tools, streaming and the answer check
backend/app/workflow/    the write-check-save pipeline the write tool runs
backend/app/             ingestion, chunking, retrieval, generation, verification,
                         editing, storage, prompts, API
web/                     the chat UI: React, Tailwind, Motion (built to static files)
eval/                    run_eval.py, metrics, labelled datasets, REPORT.md
tests/                   one test file per layer
docs/                    architecture, design decisions, deployment, input formats
```

## Notes

- `.env`, credentials, uploaded decks and generated user content are never committed.
- Every number in `eval/REPORT.md` comes from a real run.
- The SARAL logo, colours and typeface are SARAL's (saral.democratiseresearch.in,
  Democratise Research), used because this module is built for SARAL.
- External resources are listed in [ATTRIBUTION.md](ATTRIBUTION.md).
