# Tests

One file per layer of the system, run with one command:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

| File | Covers |
|---|---|
| `test_ingestion.py` | reading PPTX, PDF (decks and papers), LaTeX (Beamer and articles, equations kept whole) and summaries; pictures, chunks, written forms, re-upload |
| `test_retrieval.py` | BM25, embeddings, fusion, attribution candidates, evaluation metrics |
| `test_writing.py` | the writer contract, the write-check-save graph, limits, the figure check, prompts |
| `test_verification.py` | the independent claim checker and requirement verdicts |
| `test_editing.py` | edit targets, scoped edits, deltas, the editor |
| `test_agent.py` | the agent's tools, limits, empty-reply retry, streaming events, interrupts, final-only drafts, answer checks |
| `test_api.py` | HTTP API and PostgreSQL storage: sessions, ledger, versions, undo/redo, start-up |
| `decks.py` | small decks built in code for the tests (not a test file) |

Model calls are replaced by fakes, so the suite is fast and free. Storage tests run
only when `SARAL_TEST_DATABASE_URL` points at a disposable PostgreSQL database with
pgvector.
