# Evaluation

One command produces every number, one file stores them, one report shows them.

```powershell
uv sync --group eval                                   # RAGAS and rouge-score, eval only
$env:SARAL_DATABASE_URL = "<a disposable test database>"
.\.venv\Scripts\python.exe -m eval.run_eval            # everything (about 15 minutes)
.\.venv\Scripts\python.exe -m eval.run_eval --only retrieval        # no model calls
.\.venv\Scripts\python.exe -m eval.run_eval --no-judge               # skip RAGAS
.\.venv\Scripts\python.exe -m eval.run_eval --report-only            # rebuild REPORT.md
```

| File | What it is |
|---|---|
| `run_eval.py` | the runner: drives the real API like the browser does |
| `metrics.py` | deterministic metrics (precision, recall, F1, nDCG, MRR, figure overlap, ROUGE-L) |
| `REPORT.md` | the generated report |
| `results/latest.json` | every per-question and per-turn record behind the report |
| `results/orchestrator_comparison.json` | agent vs the removed plan-and-execute pipeline, same 16 turns |
| `datasets/decks/` | the three reference decks (Attention, BERT, ResNet) |
| `datasets/retrieval/questions.json` | 42 labelled questions: gold slides and answer phrases (6 not answerable from the deck) |
| `datasets/agent_conversations.json` | 11 scripted conversations, 16 turns, with the acceptable tools per turn |
| `datasets/generation_tasks.json` | frozen tasks for human reference scripts |
| `human/` | instructions for human reference scripts and the three raters |

## What is measured, and how

- **Retrieval**: Precision@k, Recall@k, F1@k, nDCG@k (k = 1, 3, 5) and MRR. A
  passage is relevant when it contains a labelled answer phrase, so a chunking
  change cannot redefine correctness.
- **Answers**: the agent answers each labelled question in a fresh conversation.
  Citation precision and recall come from the slides of the cited bullets against
  the gold slides. RAGAS scores each answer with a judge model:
  - *faithfulness*: the judge splits the answer into statements and checks each
    against the deck; supported ÷ all;
  - *context precision*: are the cited bullets relevant to the reference answer,
    relevant ones ranked first;
  - *context recall*: is every statement of the reference answer found in the
    cited bullets.
- **Conversations**: right tool per turn, drafts saved, claims the checker
  supports, citation coverage, the source-overlap proxy (figures and embedding
  similarity between each cited sentence and its bullet), limits and requirements
  met, RAGAS faithfulness of each draft, latency and tokens.
- **Human metrics** (assignment): ROUGE-L and BERTScore against human-written
  scripts, and three raters' scores, read from `human/references/` and
  `human/ratings/` when people have supplied them. Never generated.

The RAGAS judge is `gemini-3.5-flash` (set `SARAL_JUDGE_MODEL`), a stronger model
than the `flash-lite` writer it judges, through Gemini's OpenAI-compatible endpoint.
Judge scores are model opinions: evidence, not ground truth.

