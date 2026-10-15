# Design decisions

**One agent, rules inside the tools.** A first version planned each message onto a
fixed menu and ran the plan. It could not react to a result, so a blocked draft just
stopped, and every turn waited for planner, writer and checker. One agent now chooses
what to do; the tools enforce how content is written, checked and saved. Both
versions were measured on the same scripted conversations (`eval/REPORT.md`).

**Cite pieces, not documents.** Citations point to a single bullet, page lines or an
equation with its location. A citation is only as useful as what it points to.

**Read the whole deck when it fits.** SARAL decks are small, so the writer gets every
piece and nothing is lost to a retrieval miss. Retrieval is used for long sources,
for `find_in_deck` and to attribute citations.

**Hybrid retrieval.** BM25 finds exact terms and figures; local embeddings
(`bge-base-en-v1.5`) find paraphrases. Reciprocal rank fusion combines them without
tuning score scales. Local embeddings keep ingestion free and private.

**Similarity is not support.** A high embedding score does not mean a passage
supports a claim. Support is judged per claim by a separate checker call that sees
only the claim and its cited pieces, plus an exact figure check in code.

**Save or block.** A draft that fails its checks gets one targeted rewrite and is
otherwise never saved. Limits stated by the user are measured in code, not judged by
a model.

**Versioned drafts.** Every save is a new version with its citations. Edits are
scoped to one part, checked, and can be undone or redone.

**Idempotent operations.** Each tool call is recorded by id before it writes, so a
retry or a resumed stream cannot save the same draft twice.

**One database.** PostgreSQL with pgvector stores documents, vectors, drafts,
conversation events and the agent's checkpoints, which keeps deployment to one
service and one database.

**Models by measurement.** Gemini flash-lite models for the agent and writer, a
different model for the checker where configured; chosen on latency and the
evaluation results.
