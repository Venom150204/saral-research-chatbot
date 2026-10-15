# Human evaluation inputs

The assignment asks for two things only people can provide:

1. **Human-written reference scripts** for three papers, used for ROUGE-L and
   BERTScore. Task sheet: [reference_script_tasks.md](reference_script_tasks.md).
2. **Three raters** scoring audience fit, factuality and helpfulness.
   Instructions: [rater_instructions.md](rater_instructions.md); score sheet template:
   [rating_sheet_template.csv](rating_sheet_template.csv).

The tasks themselves are frozen in `eval/datasets/generation_tasks.json` so the system,
the reference writers and the raters all work from the same requests and source decks.

`python -m eval.run_eval --only scripts` writes the system's scripts for those tasks
to `system_outputs/`, and builds `rater_pack.md` (shuffled items with neutral ids, so
raters cannot tell how a script was produced), the private key `rater_key.json`, and
`rating_sheet_template.csv` with one row per item. Each rater returns a filled copy as
`ratings/<rater>.csv`; human references go to `eval/datasets/references/`. The next
`python -m eval.run_eval --only references` scores them into the report.

Nothing in this folder is a result. Scores and references are recorded only when a
person has actually supplied them.
