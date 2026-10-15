# SARAL model prompts

These Python modules contain importable string constants with the reviewable
instructions sent to models. Provider adapters assemble trusted runtime data and
substitute the `{{UPPER_CASE}}` placeholders through `render_prompt()`.

| Template | Model responsibility |
| --- | --- |
| `agent.py` | The conversational agent: answer from the deck, or choose a tool. |
| `generation.py` | Produce grounded, audience-adapted communication artifacts. |
| `verification.py` | Check factual support and whole-deliverable quality. |
| `editing.py` | Propose changes only to authorized content units. |
| `visual_description.py` | Describe visible slide content without inference. |

Each module exports `PROMPT`; the package exports descriptive aliases such as
`GENERATION_PROMPT`. The renderer performs one substitution pass, rejects missing
or unexpected variables, and does not interpret placeholder-like text inside user
or source data.
JSON schemas and Pydantic/domain validation remain in code because they are
executable contracts rather than natural-language prompts.
