# Human reference scripts: task sheet

These scripts are the human-written references that system output is compared against
with ROUGE-L and BERTScore. They must be written by a person, under the same conditions
the system is given. Do not look at any system output before writing.

## What to write

One script per paper for the required task. That is three scripts in total.

| Paper | Source you may use | File to save |
|---|---|---|
| Attention Is All You Need | `eval/datasets/decks/attention.pptx` | `eval/datasets/references/attention__script90-policy.txt` |
| BERT | `eval/datasets/decks/bert.pptx` | `eval/datasets/references/bert__script90-policy.txt` |
| Deep Residual Learning (ResNet) | `eval/datasets/decks/resnet.pptx` | `eval/datasets/references/resnet__script90-policy.txt` |

Required task, word for word what the system receives:

> Make a 90-second plain-English script for policymakers explaining this research and
> its practical significance.

## Rules

1. Use only what the deck says. Do not add facts from the paper or from memory.
2. Write between 191 and 259 words. That is 90 seconds at 150 words per minute, within
   15 percent. The system is held to the same band.
3. Write words to be spoken aloud: no bullet points, no headings, no citations.
4. Plain text only, one script per file, nothing else in the file.
5. At the top of `eval/datasets/references/AUTHORS.md` record who wrote each script,
   the date, and roughly how long it took. Do not use an AI system to write or polish
   them; the assignment asks for human-authored references.

## Optional extra references

If there is time, the same three papers for the two other core tasks. Each is scored
only against system output for the identical task.

| Task id | Request |
|---|---|
| `script90-grad` | Make a 90-second technical script for graduate students focusing on the methods. |
| `script90-press` | Make a 90-second press-release style script for journalists announcing the main result. |

Save as `<paper>__<task id>.txt` in the same folder.
