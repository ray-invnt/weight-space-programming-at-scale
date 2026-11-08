# Code and logs for "Weight-Space Programming at Scale"

This is independent research. All models and benchmarks used are public. The models built here are research artifacts. They are not part of any Doctronic product and are not intended for clinical use. Views are the author's own.

What backs the numbers and quotes in the blog post
[Weight-Space Programming at Scale](https://ray-invnt.github.io/blog/weight-space-programming-at-scale/). One folder
per section, in the post's order. Each folder's README maps the post's numbers to its files.

| Folder | Section | What is there |
|---|---|---|
| [`medgemma_skill/`](medgemma_skill) | MedGemma 4B into Gemma 3 4B | Fable's run, the merge recipe, scores, quoted answers |
| [`translategemma/`](translategemma) | TranslateGemma 4B | Fable's run, the edit, scores for 11 languages, quoted translations |
| [`gemma4_26b/`](gemma4_26b) | Gemma 4 26B | Fable's run, the keep lists and mask, expert counts, scores, quoted answers |
| [`qwen3_lora/`](qwen3_lora) | Qwen3 4B, five LoRA adapters | Fable's run, the merge recipe, scores for every method |
| [`gptoss_120b/`](gptoss_120b) | gpt-oss-120b | Fable's run, the masks, scores, quoted answers |

## In each folder

- `prompt.txt`, the task Fable received.
- `steps.jsonl`, every step Fable took, in order, with its explanation, its code or test-run request, and what
  came back, cut at 2,000 characters.
- `final_answer.md`, Fable's final message, and `audit.txt`, the verdict of the check made after the run.
- The edit in its smallest form (a recipe, a keep list, a mask or a script) and a script that rebuilds the edited
  model from the public base model.
- `scores.csv`, one row per test question per model, with the public dataset's question id.
- `examples.json`, the answers the post quotes.

## How Fable worked

Fable is Claude Fable 5.1 (`claude-fable-5-1`), used as an agent, at maximum reasoning effort in the MedGemma,
TranslateGemma and Gemma 4 runs and at high effort in the LoRA and gpt-oss runs. It acted one message at a time. Each message held its explanation and one block, either Python for a persistent session over the
model's weights or one request for the test-run tool. The Python session had no GPU and no internet. The test-run
tool ran the model, with Fable's edit, on practice questions Fable picked, and returned the scores. Training,
gradients and any look at the final tests were not allowed. Each run's `audit.txt` gives the verdict of the
check made after it. Each prompt describes the tools in full. The tool server itself is not part of
this release.

## Context Fable used (Figure 12)

The most context Fable held in one turn of each run, counted as input plus cached tokens from the run's session
log.

| Model | Tokens |
|---|---|
| Gemma 3 4B and MedGemma 4B | 272,828 |
| TranslateGemma 4B | 299,054 |
| Gemma 4 26B | 426,273 |
| gpt-oss-120b, pruned to 60B | 277,843 |

## Not included

- Model weights. Each folder's build script rebuilds its models from the public base models.
- Benchmark and practice texts. Rows carry each public dataset's own ids.
- Model answers beyond the ones the post quotes.
- Fable's session transcripts and work folders.
- Runs and results the post does not report.

Paths in the prompts and steps are shortened to placeholders. `<repo>` is the experiment repository, `<home>` the
home directory, `<data>` the data disk and `<tmp>` a temporary folder.

## License

Code under MIT, everything else under CC BY 4.0. See [`LICENSE`](LICENSE).
