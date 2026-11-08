# qwen3_lora: one adapter from five public LoRA adapters (run l2)

**Task.** The base model is Qwen3 4B (`Qwen/Qwen3-4B-Instruct-2507`), kept frozen. Five public rank-64 LoRA adapters were
trained on it by others, one task each:
- Circa: indirect yes/no answers, 5 classes;
- MNIST as text (`mnist_ascii`): handwritten digits drawn in ASCII;
- Banking77: 77 customer intents;
- XNLI Spanish (`xnli_es`);
- Disfl-QA: disfluent reading comprehension.

Fable built one adapter that carries all five skills, by arithmetic on the five adapters' tensors. There was no
training. It had 30 test runs on practice questions (it used 24) and never saw a test set.

The plain sum of the five adapters loses Banking77: the digit adapter's update is 2 to 9 times larger than the others
and overrides it. Fable reweighted the adapters, halving the digit adapter's MLP part. It also projected each adapter so
that it responds less to the other tasks' inputs. Banking77 comes back.

## Files

| file | what it is |
|---|---|
| `prompt.txt` | the editor's task (the driver prompt from its TASK section on; author names in citations removed) |
| `steps.jsonl` | every executed step in order, 47 in all (23 `python`, 24 `test_run`): `step, kind, explanation, code` or `request`, `output` (cut at 2,000 characters) |
| `final_answer.md` | Fable's final message, verbatim |
| `audit.txt` | the run's audit verdict |
| `merge_recipe.json` | the five adapters (repos, revisions, file hashes, update norms), Fable's composite step by step with its exact weights, and the baselines' settings and practice-pool selection |
| `build.py` | builds Fable's composite, the plain sum or task arithmetic from the five public adapters as a PEFT adapter |
| `score.py` | the graders for the five tasks |
| `scores.csv` | one row per held-out question per arm: `id, set, arm, correct, cut, f1` (`f1` for Disfl-QA only) |

The arms in `scores.csv` are:
- `base`;
- `adapter_<task>`: each adapter alone, on its own task;
- `plain_sum`;
- `task_arithmetic_0.7`;
- `ties_d0.2_s1.0`;
- `dare_ties_p0.9`;
- `fable_l2`.

Every arm was scored on the same questions with the same prompts and greedy decoding, in one vLLM engine
(max_lora_rank 320). The rows are the clean held-out sets. The strict leakage rule dropped 11 test items before scoring,
and they have no rows:
- Circa: `circa-train-4863`, `-6433`, `-16204`, `-16538`, `-19520`, `-19615`, `-23123`, `-24934`, `-30091`;
- Disfl-QA: `disfl_qa-all-4177` and `-7596`.

Ids are `<task>-<public split>-<row>`.

## Number map (post number -> rows -> value)

Each score is the mean of `correct` times 100 over a task's rows. The average is the mean of the five task scores.

| post | rows | value |
|---|---|---|
| About 500 held-out questions per task | `set` counts | 491 / 500 / 500 / 500 / 498 (2,489) |
| Digit adapter's update 2 to 9 times larger | `merge_recipe.json` `adapters.*.delta_norm` | 5.891 vs 2.440 / 1.624 / 1.275 / 0.658: 2.4 to 9.0 times |
| Plain sum: Banking77 62 -> 41 | `banking77`, `adapter_banking77` vs `plain_sum` | 62.2 -> 40.6 (base 38.2) |
| Average 54 -> 58, Banking77 comes back | `plain_sum` vs `fable_l2` | 53.8 -> 58.0; Banking77 40.6 -> 61.4 |
| Averages: Fable 58.0, task arithmetic 58.9, plain sum 53.8, TIES 56.2, DARE 56.5 | `fable_l2`, `task_arithmetic_0.7`, `plain_sum`, `ties_d0.2_s1.0`, `dare_ties_p0.9` | 58.0 / 58.9 / 53.8 / 56.2 / 56.5 |
| Methods tuned on the same practice questions | `merge_recipe.json` `baselines.selection_on_practice` | 21 candidates on the 500 practice questions Fable's test runs used; 0.7 best (58.4) |
| TIES and DARE served as their closest rank-320 version | `merge_recipe.json` `baselines.*.served_as` | rank 320 keeps 93.1% (TIES) and 55.0% (DARE) of the update's energy |
| Figure 6, per-task raw scores | see the table below | |
| What Fable did (reweighting, halving the digit MLP part, blinding to other tasks' inputs) | `merge_recipe.json` `fable_l2`; `steps.jsonl` (the final build is step 47); `final_answer.md` | |

Figure 6 values (accuracy; Disfl-QA is exact match, and its token F1 is in `f1`):

| arm | Circa | MNIST as text | Banking77 | XNLI Spanish | Disfl-QA | average |
|---|---:|---:|---:|---:|---:|---:|
| base | 8.4 | 8.2 | 38.2 | 62.8 | 35.9 | 30.7 |
| each adapter alone | 86.6 | 33.6 | 62.2 | 74.6 | 48.4 | 61.1 |
| plain sum | 80.2 | 30.8 | 40.6 | 73.6 | 43.8 | 53.8 |
| task arithmetic 0.7 | 85.1 | 27.2 | 61.6 | 75.0 | 45.6 | 58.9 |
| TIES (d 0.2, s 1.0) | 81.5 | 24.8 | 64.6 | 65.8 | 44.4 | 56.2 |
| DARE-TIES (p 0.9) | 81.9 | 28.8 | 51.2 | 75.0 | 45.6 | 56.5 |
| Fable (run l2) | 83.7 | 30.4 | 61.4 | 72.6 | 42.0 | 58.0 |

## How the editor worked

- **Effort.** The LoRA run (l2) ran at effort "high" with adaptive thinking. This was verified from the API parameters
  recorded in the driver transcript.
- **Sandbox.** The editor's Python session had no GPU and no internet, and the kernel enforced both:
  - no CUDA device was visible;
  - Hugging Face was offline, proxies pointed at a dead port, and socket connect was blocked.

  The kernel could not read the test files. The editor's shell commands were logged and audited for forbidden reads
  and model runs.
- **Audit.** `PASS: 23 python blocks, 24 forward blocks, 24 of 30 passes, 0 FAIL, 0 WARN` (`audit.txt`).
- **Hardware.** Test runs and scoring ran on one 46 GB GPU.

## Rebuilding and scoring

To rebuild, download the five adapter repos at the revisions in `merge_recipe.json` into `<dir>/<task>/`, then run
`python build.py --adapters <dir> --method fable_l2 --out <out>`. Other methods are `plain_sum` and
`task_arithmetic --scale 0.7`. The script needs only torch and safetensors.

The sums are exact. Fable's composite uses SVD and QR, so a rebuild matches the scored adapter to float precision but not
bit for bit. `build.py --compare <dir>` reports the difference. On the original machine the rebuild came within 2.2e-3
relative per module of the scored adapter, with delta norm 5.574003 against the recorded 5.568922. Task arithmetic
0.7 and the plain sum rebuilt exactly.

Prompts are each adapter's own training format: the task instruction as the system turn, the item as the user turn,
and the Qwen chat template. Decoding is greedy, with 32 new tokens (64 for Disfl-QA). `score.py` grades the output.

**Leakage.** Each test set was checked against the 100 rows each adapter was trained on and against the practice
questions. Matches were exact text, or rare 13-word spans covering half the item with the same label.

**Not included.** The release leaves out question texts, generations, transcripts, logs, work folders and weights.
Placeholders: `<repo>` is the experiment repository, `<data>` its data folder and `<home>` the home directory.
