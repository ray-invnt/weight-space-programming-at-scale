# medgemma_skill: moving MedGemma's skill onto Gemma 3 (run skillmove_run1)

**Task.** Gemma 3 4B (`google/gemma-3-4b-it`) and MedGemma 4B (`google/medgemma-4b-it`) have the same architecture. Fable
read the exact difference between the two checkpoints and built Gemma 3 plus a chosen part of that difference. It had
no training and no gradients. It had a Python session over the three weight sets, 25 test runs on practice items (MedQA
and MedMCQA train, MMLU validation, practice scans from SLAKE's training split), and it never saw a test set. Run
skillmove_run1 is run t1 in the logs.

The result is in `merge_recipe.json`. The image encoder, the projector and the MLPs of layers 0 to 16 are copied whole.
Every other language-model tensor gets half of MedGemma's change.

## Files

| file | what it is |
|---|---|
| `prompt.txt` | the editor's task (the driver prompt from its TASK section on) |
| `steps.jsonl` | every executed step in order, 44 in all (19 `python`, 25 `test_run`): `step, kind, explanation, code` or `request`, `output` (cut at 2,000 characters) |
| `final_answer.md` | Fable's final message, verbatim |
| `audit.txt` | the run's audit verdict |
| `merge_recipe.json` | the edit: which tensors are copied whole and which are scaled by 0.5, the two public repos and revisions, and a digest of the result |
| `build.py` | rebuilds the scored model from the two public checkpoints and checks the digest |
| `score.py` | the SLAKE / VQA-RAD grader (token F1 and exact match) |
| `scores.csv` | one row per question, arm and seed: `id, set, arm, seed, correct, cut, f1, in_clean_set` |
| `examples.json` | the quoted answers only (opening scan and examples panel): `id, set, arm, answer, extracted, key, correct, tokens` |
| `audit_six_pieces.json` | the blind six-piece prediction (locked before scoring) and the measured SLAKE change of each piece |

The arms in `scores.csv` are `gemma3`, `medgemma` and `fable_merge`, plus `vision_tower_only` (Gemma 3 with only the image
encoder and projector change, all three seeds) and five `piece_*` arms (the audit's other pieces, SLAKE seed 0).
`in_clean_set` is `false` for test items that overlap the practice data. Every row of the full sets is kept. **The post
reports the clean set** wherever a set loses items. `f1` is filled for SLAKE and VQA-RAD only.

## Number map (post number -> rows -> value)

Each value is the mean of a column times 100 over the rows named. "3 seeds" means the mean of the per-seed means for
seeds 0, 1 and 2.

| post | rows (`set`, `arm`, filter) | value |
|---|---|---|
| Opening: SLAKE chest X-ray. Gemma 3 says "Bronchitis" (wrong), MedGemma and Fable say "Enlarged heart" (right) | `examples.json` id `slake-11968` | key `Cardiomegaly`; the post translates it as "Enlarged heart" |
| Card: MedQA 51 -> 63.5; MedGemma 63.6 | `medqa`, seed 0, `in_clean_set=true`, `correct` | 51.0 / 63.6 / 63.5 (n 1,250). Full set 51.3 / 63.7 / 63.7 |
| Matches MedGemma on MedQA | same | 63.5 vs 63.6 (99.4% of the gain) |
| Keeps 97% of the VQA-RAD gain | `vqa_rad`, seed 0, `f1` | (49.6 - 31.9) / (50.1 - 31.9) = 97.4% (n 735, no overlap) |
| Keeps 76% of the SLAKE gain | `slake`, 3 seeds, clean, `f1` | (58.3 - 34.4) / (65.9 - 34.4) = 75.9% (n 857). Full set 76.5% |
| 5 more GSM8K points than MedGemma | `gsm8k`, seed 0 | 80.3 vs 75.3 (n 1,319, no overlap) |
| The image encoder alone makes Gemma 3 worse at scans (SLAKE 34 -> 20) | `slake`, `vision_tower_only` vs `gemma3`, 3 seeds, clean, `f1` | 34.4 -> 20.0. Full set 34.2 -> 19.8 |
| Figure 2, VQA-RAD | `vqa_rad`, seed 0, `f1` | 31.9 / 50.1 / 49.6 (full = clean) |
| Figure 2, SLAKE | `slake`, 3 seeds, clean, `f1` | 34.4 / 65.9 / 58.3 (n 857 per seed). Full set 34.2 / 63.5 / 56.6 |
| Figure 2, MedQA | `medqa`, seed 0, clean | 51.0 / 63.6 / 63.5 |
| Figure 2, IFEval | `ifeval`, 3 seeds, `correct` (prompt-level strict) | 74.7 / 76.3 / 80.3 (n 541, no overlap) |
| Figure 2, GSM8K | `gsm8k`, seed 0 | 88.7 / 75.3 / 80.3 |
| Figure 2, MMLU-Pro | `mmlu_pro`, seed 0, clean | 43.8 / 41.5 / 42.9 (n 11,954). Full set 43.7 / 41.4 / 42.9 |
| Runcard, MedXpertQA (text) | `medxpertqa_text`, seed 0, clean | 11.9 / 15.2 / 13.2 (n 2,408). Full set 11.9 / 15.4 / 13.5 |
| "The scan quiz and instruction scores average three runs" | SLAKE and IFEval have seeds 0 to 2 | VQA-RAD, MedQA, GSM8K, MMLU-Pro and MedXpertQA are one run (seed 0) |
| Six hidden pieces, all six predicted right | `audit_six_pieces.json`; `slake`, seed 0, arms `vision_tower_only` and `piece_*` minus `gemma3` | predicted -4 / +10 / +1 / +3 / -2 / -14; measured -14.3 / +21.2 / +3.0 / +18.6 / -3.0 / -13.4 (full set; same signs on the clean set): 6 of 6 signs |
| The skill lives in the first 17 layers | `steps.jsonl` test runs at steps 12 (image block + layers 0 to 16) and 14 (+ layers 17 to 33, which loops); then the half-strength soup (step 22), layers 0 to 16 whole on top of it (25), layers 0 to 6 (28), attention 0 to 16 only (31) and MLPs 0 to 16 only, the final choice (34); `final_answer.md` | |
| Each of the 34 layers changed by about the same amount | `steps.jsonl` step 1 output (per-layer relative change: MLPs 0.24 to 0.31; attention 0.21 to 0.58, the four spikes traced at steps 4 and 5 to a rescaling the norms cancel); `final_answer.md`, "a 0.25 to 0.37 relative change in every projection of every layer" | |
| The same early layers also hold most of the new medical knowledge | `final_answer.md`, "the medical text gain is carried mostly by the same early MLPs"; held-out test runs at steps 42 and 43 | |
| Examples panel: neck CT (Gemma 3 "Thyroid", 3 tokens; MedGemma and Fable "Spinal cord", 5 tokens), chest X-ray, cake, trains | `examples.json` ids `slake-12991`, `vqa_rad-864`, `gsm8k-test-436`, `gsm8k-test-16` | as quoted. SLAKE's own metadata files the neck CT under location "Abdomen" |
| Fold: 25 test runs, every step, final message | `steps.jsonl` (25 `test_run`), `prompt.txt`, `final_answer.md` | |
| Figure 12, peak context 272,828 tokens | the driver session's API usage records (not shipped) | the largest input of any turn (fresh + cache-write + cache-read tokens) |

## How the editor worked

- **Effort.** skillmove_run1 (t1) ran at effort "max". This was verified from the API parameters recorded in the
  driver transcript.
- **Sandbox.** The editor's Python session had no GPU and no internet, and the kernel enforced both:
  - no CUDA device was visible;
  - Hugging Face was offline, proxies pointed at a dead port, and socket connect was blocked.

  The editor's shell commands were logged and audited for forbidden reads and model runs.
- **Audit.** `PASS: 19 python blocks, 26 forward blocks, 25 of 25 passes, 0 FAIL, 0 WARN` (`audit.txt`).
- **Hardware.** Test runs and scoring ran on one 46 GB GPU.

## Rebuilding and scoring

To rebuild the model, download both repos at the revisions in `merge_recipe.json`, then run
`python build.py --base <gemma-3-4b-it> --source <medgemma-4b-it> --out <dir>`. The script needs only torch and
safetensors. It rebuilds all 883 tensors with
`MERGED = bf16(float32(BASE) + scale * float32(bf16(float32(SOURCE) - float32(BASE))))` and checks the digest of the
result. On the original machine it matched the evaluated checkpoint:
- the digest matched, and 4,259,637,526 entries changed versus Gemma 3;
- in the "copied whole" tensors, 185,155,270 of 1,756,750,704 entries differ from MedGemma's own values by bf16 rounding.

Evaluation used one protocol for every model:
- **Engine:** vLLM in bf16, the Gemma 3 chat template, one user turn, no system prompt.
- **Sampling:** temperature 1.0, top_k 64, top_p 0.95.
- **Seed:** for question i in seed run s, the seed is s x 1,000,003 + i.
- **Token limit:** 2,048 new tokens, or 128 for SLAKE and VQA-RAD.
- **Multiple choice:** the last "Final Answer: (X)" is graded.
- **GSM8K:** the "Final Answer: N" number must match exactly. If that line is missing, the last number is used.
- **IFEval:** the official checkers, prompt-level strict.
- **SLAKE / VQA-RAD:** `score.py`.

The sets are SLAKE English test (1,061), VQA-RAD balanced test (735), MedQA USMLE 4-option test (1,273), IFEval (541),
GSM8K test (1,319), MMLU-Pro test (12,032) and MedXpertQA text (2,450).

**Leakage.** Every test set was checked against the practice items:
- exact or near-duplicate text;
- a shared rare 13-word span with the same key;
- for images, pixel md5 and perceptual hash, checked by eye.

Dropped items:
- SLAKE loses 204 questions on 19 images. These are adjacent slices of practice scans, not identical images.
- MedQA loses 23 train/test variants.
- MMLU-Pro loses 78 items that duplicate MMLU validation.
- MedXpertQA loses 42 rewrites of MedQA train cases.

VQA-RAD, GSM8K and IFEval have no overlap.

**Not included.** The release leaves out benchmark and practice question texts and images, and generations other than
the quoted answers. Practice texts in step outputs read `[omitted: dataset text]`, and teacher-forced texts in requests
are replaced by a count. It also leaves out transcripts, logs, work folders and weights. Placeholders: `<repo>` is the
experiment repository, `<data>` its data folder and `<home>` the home directory.
