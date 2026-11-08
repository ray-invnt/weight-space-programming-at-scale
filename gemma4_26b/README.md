# Gemma 4 26B: Fable's pruning

Fable's run that chose which experts each layer of Gemma 4 26B-A4B keeps, the keep lists of every pruned model the
post compares, and every exam and instruction answer graded.

| File | What it is |
|---|---|
| `prompt.txt` | the task Fable received |
| `steps.jsonl` | all 186 steps in order (105 Python runs, 81 test runs, one of which failed and was not counted), each with the explanation, the code or the request, and the output cut at 2,000 characters. Practice texts inside requests are replaced by a count |
| `final_answer.md` | Fable's final message |
| `audit.txt` | the verdict of the check of every shell command Fable ran |
| `keep/` | for each of the 30 layers, the original ids of the experts kept, in `frequency_44.json` (frequency pruning), `reap_44.json` (REAP pruning), `random_44.json`, `frequency_64.json` (the 64-expert starting point) and `fable_mask.json` (Fable's pruning, 1,320 experts) |
| `expert_stats.csv` | per layer and expert, how often the router picked it on 2.5 million tokens of medical prompts (MedQA and MedMCQA training questions and PubMedQA texts) and its REAP score on the same tokens |
| `prune.py` | cuts `google/gemma-4-26B-A4B-it` down to a keep list |
| `score.py` | the grading rules |
| `scores.csv` | one row per question per model, `id,set,arm,correct,cut`. Sets `medxpertqa` (2,455 questions of MedXpertQA Text) and `ifeval` (541 prompts). Arms `full`, `frequency`, `reap`, `random` and `fable`. `random` was scored on 1,005 of the exam questions |
| `examples.json` | the four exam answers the post quotes, for `full`, `frequency` and `fable` |

## The models

- `full`, the unpruned model, served with FP8 weights because the bf16 weights do not fit our GPU.
- `frequency`, `reap` and `random`, the model cut to the matching keep list with `prune.py`, bf16.
- `fable`, the 64-expert model (`keep/frequency_64.json`) with `keep/fable_mask.json` applied in the router. In each
  layer, an expert outside the mask gets a router logit of -1e9 before the top 8 are chosen, so it is never used.

The frequency lists are the experts with the highest `medical_count` in each layer, and the REAP list the ones with
the highest `reap_saliency`. Both counts come from prompts only, never from the model's answers, which is the blind
spot the post describes.

## Evaluation

The model's chat template with thinking off, the system message "You are a medical expert.", the question with its
lettered options and "Answer with the letter of the correct option, in the form 'Answer: X'.", temperature 1.0,
top_k 64, top_p 0.95, at most 4,096 new tokens, one answer per question, the same seed per question in every model.
`cut` is 1 when the answer ran to the 4,096-token limit. IFEval is prompt-level strict accuracy. Fable never saw
these questions. Its practice questions came from MedQA and MedMCQA training data, screened against the test sets
before the run (exact copies and shared 13-word passages) and checked again after, with no overlap found.

## The post's numbers

Accuracies are means of `correct` over one arm and set in `scores.csv`. Paired changes are the mean over questions
of one arm's `correct` minus frequency pruning's, with its standard error.

| In the post | Number | Where |
|---|---|---|
| "from 52 GB to 22 GB" | 51.6 GB and 21.6 GB of bf16 weights | `prune.py` with `keep/frequency_44.json` |
| "We kept 44 of 128", 1,320 experts | | `keep/frequency_44.json` |
| "It scored 12 points lower on a hard medical exam" | 45.9 against 33.5 | `full`, `frequency`, `medxpertqa` |
| "picked at random scores 26 points lower" | 9.0 against 34.8, on the 1,005 questions `random` was scored on | `random`, `frequency` |
| REAP pruning "10 points lower" on the exam and on following instructions | 23.4 against 33.5, and 50.3 against 60.3 on IFEval | `reap`, `frequency` |
| REAP "keeps 52% of the routing medical text uses, against 87% for the count" | 52.3% and 86.9% | `expert_stats.csv` and the keep lists, as the share of `medical_count` kept in each layer, averaged over the layers |
| Figure 9, exam | 45.9, 33.5, 23.4, 37.2 | `full`, `frequency`, `reap`, `fable` |
| Figure 9, answers stuck looping | 0.1%, 5.7%, 7.3%, 2.8% | mean of `cut` |
| "3.7 points higher", "half the looping" | +3.7 (SE 0.9) | `fable` minus `frequency` |
| "turns 270 of frequency pruning's wrong answers right and 179 right answers wrong, a net 91" | 270 and 179 | the same pairs |
| "more in the first layers and fewer in the middle" | 60 in each of layers 0 to 2, 38 to 42 in layers 10 to 26 | `keep/fable_mask.json` |
| "stopped 89 of 131 loops on practice questions" | | `steps.jsonl`, search for "131" |
| "80 test runs (it used all 80)" | | `steps.jsonl`, kind `test_run` |
| The four exam questions | Text-1537, Text-328, Text-2209, Text-1859 | `examples.json` |
