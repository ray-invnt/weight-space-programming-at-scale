# gpt-oss-120b: a first look at 120B-class models

Fable's run that pruned gpt-oss-120b for medicine (run m2), the masks of every pruned model the post compares, and
every math and exam answer graded.

| File | What it is |
|---|---|
| `prompt.txt` | the task Fable received |
| `steps.jsonl` | all 69 steps in order (36 Python runs, 33 test runs), each with the explanation, the code or the request, and the output cut at 2,000 characters. Practice texts are replaced by a placeholder or a count. Step 64 was restored from the request records, because the tool log missed it |
| `final_answer.md` | Fable's final message |
| `audit.txt` | the verdict of the check made after the run |
| `masks/` | for each of the 36 layers, the original ids of the experts kept (below) |
| `build.py` | cuts `openai/gpt-oss-120b` (revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`) down to a mask |
| `score.py` | the grading rules |
| `scores.csv` | one row per question per model, `id,set,arm,correct,cut,in_clean_set,machine` |
| `examples.json` | the three answers the post quotes, for `frequency` and `fable_built_during_run` |

## The models

gpt-oss-120b has 36 layers of 128 experts, 4 used per token, 117 billion parameters in all. To fit our GPU, the
starting copy keeps the 64 experts per layer that medical prompts use most (`masks/superset_64.json`, 2,304
experts, 59.5 billion parameters). Each pruned model is a mask over the starting copy. In each layer, an expert
outside the mask gets a router logit of -1e9 before the top 4 are chosen, so it is never used.

| Arm | Mask | Experts |
|---|---|---|
| `starting_copy` | `superset_64.json` | 2,304 |
| `frequency` | `frequency.json`, the 32 per layer medical prompts use most | 1,152 |
| `reap` | `reap.json`, the 32 per layer with the highest REAP score | 1,152 |
| `fable_built_during_run` | `fable_built_during_run.json`, the version the post shows, built at step 39 | 1,152 |
| `fable_finalized` | `fable_finalized.json`, the more cautious version Fable finalized | 1,152 |
| `control_frequency_plus5` | `control_frequency_plus5.json`, frequency pruning with 5% more experts | 1,209 |
| `control_router_rebalance` | `control_router_rebalance.json`, the frequency list with router biases that rebalance the kept experts toward the unpruned model's choices | 1,209 |

## Evaluation

vLLM with the MXFP4 weights and the harmony chat template, temperature 1.0, top_k 64, top_p 0.95, one answer per
question, the same seed per question in every model. The set name gives the reasoning effort. Medium effort allows
4,096 new tokens and low effort 2,048. `cut` is 1 when the answer ran to that limit. Only the final channel is
graded. MedXpertQA has the system message "You are a medical expert." and ends with "Answer with the letter of the
correct option, in the form 'Answer: X'.". GSM8K ends with "Answer with the final number only, in the form
'Answer: N'.".

`machine` says which of our two machines scored the row (1 or 2, the second machine in the post). Paired changes
compare rows from the same machine and set.

After the runs, the practice and calibration texts were checked against the test sets. 22 of the 1,500 held-out exam
questions (8 of the 500 dev questions) rework MedQA training questions in them. Their rows have `in_clean_set`
false, and the post uses the others.

## The post's numbers

Accuracies are means of `correct` over one arm and set, on clean rows. Paired changes are the mean over questions of
one arm's `correct` minus frequency pruning's, with its standard error.

| In the post | Number | Where |
|---|---|---|
| Figure 10, starting copy | 88.4 on 500 math problems, 28.0 on 492 exam questions | `starting_copy`, `gsm8k500_medium` and `medxpertqa_dev500_medium` |
| "grade-school math falls from 88% to 2%" | 88.4 and 1.9 | `starting_copy`, and `frequency` on `gsm8k1319_medium`, machine 1 |
| Figure 10, math, frequency pruning and Fable's pruning | 1.9 and 72.9 | `frequency`, `fable_built_during_run`, `gsm8k1319_medium`, machine 1 |
| Figure 10, REAP's math, "scored on our second machine" | 70.1, +68.1 (SE 1.3) over frequency pruning's 2.0 there | `reap` and `frequency`, `gsm8k1319_medium`, machine 2 |
| Figure 10, exam, frequency pruning, REAP pruning and Fable's pruning | 21.5, 21.2 and 24.5 on 1,478 questions | `medxpertqa_heldout1500_medium`, machine 1 |
| "scores 3 points higher on a hard medical exam" | +3.0 (SE 1.1) | `fable_built_during_run` minus `frequency`, `medxpertqa_heldout1500_medium` |
| The finalized version, "2 points above frequency pruning" | +1.8 (SE 1.1), at low effort | `fable_finalized` minus `frequency`, `medxpertqa_heldout1500_low` |
| "Adding 5% more experts by frequency pruning, or rebalancing the router, still scores 1 to 2%" | 2.2 and 1.2 | the two `control_` arms, `gsm8k500_medium` |
| Figure 11, "REAP swaps 495 of them for others, and Fable swaps 289" | 495 and 289 | experts in `reap.json` and `fable_built_during_run.json` that are not in `frequency.json` |
| "looked about 5 points worse on the medical exam", why Fable finalized the cautious version | | `steps.jsonl` and `final_answer.md` |
| Math problem 1, key 38. Frequency pruning 15 in 155 tokens, Fable's pruning 38 in 68 | | `examples.json`, `gsm8k-test-1108` |
| Math problem 2 and the medical exam question | | `examples.json`, `gsm8k-test-179` and `Text-2039` |
