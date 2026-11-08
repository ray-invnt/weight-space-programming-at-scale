# TranslateGemma 4B: keeping a translator in the language it is asked for

Fable's run on TranslateGemma 4B, its one-tensor edit, and every held-out translation scored before and after.

| File | What it is |
|---|---|
| `prompt.txt` | the task Fable received |
| `steps.jsonl` | all 20 steps in order (11 Python runs, 9 test runs), each with the explanation, the code or the request, and the output cut at 2,000 characters. Practice sentences inside requests are replaced by a count |
| `final_answer.md` | Fable's final message |
| `audit.txt` | the verdict of the check of every shell command Fable ran |
| `build.py` | rebuilds the edited model from `google/translategemma-4b-it` (revision `10042cb0e6e7fdce748996a71dc3dc432a4e0c89`) and checks the edited tensor against the model the post scored |
| `score.py` | slip, chrF and COMET for one sentence |
| `scores.csv` | one row per sentence per model, `id,language,set,arm,slip,chrf,comet,cut`. Arms `translategemma` and `fable_edit`. 2,091 TICO-19 test sentences per language (the 9 that overlap the development set are left out) and 1,012 FLORES-200 devtest sentences each for Georgian, Sinhala and Lao |
| `examples.json` | the translations the post quotes |
| `measured.json` | the numbers behind the post's measured figure, which are how alike the writing systems' average rows are, before and after, and the next-word probabilities at two slips |

## The post's numbers

Shares are means of a column over one language and arm in `scores.csv`. Changes are the edited mean minus the
unedited one.

| In the post | Number | Where |
|---|---|---|
| Amharic translations that slip | 81% to 7% | `slip`, language `am` |
| Figure 5, slips | Tigrinya 65% to 4%, Khmer 95% to 62%, Georgian 64% to 6%, Sinhala 46% to 2%, Burmese 66% to 10%, Lao 99% to 100% | `slip` |
| Figure 5, change in COMET | Amharic +7.5, Tigrinya +6.4, Khmer +5.7, Georgian +7.3, Sinhala -1.9, Burmese -0.9, Lao -5.0 | `comet` |
| "1 to 2 points of chrF in Hindi and Chinese and 1 to 2 points of COMET in Burmese and Sinhala" | chrF -1.2 and -1.6, COMET -0.9 and -1.9 | `chrf`, `comet` |
| "Hindi, Chinese and Spanish almost never slip" | 0.4%, 1.5%, 0.4% | `slip`, arm `translategemma` |
| Amharic has 847 rows, Russian and Hindi more than 13,000 each | 847, 13,397, 13,638 | printed by `build.py` |
| 21 non-Latin writing systems pushed at half strength, Latin left alone | 89,593 rows changed | `build.py` |
| "It used 9 of its 30 test runs" | 9 | `steps.jsonl`, kind `test_run` |
| "Its first try pushed every writing system, Latin included" | | `steps.jsonl`, steps 5 to 9 |
| Figure 4, similarity of Amharic's rows with Russian, Hindi, Chinese and Thai | 0.23 to 0.11, 0.25 to 0.09, 0.25 to 0.12, 0.23 to 0.10 | `measured.json`, `script_similarity` |
| Figure 4, the Russian word for "now" at 64%, all Amharic words 26% to 99.9% | | `measured.json`, case `CMU_1:8` |
| Khmer "level", 97% on the Thai spelling, 95% on the Khmer syllable after the edit | | `measured.json`, case `CMU_1:4` |
| The quoted translations | | `examples.json` |

## Setup

TranslateGemma's plain prompt, greedy decoding, at most 512 new tokens, in vLLM with bf16 weights. A translation
slips when it has a letter from a writing system that is neither the target's nor Latin. chrF is sacrebleu's
default, per sentence. COMET is `Unbabel/wmt22-comet-da`, times 100. Fable practiced on the 971 TICO-19
development sentences into Amharic and could run 40 English sentences into Burmese with no reference or score. It
never saw the test sentences or any other language's files.
