#!/usr/bin/env python3
"""Grade Gemma 4 26B answers the way scores.csv was graded.

    python3 score.py medxpertqa answers.jsonl              lines {"id", "answer", "key"}
    python3 score.py ifeval answers.jsonl --prompts input_data.jsonl      lines {"id", "answer"}

prints one CSV row per answer: id,correct

MedXpertQA (Text subset, ten options A to J): the letter is the last "Answer: X" or \\boxed{X} in the answer (any case
for the word, markdown and brackets allowed; "Answer: A 55-year-old" or "Answer: I think" is prose, not a letter),
else a last line that is only a letter ("B", "(B)", "B.", "**B**"). No letter counts as wrong. Every answer was
written with thinking off, at most 4,096 new tokens; cut = 1 in scores.csv when it hit that limit.

IFEval: prompt-level strict accuracy from the official checkers as ported in lm-eval 0.4 (pip install lm-eval
langdetect); --prompts is IFEval's input_data.jsonl. The checkers draw random numbers (langdetect, and a random
letter when a prompt counts "!"), so about 0.3 percent of rows can grade differently from scores.csv; this script
seeds them so its own output repeats.
"""
import argparse, json, random, re

_ANSWER = re.compile(r"(?i:answer)\s*(?:\*\*|__|\*)?\s*[:\uff1a]\s*(?:\*\*|__|\*)?\s*[\(\[]?\s*([A-Z])(?![A-Za-z0-9])"
                     r"(?!(?<=[AI])\s+[a-z0-9])")
_BOXED = re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{)?\s*\(?\s*([A-Z])(?![A-Za-z0-9])")
_BARE = re.compile(r"^[\*_#>\s]*[\(\[]?\s*([A-Z])\s*[\)\]]?[\.\:]?[\*_\s]*$")
LETTERS = set("ABCDEFGHIJ")


def letter(text):
    hits = [(m.start(), m.group(1)) for rx in (_ANSWER, _BOXED) for m in rx.finditer(text or "")]
    if hits:
        return max(hits)[1]
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    m = _BARE.match(lines[-1]) if lines else None
    return m.group(1) if m and m.group(1) in LETTERS else None


def ifeval_correct(prompt, answer):
    from langdetect import DetectorFactory
    from lm_eval.tasks.ifeval.utils import process_results
    DetectorFactory.seed = 0
    random.seed(0)
    doc = {k: prompt[k] for k in ("key", "instruction_id_list", "prompt", "kwargs")}
    return int(bool(process_results(doc, [answer])["prompt_level_strict_acc"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=["medxpertqa", "ifeval"])
    ap.add_argument("answers")
    ap.add_argument("--prompts")
    a = ap.parse_args()
    prompts = {str(p["key"]): p for p in map(json.loads, open(a.prompts))} if a.prompts else {}
    print("id,correct")
    for r in map(json.loads, open(a.answers)):
        if a.set == "medxpertqa":
            ok = int(letter(r["answer"]) == r["key"])
        else:
            ok = ifeval_correct(prompts[str(r["id"])], r["answer"])
        print(f'{r["id"]},{ok}')


if __name__ == "__main__":
    main()
