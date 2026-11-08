"""SLAKE and VQA-RAD grading as used for every VQA row of scores.csv (the other sets use standard graders, see README).

The model sees the image, then the question and "Answer with a single word or short phrase."; at most 128 new tokens.
Grading: the first line of the answer, a trailing period removed (and anything before "answer is:" if the model
restates the question); then lowercase, punctuation and the articles a/an/the removed. f1 = token F1 against the gold
answer under that normalization; correct = exact match under it.
usage: python score.py rows.jsonl   (one {"generation", "gold"} per line; prints mean F1 and exact match)
"""
import json
import re
import string
import sys
from collections import Counter


def normalize(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def extract(text):
    line = text.strip().split("\n")[0].strip().rstrip(".")
    m = re.search(r"answer\s*(?:is)?\s*:?\s*(.+)$", line, re.I)
    return (m.group(1) if m else line).strip()


def grade(generation, gold):
    ext = extract(generation)
    return {"extracted": ext, "f1": token_f1(ext, gold), "correct": int(normalize(ext) == normalize(gold))}


if __name__ == "__main__":
    rows = [grade(r["generation"], r["gold"]) for r in map(json.loads, open(sys.argv[1]))]
    print(f"F1 {100 * sum(r['f1'] for r in rows) / len(rows):.2f}  exact {100 * sum(r['correct'] for r in rows) / len(rows):.2f}  ({len(rows)} rows)")
