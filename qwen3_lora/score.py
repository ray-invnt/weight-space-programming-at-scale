"""Graders for the five LoRA tasks, as used for every row of scores.csv (greedy decoding, the adapters' own prompts).

  circa, mnist_ascii, banking77, xnli_es: the first integer in the output must equal the gold class id (the adapters were
      trained to answer with the class index); if the output has no integer, circa and xnli_es fall back to a label name
      at the start of the output (this fired on 9 circa rows over all scored arms; banking77's name fallback never fired
      and is not reproduced here).
  disfl_qa: the first line of the output, normalized (lowercase, no punctuation, no articles), must equal the normalized
      gold span; token F1 on the same normalization is reported beside it.
usage: python score.py rows.jsonl   (one {"axis", "generation", "gold"} per line; prints accuracy per axis)
"""
import json
import re
import string
import sys
from collections import Counter, defaultdict

INT = re.compile(r"-?\d+")
NAME_FALLBACK = {
    "xnli_es": {"entailment": 0, "neutral": 1, "contradiction": 2, "implicación": 0, "contradicción": 2},
    "circa": {"yes, subject to some conditions": 3, "conditional yes": 3, "neither": 2, "in the middle": 2, "other": 4,
              "yes": 0, "no": 1},
}


def extract_label(axis, text):
    s = text.strip()
    m = INT.search(s)
    if m:
        return str(int(m.group()))
    low = s.lower()
    for name, idx in sorted(NAME_FALLBACK.get(axis, {}).items(), key=lambda kv: -len(kv[0])):
        if low.startswith(name):
            return str(idx)
    return None


def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def token_f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    n = sum((Counter(p) & Counter(g)).values())
    if n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)


def grade(axis, generation, gold):
    if axis in ("circa", "mnist_ascii", "banking77", "xnli_es"):
        ext = extract_label(axis, generation)
        return {"extracted": ext, "correct": int(ext is not None and ext == str(gold))}
    if axis == "disfl_qa":
        pred = generation.strip().split("\n")[0].strip()
        return {"extracted": pred, "correct": int(norm(pred) == norm(gold)), "f1": token_f1(pred, gold)}
    raise KeyError(axis)


if __name__ == "__main__":
    acc = defaultdict(list)
    for line in open(sys.argv[1]):
        r = json.loads(line)
        acc[r["axis"]].append(grade(r["axis"], r["generation"], r["gold"])["correct"])
    for a, v in acc.items():
        print(f"{a}: {100 * sum(v) / len(v):.1f} ({len(v)} rows)")
