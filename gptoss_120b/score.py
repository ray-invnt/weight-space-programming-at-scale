#!/usr/bin/env python3
"""Grade gpt-oss-120b answers the way scores.csv was graded.

    python3 score.py gsm8k answers.jsonl          lines {"id", "answer", "key"}
    python3 score.py medxpertqa answers.jsonl     lines {"id", "answer", "key"}

prints one CSV row per answer: id,correct. "answer" is the raw completion (the analysis channel, then the final
channel after "assistantfinal"); only the final channel is graded.

GSM8K: the last "Answer: N" in the final channel, else its last number, equal to the key (the value after "####" in
GSM8K's answer, commas removed) within 1e-6. MedXpertQA (Text subset, ten options A to J): the last "Answer: X", else
the last \\boxed{X}, else the last "(X)", else a final letter not preceded by another letter. No answer counts as
wrong. Every answer was written at medium reasoning effort, at most 4,096 new tokens; cut = 1 in scores.csv when it
hit that limit.
"""
import argparse, json, re

ANSWER = re.compile(r"Answer\s*[:：]\s*\(?\s*([A-J])\b", re.I)
BOXED = re.compile(r"\\boxed\{\(?([A-J])\)?\}")
PAREN = re.compile(r"\(([A-J])\)")
NUMBER = re.compile(r"Answer\s*[:：]\s*\$?\s*(-?[\d,]*\.?\d+)")
ANY_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def final_channel(text):
    for mark in ("assistantfinal", "<|channel|>final<|message|>"):
        i = text.rfind(mark)
        if i >= 0:
            return text[i + len(mark):]
    return text


def letter(text):
    ft = final_channel(text)
    for rx in (ANSWER, BOXED, PAREN):
        m = rx.findall(ft)
        if m and m[-1].upper() in "ABCDEFGHIJ":
            return m[-1].upper()
    s = ft.strip()
    if s and s[-1].upper() in "ABCDEFGHIJ" and (len(s) == 1 or not s[-2].isalpha()):
        return s[-1].upper()
    return None


def number(text):
    ft = final_channel(text)
    m = NUMBER.findall(ft) or ANY_NUMBER.findall(ft)
    try:
        return float(m[-1].replace(",", "")) if m else None
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=["gsm8k", "medxpertqa"])
    ap.add_argument("answers")
    a = ap.parse_args()
    print("id,correct")
    for r in map(json.loads, open(a.answers)):
        if a.set == "gsm8k":
            key, got = float(str(r["key"]).split("####")[-1].strip().replace(",", "")), number(r["answer"])
            ok = int(got is not None and abs(got - key) < 1e-6)
        else:
            ok = int(letter(r["answer"]) == r["key"])
        print(f'{r["id"]},{ok}')


if __name__ == "__main__":
    main()
