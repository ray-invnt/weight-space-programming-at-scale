#!/usr/bin/env python3
"""Rebuild Fable's edit of TranslateGemma 4B from the public weights.

One tensor changes, the tied embedding, which is also the output layer (the vocabulary table in the post). Every
vocabulary row is labelled with a writing system: the Unicode script of its letters (the first word of the letter's
Unicode name), "none" when it has no letters, "mixed" when its letters come from more than one script. Then

    mu[s]  = mean row of script s  -  mean of all 262,144 vocabulary rows
    mbar   = mean of mu[s] over the 22 scripts with at least 400 rows (Latin included; none, mixed, CUNEIFORM and
             MATHEMATICAL left out)
    row   += 0.5 * (mu[s] - mbar)          for every row of the 21 non-Latin scripts among those 22

Latin rows, rows without letters, mixed rows, the 64 padding rows past the vocabulary and every other tensor stay
TranslateGemma's, bit for bit. The arithmetic follows Fable's own code (steps 2, 5, 7 and 13 of steps.jsonl): sums
in float64 over chunks of 16,384 rows, the edit in float32, the result rounded to bfloat16.

    pip install torch safetensors tokenizers
    python3 build.py --model translategemma-4b-it --out translategemma-4b-it-edited

--model is a local copy of google/translategemma-4b-it. The output folder is that copy with the one tensor
replaced. The script prints the sha256 of the edited tensor's bytes and checks it against the model the post
scored.
"""
import argparse, collections, hashlib, json, os, shutil, unicodedata

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer

EMB = "language_model.model.embed_tokens.weight"
VOCAB = 262144
CHUNK = 16384
EXPECTED_SHA256 = "88775055fe9be59c77c3309c32e31746ac303d5e589006db6d62bc6053dbf923"


def script_of_token(text):
    c = collections.Counter()
    for ch in text.replace("▁", " "):
        if unicodedata.category(ch).startswith("L"):
            c[unicodedata.name(ch, "?").split()[0]] += 1
    if not c:
        return "none"
    (s, n), = c.most_common(1)
    return s if n == sum(c.values()) else "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    script = np.array([script_of_token(t) if (t := tok.id_to_token(i)) is not None else "none" for i in range(VOCAB)],
                      dtype=object)
    cnt = collections.Counter(script.tolist())
    scripts = [s for s, _ in cnt.most_common()]

    index = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    shard = index[EMB]
    with safe_open(os.path.join(a.model, shard), "pt") as f:
        E = f.get_tensor(EMB).float()                      # 262,208 x 2,560: the vocabulary plus 64 padding rows

    sums = {s: np.zeros(E.shape[1]) for s in scripts}
    for lo in range(0, VOCAB, CHUNK):
        hi = min(VOCAB, lo + CHUNK)
        rows, lab = E[lo:hi].numpy(), script[lo:hi]
        for s in set(lab.tolist()):
            sums[s] += rows[lab == s].sum(0, dtype=np.float64)
    gmean = sum(sums.values()) / VOCAB
    mu = {s: sums[s] / cnt[s] - gmean for s in scripts}

    big = [s for s in scripts if s not in ("none", "mixed", "CUNEIFORM", "MATHEMATICAL") and cnt[s] >= 400]
    mbar = np.mean([mu[s] for s in big], axis=0)
    edited_rows = 0
    for s in big:
        if s == "LATIN":
            continue
        ii = torch.as_tensor(np.where(script == s)[0])
        E[ii] += 0.5 * torch.as_tensor(mu[s] - mbar, dtype=torch.float32)
        edited_rows += len(ii)
    E = E.to(torch.bfloat16)
    digest = hashlib.sha256(E.view(torch.int16).numpy().tobytes()).hexdigest()
    print(f"{len(big)} scripts with at least 400 rows, {len(big) - 1} edited, {edited_rows:,} rows changed "
          f"(Ethiopic {cnt['ETHIOPIC']:,} rows, Cyrillic {cnt['CYRILLIC']:,}, Devanagari {cnt['DEVANAGARI']:,})")
    print(f"edited tensor sha256 {digest}, " + ("the same as the model the post scored" if digest == EXPECTED_SHA256
                                                 else "DIFFERENT from the model the post scored"))

    shutil.copytree(a.model, a.out, dirs_exist_ok=True)
    with safe_open(os.path.join(a.model, shard), "pt") as f:
        tensors = {k: (E if k == EMB else f.get_tensor(k)) for k in f.keys()}
        meta = f.metadata()
    save_file(tensors, os.path.join(a.out, shard), metadata=meta)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
