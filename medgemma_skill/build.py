"""Rebuild Fable's merge (run skillmove_run1) from the two public checkpoints and merge_recipe.json.

Needs torch and safetensors. Download both repos at the revisions in merge_recipe.json, then:
  python build.py --base <gemma-3-4b-it dir> --source <medgemma-4b-it dir> --out <dir>      # write the merged checkpoint
  python build.py --base ... --source ... --verify-only                                     # build in memory, check digests
Every tensor is rebuilt with the recipe's rule and the digest of the result is compared with the recipe's.
The output is written as one shard per 4 GiB with the base's config, tokenizer and processor files.
"""
import argparse
import hashlib
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
COPY = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json", "processor_config.json",
        "chat_template.json", "chat_template.jinja")
EARLY_MLP = re.compile(r"language_model\.model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$")


def scale_of(name):
    if name.startswith(("vision_tower.", "multi_modal_projector.")):
        return 1.0
    m = EARLY_MLP.match(name)
    if m and int(m.group(1)) <= 16:
        return 1.0
    if name.startswith("language_model."):
        return 0.5
    raise ValueError(f"tensor outside the recipe: {name}")


class Reader:
    def __init__(self, d):
        wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.idx, self.h = {k: os.path.join(d, v) for k, v in wm.items()}, {}

    def get(self, name):
        f = self.idx[name]
        if f not in self.h:
            self.h[f] = safe_open(f, framework="pt")
        return self.h[f].get_tensor(name)


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--recipe", default=os.path.join(HERE, "merge_recipe.json"))
    a = ap.parse_args()
    if not (a.out or a.verify_only):
        ap.error("--out or --verify-only")
    torch.set_grad_enabled(False)
    rec = json.load(open(a.recipe))["check"]
    base, src = Reader(a.base), Reader(a.source)
    names = sorted(base.idx)
    assert names == sorted(src.idx), "BASE and SOURCE tensor names differ"
    lines_m, lines_b, changed, diff_src, n1 = [], [], 0, 0, 0
    shard, size, sid, wmap = {}, 0, 0, {}

    def flush():
        nonlocal shard, size, sid
        if shard:
            fn = f"model-{sid:05d}.safetensors"
            save_file(shard, os.path.join(a.out, fn), metadata={"format": "pt"})
            wmap.update({k: fn for k in shard})
            shard, size, sid = {}, 0, sid + 1

    if a.out:
        os.makedirs(a.out, exist_ok=True)
    for i, n in enumerate(names):
        b, t = base.get(n), src.get(n)
        s = scale_of(n)
        d = (t.float() - b.float()).to(torch.bfloat16).float()
        m = (b.float() + s * d).to(torch.bfloat16).contiguous()
        lines_b.append(f"{n} {sha(b)}\n")
        lines_m.append(f"{n} {sha(m)}\n")
        changed += int((m.view(torch.int16) != b.view(torch.int16)).sum())
        if s == 1.0:
            n1 += m.numel()
            diff_src += int((m.view(torch.int16) != t.view(torch.int16)).sum())
        if a.out:
            shard[n] = m
            size += m.numel() * 2
            if size >= 4 << 30:
                flush()
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(names)}", flush=True)
    if a.out:
        flush()
        total = sum(os.path.getsize(os.path.join(a.out, f)) for f in set(wmap.values()))
        json.dump({"metadata": {"total_size": total}, "weight_map": wmap},
                  open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
        for fn in COPY:
            if os.path.exists(os.path.join(a.base, fn)):
                shutil.copy(os.path.join(a.base, fn), os.path.join(a.out, fn))
    dm = hashlib.sha256("".join(lines_m).encode()).hexdigest()
    db = hashlib.sha256("".join(lines_b).encode()).hexdigest()
    print(f"base digest {'OK' if db == rec['base_digest'] else 'DIFFERS (wrong base revision?)'}")
    print(f"merged digest {'OK' if dm == rec['merged_digest'] else 'DIFFERS'}: {dm}")
    print(f"entries changed vs base: {changed:,} (recipe: {rec['entries_changed_vs_base']:,})")
    print(f"tensors at scale 1.0: {diff_src:,} of {n1:,} entries differ from the source's own values (bf16 rounding)")
    raise SystemExit(0 if dm == rec["merged_digest"] else 1)


if __name__ == "__main__":
    main()
