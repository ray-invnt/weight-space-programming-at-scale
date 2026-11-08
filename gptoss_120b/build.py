"""Turn an expert list from masks/ into a pruned gpt-oss-120b checkpoint (openai/gpt-oss-120b, revision in README).

  python build.py --src <gpt-oss-120b dir> --mask masks/fable_finalized.json --out <dir>
  python build.py --src <gpt-oss-120b dir> --mask masks/superset_64.json --out <dir>       # the 60B starting copy

A mask is {"<layer>": [original expert ids]} for the 36 layers. For each layer the kept experts are sliced out of every
expert tensor and the router (MXFP4 blocks and scales copied bit for bit, so nothing is re-quantized); everything else is
copied unchanged. vLLM and transformers read one num_local_experts for all layers, so when the per-layer counts differ
the layers are padded to the largest count with dead experts (zero weights and biases, router weight zero, router bias
-1e4): never among the top 4, so the padded model computes what the unpadded one would. A mask file may carry
"router_bias_edits" {layer: {expert id: absolute router.bias value}} (the router-rebalance control); those values are
written into the sliced router bias.
How the scored models were run: every arm was the 64-expert superset (superset_64.json) with the arm's dropped experts
masked in the engine (router logit -1e9 before the top-4; gpt-oss renormalizes over the 4 it selects, so a masked
expert is deleted). Slicing the same experts out gives the same computation up to batch numerics.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT = ("mlp.experts.gate_up_proj_blocks", "mlp.experts.gate_up_proj_scales", "mlp.experts.gate_up_proj_bias",
          "mlp.experts.down_proj_blocks", "mlp.experts.down_proj_scales", "mlp.experts.down_proj_bias",
          "mlp.router.weight", "mlp.router.bias")
SHARD = 4_500_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    L, E, K = cfg["num_hidden_layers"], cfg["num_local_experts"], cfg["num_experts_per_tok"]
    raw = json.load(open(a.mask))
    keep = {int(k): sorted(int(x) for x in v) for k, v in raw.items() if k.isdigit()}
    edits = {int(l): {int(e): float(v) for e, v in d.items()} for l, d in raw.get("router_bias_edits", {}).items()}
    assert set(keep) == set(range(L)), f"the mask must list layers 0..{L - 1}"
    for l, ids in keep.items():
        assert len(set(ids)) == len(ids) >= K and 0 <= ids[0] and ids[-1] < E, f"bad expert list in layer {l}"
        assert set(edits.get(l, {})) <= set(ids), f"router_bias_edits name an expert not kept in layer {l}"
    kmax = max(len(v) for v in keep.values())
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.src):
        if (f.endswith(".json") and f != "model.safetensors.index.json") or f.endswith(".jinja") or f in ("LICENSE", "USAGE_POLICY"):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    out_map, shard, size, sid = {}, {}, 0, 0

    def flush():
        nonlocal shard, size, sid
        if shard:
            fn = f"model-{sid:05d}.safetensors"
            save_file(shard, os.path.join(a.out, fn), metadata={"format": "pt"})
            out_map.update({k: fn for k in shard})
            shard, size, sid = {}, 0, sid + 1

    by_file = {}
    for n, f in idx.items():
        by_file.setdefault(f, []).append(n)
    for f in sorted(by_file):
        with safe_open(os.path.join(a.src, f), "pt") as sf:
            for n in sorted(by_file[f]):
                t = sf.get_tensor(n)
                if n.startswith("model.layers.") and n.endswith(EXPERT):
                    l = int(n.split(".")[2])
                    t = t.index_select(0, torch.tensor(keep[l], dtype=torch.long))
                    if n.endswith("mlp.router.bias") and edits.get(l):
                        t = t.clone()
                        for slot, e in enumerate(keep[l]):
                            if e in edits[l]:
                                t[slot] = edits[l][e]
                    pad = kmax - len(keep[l])
                    if pad:
                        fill = -1e4 if n.endswith("mlp.router.bias") else 0
                        t = torch.cat([t, torch.full((pad,) + tuple(t.shape[1:]), fill, dtype=t.dtype)], 0)
                t = t.contiguous()
                nb = t.numel() * t.element_size()
                if size + nb > SHARD:
                    flush()
                shard[n] = t
                size += nb
    flush()
    cfg["num_local_experts"] = kmax
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in set(out_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": out_map},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    n_keep = sum(len(v) for v in keep.values())
    print(f"wrote {a.out}: kept {n_keep} of {L * E} experts (per layer {min(len(v) for v in keep.values())} to {kmax}, "
          f"padded to {kmax}), {total / 1e9:.1f} GB" + (f", {sum(len(d) for d in edits.values())} router biases set" if edits else ""))


if __name__ == "__main__":
    main()
