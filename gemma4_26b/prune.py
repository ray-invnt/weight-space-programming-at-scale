#!/usr/bin/env python3
"""Cut Gemma 4 26B-A4B down to the experts in a keep list (bf16, Hugging Face layout).

    pip install torch safetensors
    python3 prune.py --base gemma-4-26B-A4B-it --keep keep/frequency_44.json --out gemma4-26b-frequency-44

--base is a local copy of google/gemma-4-26B-A4B-it. The keep lists in keep/ give, for each of the 30 layers, the
original ids of the experts to keep; every layer must keep the same number (the architecture has one expert count).
Per layer, four tensors lose the dropped experts along their first axis:

    experts.gate_up_proj [128, 1408, 2816]   experts.down_proj [128, 2816, 704]
    router.proj.weight   [128, 2816]         router.per_expert_scale [128]

and text_config.num_experts becomes the kept count. Everything else is copied unchanged. Kept experts are renumbered
0..k-1 in ascending original order. Gemma 4 renormalises the router's softmax over the 8 experts it selects, so a token
whose 8 experts were all kept is routed and weighted exactly as before.

Fable's pruning (keep/fable_mask.json) keeps different numbers per layer, so it is applied as a routing mask over the
64-expert model (built from keep/frequency_64.json): in every layer, the router logit of an expert that is not in the
mask is set to -1e9 before the top 8 are chosen, so it is never selected.
"""
import argparse, json, os, re, shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT_AXIS = (".experts.gate_up_proj", ".experts.down_proj", ".router.proj.weight", ".router.per_expert_scale")
LAYER = re.compile(r"\.layers\.(\d+)\.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.base, "config.json")))
    tc = cfg["text_config"]
    keep = {int(l): sorted(v) for l, v in json.load(open(a.keep)).items()}
    assert sorted(keep) == list(range(tc["num_hidden_layers"])), "the keep list must name every layer"
    sizes = {len(v) for v in keep.values()}
    assert len(sizes) == 1, "every layer must keep the same number of experts (apply other lists as a routing mask)"
    k = sizes.pop()

    os.makedirs(a.out, exist_ok=True)
    index = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))["weight_map"]
    shards, buf, nbytes = [], {}, 0

    def flush():
        nonlocal buf, nbytes
        if buf:
            shards.append(list(buf))
            save_file(buf, os.path.join(a.out, f"part-{len(shards)}.safetensors"), metadata={"format": "pt"})
            buf, nbytes = {}, 0

    for shard in sorted(set(index.values())):
        with safe_open(os.path.join(a.base, shard), "pt") as f:
            for name in f.keys():
                if name.endswith(EXPERT_AXIS) and ".language_model." in name:
                    sl = f.get_slice(name)
                    ids = keep[int(LAYER.search(name).group(1))]
                    t = torch.cat([sl[e:e + 1] for e in ids]).contiguous()
                else:
                    t = f.get_tensor(name)
                buf[name] = t
                nbytes += t.numel() * t.element_size()
                if nbytes > 5e9:
                    flush()
    flush()
    weight_map, size = {}, 0
    for i, names in enumerate(shards):
        fn = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        os.replace(os.path.join(a.out, f"part-{i + 1}.safetensors"), os.path.join(a.out, fn))
        size += os.path.getsize(os.path.join(a.out, fn))
        weight_map.update({n: fn for n in names})
    json.dump({"metadata": {"total_size": size}, "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    tc["num_experts"] = k
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for fn in os.listdir(a.base):
        p = os.path.join(a.base, fn)
        if os.path.isfile(p) and not fn.endswith(".safetensors") and fn not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, os.path.join(a.out, fn))
    print(f"{a.out}: {k} experts in each of {len(keep)} layers")


if __name__ == "__main__":
    main()
