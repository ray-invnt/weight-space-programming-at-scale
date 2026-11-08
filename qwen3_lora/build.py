"""Build Fable's l2 composite, the plain sum or task arithmetic from the five public adapters (merge_recipe.json).

Needs torch and safetensors. Download the five adapter repos at the revisions in merge_recipe.json, then e.g.
  python build.py --adapters <dir with circa/ mnist_ascii/ banking77/ xnli_es/ disfl_qa/> --method fable_l2 --out composite/
  python build.py --adapters <dir> --method task_arithmetic --scale 0.7 --out ta_0.7/
  python build.py --adapters <dir> --method plain_sum --out sum/          (task arithmetic at scale 1.0)
  python build.py --adapters <dir> --method fable_l2 --compare <a scored adapter dir>   (per-module delta comparison)
The output is one PEFT LoRA adapter (rank 320, lora_alpha 320, so scaling 1) for Qwen/Qwen3-4B-Instruct-2507, loadable by
PEFT and vLLM (max_lora_rank 320). Fable's composite is stored in bf16 as scored; the sums in fp32 as scored. Fable's
composite uses SVD and QR, so a rebuild matches the scored file to float precision, not bit for bit.
"""
import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "base_model.model."
ADAPTERS = ["banking77", "circa", "disfl_qa", "mnist_ascii", "xnli_es"]       # Fable's composite: alphabetical
SUM_ORDER = ["circa", "mnist_ascii", "banking77", "xnli_es", "disfl_qa"]       # the scored sums' concatenation order
ATT = {"q_proj", "k_proj", "v_proj", "o_proj"}


def load(adir):
    """{module: (A [64, in] * alpha / r, B [out, 64])} in float32 for one public adapter."""
    cfg = json.load(open(os.path.join(adir, "adapter_config.json")))
    s = cfg["lora_alpha"] / cfg["r"]
    f = safe_open(os.path.join(adir, "adapter_model.safetensors"), "pt")
    out = {}
    for k in f.keys():
        if k.endswith(".lora_A.weight"):
            m = k[len(PREFIX):-len(".lora_A.weight")]
            out[m] = (f.get_tensor(k).float() * s, f.get_tensor(k.replace("lora_A", "lora_B")).float())
    return out


def weight(adapter, module):
    """Fable's per-adapter weights (from run l1, kept in l2)."""
    if adapter == "mnist_ascii":
        return 1.0 if module.rsplit(".", 1)[-1] in ATT else 0.5
    return {"circa": 0.9, "banking77": 1.1}.get(adapter, 1.0)


def fable_l2(F, k=64, protect=("mnist_ascii",)):
    """Input-direction blinding: project the other four tasks' input directions out of each adapter's A, then weight."""
    comp = {}
    for m in sorted(F[ADAPTERS[0]]):
        A = {a: F[a][m][0] for a in ADAPTERS}
        B = {a: F[a][m][1] for a in ADAPTERS}
        ash = torch.stack([A[a] for a in ADAPTERS]).median(0).values           # the shared initialization, estimated
        V = {}
        for a in ADAPTERS:
            d = A[a] - ash                                                       # the task-specific part dA
            V[a] = torch.linalg.svd(d, full_matrices=False)[2][:k].T             # its row space [in, k]
            A[a] = ash + 1.0 * d
        new = {}
        for a in ADAPTERS:
            U = torch.cat([V[b] for b in ADAPTERS if b != a], 1)
            if a in protect:
                U = U - V[a] @ (V[a].T @ U)                                      # keep mnist's own directions
            Q, R = torch.linalg.qr(U)
            Q = Q[:, R.diagonal().abs() > 1e-3]
            new[a] = weight(a, m) * (A[a] - (A[a] @ Q) @ Q.T)
        comp[m] = (torch.cat([new[a] for a in ADAPTERS], 0), torch.cat([B[a] for a in ADAPTERS], 1))
    return comp


def task_arithmetic(F, scale):
    """Exact weighted sum of the five deltas (PEFT add_weighted_adapter 'cat'): A = [scale * A_t], B = [B_t]."""
    return {m: (torch.cat([scale * F[a][m][0] for a in SUM_ORDER], 0), torch.cat([F[a][m][1] for a in SUM_ORDER], 1))
            for m in sorted(F[SUM_ORDER[0]])}


def delta_norm(comp):
    return sum(float(torch.trace((B.T.double() @ B.double()) @ (A.double() @ A.double().T))) for A, B in comp.values()) ** .5


def write(comp, out, dtype):
    os.makedirs(out, exist_ok=True)
    r = max(A.shape[0] for A, _ in comp.values())
    t = {}
    for m, (A, B) in sorted(comp.items()):
        t[f"{PREFIX}{m}.lora_A.weight"] = A.to(dtype).contiguous()
        t[f"{PREFIX}{m}.lora_B.weight"] = B.to(dtype).contiguous()
    save_file(t, os.path.join(out, "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
           "r": r, "lora_alpha": r, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
           "inference_mode": True, "target_modules": sorted({m.rsplit(".", 1)[-1] for m in comp}),
           "use_rslora": False, "use_dora": False}
    json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"), indent=1)


def compare(comp, ref_dir):
    ref = load(ref_dir)                                     # a composite dir has scaling 1 (lora_alpha = r)
    worst = 0.0
    for m, (A, B) in comp.items():
        Ar, Br = ref[m]
        D, Dr = B.double() @ A.double(), Br.double() @ Ar.double()
        worst = max(worst, float((D - Dr).norm() / Dr.norm()))
    print(f"compare with {ref_dir}: worst per-module relative delta difference {worst:.2e} over {len(comp)} modules")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--method", choices=["fable_l2", "plain_sum", "task_arithmetic"], required=True)
    ap.add_argument("--scale", type=float, default=0.7)
    ap.add_argument("--out")
    ap.add_argument("--compare")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    F = {n: load(os.path.join(a.adapters, n)) for n in ADAPTERS}
    if a.method == "fable_l2":
        comp, dtype = fable_l2(F), torch.bfloat16
    else:
        comp, dtype = task_arithmetic(F, 1.0 if a.method == "plain_sum" else a.scale), torch.float32
    print(f"{a.method}: {len(comp)} modules, rank {max(A.shape[0] for A, _ in comp.values())}, "
          f"delta norm {delta_norm(comp):.6f} (recipe: fable_l2 5.568922, plain_sum 6.870532, scale s: s * 6.870532)")
    if a.out:
        write(comp, a.out, dtype)
    if a.compare:
        compare(comp, a.compare)


if __name__ == "__main__":
    main()
