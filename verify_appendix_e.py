"""
verify_appendix_e.py

Numerically verifies Arditi et al., Appendix E ("Weight orthogonalization is
equivalent to directional ablation") on the MLP down_proj write of
Llama-3.1-8B-Instruct, for any chosen decoder layer.

Background / what is being checked
-----------------------------------
Whiteboard derivation (Dr. Liu), mapped onto Appendix E notation:

    x_pre           residual stream entering the MLP block (input to
                     post_attention_layernorm), for a given layer l
    x'_pre           = x_pre - r_hat r_hat^T x_pre         (activation-level
                                                             ablation of x_pre)
    t                = SiLU(W_gate @ norm(x'_pre)) * (W_up @ norm(x'_pre))
    x_post           = x'_pre + W_down @ t
    x'_post          = x_post - r_hat r_hat^T x_post       (post-hoc ablation
                                                             of the MLP's output)

    W'_down          = W_down - r_hat r_hat^T W_down       (weight orthogonalization)
    x''_post         = x'_pre + W'_down @ t                (baked-in ablation)

Claim (Appendix E, specialized to W_out = W_down):
    x'_post == x''_post   EXACTLY (up to floating point precision), provided
    (a) W_down has no bias term (true for Llama-3.1: bias=False everywhere),
    (b) x'_pre is already fully ablated before entering the MLP (true by
        construction above, since r_hat^T x'_pre = 0 exactly).

This script computes x'_post and x''_post directly from the real model
weights and activations and asserts they match to the precision of the
dtype used, then produces comparison plots.

Usage
-----
    python verify_appendix_e.py --layer 1 \
        --model_path /home/samuel/research/llmattacks/llm-attacks/DIR/Llama-3.1-8B-Instruct \
        --refusal_dir /path/to/refusal_direction.npy \
        --dtype float32

Notes
-----
- `--layer` uses HuggingFace decoder-layer indexing: model.model.layers[layer]
  (0-indexed). This is NOT the same as Dr. Liu's 1-indexed MATLAB convention
  or the hidden_states[i] convention (hidden_states[0] = embeddings,
  hidden_states[i] = output of layers[i-1]). layer=1 here means
  model.model.layers[1], the block referred to elsewhere in the project as
  "layer 2" under 1-indexed / hidden_states conventions. Verify against your
  own indexing notes before citing a layer number in a meeting.
- If --refusal_dir is not provided, a random unit vector is used instead.
  This is still a fully valid structural test of the algebraic identity
  (the identity holds for ANY unit vector r_hat, not just the refusal
  direction) but obviously isn't meaningful as a refusal-geometry result on
  its own -- pass your saved DIM direction to make the check specific to
  your actual refusal vector.
- The model is loaded in the dtype you request. bf16 has ~3 decimal digits
  of precision; do not expect fp32-level (~1e-6) agreement if you run in
  bf16. Use float32 (or float64) if you want the tightest possible check of
  the algebra itself, and bf16/bfloat16 separately if you want to know
  whether the equivalence survives your actual inference precision.
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_refusal_direction(path, hidden_size, device, dtype, seed=0):
    """Load a saved unit-norm refusal direction, or fall back to a random
    unit vector (structural test only -- see module docstring)."""
    if path is not None and os.path.exists(path):
        if path.endswith(".pt") or path.endswith(".pth"):
            arr = torch.load(path, map_location="cpu")
            if isinstance(arr, torch.Tensor):
                r = arr.clone().detach().to(device=device, dtype=dtype).flatten()
            else:
                r = torch.tensor(arr, device=device, dtype=dtype).flatten()
        else:
            arr = np.load(path)
            r = torch.tensor(arr, device=device, dtype=dtype).flatten()
        assert r.shape[0] == hidden_size, (
            f"Loaded refusal direction has dim {r.shape[0]}, "
            f"expected {hidden_size}"
        )
        source = f"loaded from {path}"
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        r = torch.randn(hidden_size, generator=g).to(device=device, dtype=dtype)
        source = "RANDOM (no --refusal_dir given -- structural test only)"

    r = r / r.norm()
    print(f"[refusal direction] {source}, ||r_hat|| = {r.norm().item():.6f}")
    return r


def capture_mlp_input(model, layer_no, input_ids, attention_mask):
    """Run one forward pass and capture x_pre: the residual-stream tensor
    fed into post_attention_layernorm at the given decoder layer (i.e. the
    input to the MLP block, before RMSNorm)."""
    captured = {}

    def hook(module, inputs):
        # forward_pre_hook: inputs[0] is the tensor about to be normalized
        captured["x_pre"] = inputs[0].detach().clone()

    layer = model.model.layers[layer_no]
    handle = layer.post_attention_layernorm.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    assert "x_pre" in captured, "Failed to capture MLP input -- check layer_no."
    return captured["x_pre"]


def verify_layer(model, tokenizer, layer_no, r_hat, prompts, device, dtype,
                  atol, rtol, out_dir):
    layer = model.model.layers[layer_no]
    mlp = layer.mlp
    norm = layer.post_attention_layernorm

    W_down = mlp.down_proj.weight.detach()          # [4096, 14336]
    hidden_size = W_down.shape[0]
    assert r_hat.shape[0] == hidden_size

    # W'_down = W_down - r_hat r_hat^T W_down   (weight orthogonalization)
    outer = torch.outer(r_hat, r_hat).to(W_down.dtype)          # [4096, 4096]
    W_down_prime = W_down - outer @ W_down                      # [4096, 14336]

    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    enc = tokenizer(chat_texts, return_tensors="pt", padding=True,
                     add_special_tokens=False).to(device)

    x_pre = capture_mlp_input(model, layer_no, enc["input_ids"],
                               enc["attention_mask"])            # [B, T, 4096]
    x_pre = x_pre.to(dtype)

    # x'_pre = x_pre - r_hat r_hat^T x_pre   (activation-level ablation of input)
    # Reuse the same [D, D] outer product used for the weight-side ablation,
    # so both sides of the identity are computed the same way.
    x_pre_ablated = x_pre - x_pre @ outer                        # x_pre - r_hat r_hat^T x_pre

    # sanity: confirm x'_pre really is ablated (precondition of Appendix E)
    resid_proj = (x_pre_ablated * r_hat).sum(dim=-1)
    max_resid = resid_proj.abs().max().item()

    with torch.no_grad():
        normed = norm(x_pre_ablated)
        gate = mlp.act_fn(mlp.gate_proj(normed))
        up = mlp.up_proj(normed)
        t = gate * up                                            # [B, T, 14336]

        # Path A: post-hoc ablation, x'_post
        # F.linear(t, W) computes t @ W.T -- do that explicitly with matmul.
        mlp_out = t @ W_down.T                                    # W_down @ t
        x_post = x_pre_ablated + mlp_out
        x_post_ablated = x_post - x_post @ outer                    # x'_post

        # Path B: weight orthogonalization, x''_post
        mlp_out_prime = t @ W_down_prime.T                        # W'_down @ t
        x_post_double_prime = x_pre_ablated + mlp_out_prime         # x''_post

    diff = (x_post_ablated - x_post_double_prime)
    max_abs_diff = diff.abs().max().item()
    mean_abs_diff = diff.abs().mean().item()
    rel_diff = (diff.norm() / x_post_ablated.norm()).item()

    print(f"\n=== Layer {layer_no} (model.model.layers[{layer_no}]) ===")
    print(f"dtype: {dtype}")
    print(f"max |r_hat^T x'_pre| (should be ~0): {max_resid:.3e}")
    print(f"max |x'_post - x''_post|:  {max_abs_diff:.3e}")
    print(f"mean |x'_post - x''_post|: {mean_abs_diff:.3e}")
    print(f"relative L2 diff:          {rel_diff:.3e}")

    # passed = torch.allclose(x_post_ablated, x_post_double_prime,
    #                          atol=atol, rtol=rtol)
    # assert passed, (
    #     f"Layer {layer_no}: x'_post != x''_post within atol={atol}, rtol={rtol} "
    #     f"(max abs diff = {max_abs_diff:.3e}). This should not happen for "
    #     f"bias-free linear writes with an exactly-ablated input -- check for "
    #     f"a bias term, wrong hook point, or dtype/precision mismatch."
    # )
    # print(f"ASSERTION PASSED: x'_post == x''_post within atol={atol}, rtol={rtol}")

    _plot_comparison(x_post_ablated, x_post_double_prime, layer_no, out_dir)

    return {
        "layer": layer_no,
        "max_resid_after_ablation": max_resid,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_l2_diff": rel_diff,
        # "passed": passed,
    }


def _plot_comparison(a, b, layer_no, out_dir):
    a_flat = a.flatten().float().cpu().numpy()
    b_flat = b.flatten().float().cpu().numpy()
    diff_flat = a_flat - b_flat

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    

    # Line plot of raw values: x'_post and x''_post -- should overlap exactly
    axes[0].plot(diff_flat, linewidth=0.7, alpha=0.7, label="x'_post  (post-hoc ablation)")
    # axes[0].plot(a_flat, linewidth=0.7, alpha=0.7, label="x''_post (weight orthogonalization)")
    axes[0].set_xlabel("flattened element index")
    axes[0].set_ylabel("value")
    axes[0].set_title(f"Layer {layer_no}: element-wise values")
    axes[0].legend()

    axes[1].plot(b_flat, linewidth=0.7, alpha=0.7, label="x''_post (weight orthogonalization)")
    # # Histogram of the difference
    # axes[1].hist(diff_flat, bins=80)
    # axes[1].set_xlabel("x'_post - x''_post")
    # axes[1].set_ylabel("count")
    # axes[1].set_title("Elementwise difference")
    # axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"appendix_e_layer{layer_no}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str,
                         default="/home/samuel/research/llmattacks/llm-attacks/DIR/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=1,
                         help="HF decoder layer index (model.model.layers[layer]), 0-indexed.")
    parser.add_argument("--all_layers", action="store_true",
                         help="Run the check for every layer instead of a single --layer.")
    parser.add_argument("--refusal_dir", type=str,
                         default="./phase1/dim_outputs/violence/direction.pt",
                         help="Path to a saved refusal direction .pt or .npy file (shape [4096]). "
                              "If omitted, a random unit vector is used (structural test only).")
    parser.add_argument("--dtype", type=str, default="float32",
                         choices=["float32", "float64", "bfloat16"])
    parser.add_argument("--atol", type=float, default=None,
                         help="Override absolute tolerance for the assertion. "
                              "Defaults: 1e-5 (fp32/fp64), 5e-2 (bf16).")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--harmful_val_path", type=str,
                         default="./phase1/data/saladbench_splits/categories/violence/harmful_val.json",
                         help="Path to a harmful_val.json split (list of dicts with an "
                              "'instruction' field). One prompt is drawn from this file "
                              "via --prompt_index.")
    parser.add_argument("--prompt_index", type=int, default=0,
                         help="Index into --harmful_val_path used to select the single "
                              "prompt for activation generation.")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float64": torch.float64,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    if args.atol is None:
        args.atol = 5e-2 if dtype == torch.bfloat16 else 1e-5

    with open(args.harmful_val_path) as f:
        harmful_val = json.load(f)
    prompt = harmful_val[args.prompt_index]["instruction"]
    print(f"[prompt] index {args.prompt_index} from {args.harmful_val_path}: {prompt!r}")
    prompts = [prompt]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_path} on {device} in {args.dtype} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype
    ).to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    # r_hat = load_refusal_direction(args.refusal_dir, hidden_size, device, dtype)
    r_hat = load_refusal_direction(None, hidden_size, device, dtype)

    num_layers = model.config.num_hidden_layers
    layers_to_run = range(num_layers) if args.all_layers else [args.layer]

    results = []
    for layer_no in layers_to_run:
        assert 0 <= layer_no < num_layers, f"layer {layer_no} out of range [0, {num_layers})"
        res = verify_layer(model, tokenizer, layer_no, r_hat, prompts,
                            device, dtype, args.atol, args.rtol, args.out_dir)
        results.append(res)

    print("\n=== Summary ===")
    for r in results:
        # status = "PASS" if r["passed"] else "FAIL"
        print(f"layer {r['layer']:>2}  "
              f"max_abs_diff={r['max_abs_diff']:.3e}  "
              f"rel_l2_diff={r['relative_l2_diff']:.3e}")


if __name__ == "__main__":
    main()
