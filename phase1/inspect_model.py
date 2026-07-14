"""
inspect_model.py
================
Run from the notebook (after Cell 3 loads `model`) with:

    %run -i inspect_model.py

NOTE: the `-i` flag is required. Without it, %run executes the script in a
fresh namespace instead of the notebook's interactive namespace, so it will
not see `model` even though Cell 3 already ran.

Prints a detailed breakdown of the loaded HookedTransformer:
  1. Global config  (model.cfg)
  2. Embedding / unembedding matrices
  3. Per-layer components for every block:
       - Attention: W_Q, W_K, W_V, W_O (+ biases)  →  shapes & param counts
       - MLP      : W_in, W_gate, W_out (+ biases)  →  shapes & param counts
       - LayerNorm: ln1, ln2  (w, b)                →  shapes & param counts
       - Hook points (no parameters, names only)
  4. Final layer-norm (ln_final)
  5. Grand totals: parameter count, dtype, estimated GPU memory
"""

import torch

# ── Helpers ──────────────────────────────────────────────────────────────────
SEP  = '=' * 72
SEP2 = '-' * 72


def _shape_str(p):
    return 'x'.join(str(s) for s in p.shape)


# ── Expect `model` from notebook namespace ───────────────────────────────────
if 'model' not in globals():
    raise NameError(
        "`model` not found.  Run Cell 3 first to load the HookedTransformer, "
        "then run this script with `%run -i inspect_model.py` (the -i flag "
        "is required so it shares the notebook's namespace)."
    )

cfg = model.cfg

# ════════════════════════════════════════════════════════════════════════════
# 1. GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print('1 ▸ GLOBAL CONFIG  (model.cfg)')
print(SEP)

cfg_fields = [
    ('model_name',              cfg.model_name),
    ('dtype',                   cfg.dtype),
    ('d_model',                 cfg.d_model),
    ('d_head',                  cfg.d_head),
    ('n_heads',                 cfg.n_heads),
    ('n_key_value_heads (GQA)', getattr(cfg, 'n_key_value_heads', cfg.n_heads)),
    ('n_layers',                cfg.n_layers),
    ('d_mlp',                   cfg.d_mlp),
    ('d_vocab',                 cfg.d_vocab),
    ('n_ctx  (max sequence)',   cfg.n_ctx),
    ('act_fn',                  cfg.act_fn),
    ('normalization_type',      cfg.normalization_type),
    ('positional_embed_type',   getattr(cfg, 'positional_embedding_type', 'N/A')),
    ('use_attn_scale',          cfg.use_attn_scale),
    ('use_local_attn',          getattr(cfg, 'use_local_attn', False)),
    ('attn_types',              getattr(cfg, 'attn_types', None)),
    ('fold_ln',                 getattr(cfg, 'fold_ln', 'N/A')),
    ('center_writing_weights',  getattr(cfg, 'center_writing_weights', 'N/A')),
    ('center_unembed',          getattr(cfg, 'center_unembed', 'N/A')),
    ('device',                  str(cfg.device)),
]
for name, val in cfg_fields:
    print(f'  {name:<30s}: {val}')

# ════════════════════════════════════════════════════════════════════════════
# 2. EMBEDDING / UNEMBEDDING
# ════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('2 ▸ EMBEDDING & UNEMBEDDING MATRICES')
print(SEP)

embed_entries = [('embed.W_E    (token embed)', model.embed.W_E)]
if hasattr(model, 'pos_embed'):
    embed_entries.append(('pos_embed.W_pos (positional)', model.pos_embed.W_pos))
embed_entries += [
    ('unembed.W_U  (logit proj)', model.unembed.W_U),
    ('unembed.b_U  (logit bias)', model.unembed.b_U),
]
for label, param in embed_entries:
    print(
        f'  {label:<38s}: shape={tuple(param.shape)}'
        f'  ({_shape_str(param)})  params={param.numel():>12,}'
    )

# ════════════════════════════════════════════════════════════════════════════
# 3. PER-LAYER BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print(f'3 ▸ PER-LAYER BREAKDOWN  (blocks.0 … blocks.{cfg.n_layers - 1})')
print(SEP)

layer_param_counts = []

for l in range(cfg.n_layers):
    blk  = model.blocks[l]
    attn = blk.attn
    mlp  = blk.mlp

    # ── Attention ──────────────────────────────────────────────────────────
    attn_params = {}
    for pname in ['W_Q', 'W_K', 'W_V', 'W_O', 'b_Q', 'b_K', 'b_V', 'b_O']:
        if hasattr(attn, pname):
            attn_params[pname] = getattr(attn, pname)

    # ── MLP (handles both standard and gated / SwiGLU variants) ───────────
    mlp_params = {}
    for pname in ['W_in', 'b_in', 'W_gate', 'b_gate', 'W_out', 'b_out']:
        if hasattr(mlp, pname):
            mlp_params[pname] = getattr(mlp, pname)

    # ── Layer-norms ────────────────────────────────────────────────────────
    ln_params = {}
    for ln_name in ['ln1', 'ln2']:
        ln = getattr(blk, ln_name, None)
        if ln is not None:
            for pn in ['w', 'b']:
                if hasattr(ln, pn):
                    ln_params[f'{ln_name}.{pn}'] = getattr(ln, pn)

    # ── Hook points (zero params) ──────────────────────────────────────────
    hook_names = sorted(
        n for n, _ in blk.named_modules() if 'hook_' in n
    )

    # ── Subtotals ──────────────────────────────────────────────────────────
    attn_total = sum(p.numel() for p in attn_params.values())
    mlp_total  = sum(p.numel() for p in mlp_params.values())
    ln_total   = sum(p.numel() for p in ln_params.values())
    blk_total  = attn_total + mlp_total + ln_total
    layer_param_counts.append(blk_total)

    print(f'\n  Layer {l:2d}  (total params: {blk_total:,})')
    print(SEP2)

    # Attention
    print('  Attention:')
    for pname, p in attn_params.items():
        print(f'    {pname:<8s}  shape={str(tuple(p.shape)):<32s}  params={p.numel():>10,}')
    print(f'    {"── subtotal":<40s}  params={attn_total:>10,}')

    # MLP
    print('  MLP:')
    for pname, p in mlp_params.items():
        print(f'    {pname:<8s}  shape={str(tuple(p.shape)):<32s}  params={p.numel():>10,}')
    print(f'    {"── subtotal":<40s}  params={mlp_total:>10,}')

    # LayerNorms
    if ln_params:
        print('  LayerNorms:')
        for pname, p in ln_params.items():
            print(f'    {pname:<10s}  shape={str(tuple(p.shape)):<30s}  params={p.numel():>10,}')
        print(f'    {"── subtotal":<40s}  params={ln_total:>10,}')

    # Hook points
    if hook_names:
        print('  Hook points (no params):')
        for hn in hook_names:
            print(f'    >> {hn}')

# ════════════════════════════════════════════════════════════════════════════
# 4. FINAL LAYER-NORM
# ════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('4 ▸ FINAL LAYER NORM  (model.ln_final)')
print(SEP)
if hasattr(model, 'ln_final'):
    for pname, p in model.ln_final.named_parameters():
        print(f'  {pname:<8s}: shape={tuple(p.shape)}, params={p.numel():,}')
else:
    print('  (no ln_final — may have been folded into the unembedding)')

# ════════════════════════════════════════════════════════════════════════════
# 5. GRAND SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('5 ▸ PARAMETER SUMMARY')
print(SEP)

total_params   = sum(p.numel() for p in model.parameters())
total_params_B = total_params / 1e9

_dtype_bytes = {
    'torch.float16':  2,
    'torch.bfloat16': 2,
    'torch.float32':  4,
    'torch.float64':  8,
}
bytes_per_param = _dtype_bytes.get(str(cfg.dtype), 2)
mem_gib = total_params * bytes_per_param / (1024 ** 3)

print(f'  Total parameters : {total_params:>15,}  ({total_params_B:.3f} B)')
print(f'  Active dtype     : {cfg.dtype}')
print(f'  Bytes / param    : {bytes_per_param}')
print(f'  Est. GPU memory  : {mem_gib:.2f} GiB  (weights only, excludes activations)')

print()
print('  Per-layer parameter counts (bar scaled to largest layer):')
max_cnt = max(layer_param_counts) if layer_param_counts else 1
for l, cnt in enumerate(layer_param_counts):
    bar = '#' * int(cnt / max_cnt * 40)
    print(f'    Layer {l:2d}: {cnt:>12,}  {bar}')

print()
print(SEP)
print('Architecture inspection complete.')
print(SEP)
