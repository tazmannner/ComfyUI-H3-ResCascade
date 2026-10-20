"""Pooled-KV attention for MiniMax H3 (token downsampling of keys/values, queries stay full resolution).

At 1 MP ~65% of an H3 step is attention, which grows with the square of the token count. Here every query
(video, audio, text) attends to:
  - every non-video row exactly (text, conditioning, references, audio)
  - the target video keys/values average-pooled over pool x pool tokens within each latent frame
Each pooled key stands for pool^2 real keys, so its logit gets +log(pool^2): the softmax mass of a region of
similar keys is kept (sum_i exp(q.k_i) ~= n exp(q.mean(k))) and text/audio rows keep their normal share.
The bias rides in one extra (zero-padded) head dimension so the fast mask-free kernels still run.
Queries, MLPs and outputs stay per token, so no pixel detail is thrown away the way a low-res pass does.
"""
import logging
import math
import os

import torch

import comfy.patcher_extension
import folder_paths

log = logging.getLogger("H3ResCascade")
PAD = 8  # extra head dims (keeps head_dim a multiple of 8 for the fused kernels)
DUMP_FLAG = os.path.join(folder_paths.get_output_directory(), "h3_rescascade_dump")  # debug only: "<sigma_max> <call_no>"


class _PoolPatch:
    def __init__(self, pool, sigma_hi, sigma_lo, bias):
        self.pool, self.sigma_hi, self.sigma_lo, self.bias = pool, sigma_hi, sigma_lo, bias
        self.installed = set()
        self.logged = False


def _pool_video(x, va, T, th, tw, p):
    b, h, _, d = x.shape
    v = x[:, :, va:].reshape(b, h, T, th // p, p, tw // p, p, d)
    return v.mean(dim=(4, 6), dtype=torch.float32).to(x.dtype).reshape(b, h, -1, d)


def _make_override(patch, previous):
    def call(func, q, k, v, heads, **kw):
        return func(q, k, v, heads, **kw) if previous is None else previous(func, q, k, v, heads, **kw)

    def override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
                 skip_output_reshape=False, **kwargs):
        to = kwargs.get("transformer_options") or {}
        layout = to.get("minimax_h3_layout")
        sigmas = to.get("sigmas")
        p = patch.pool
        ok = (mask is None and skip_reshape and not skip_output_reshape and layout is not None
              and q.shape[2] == layout.seq_len and q.shape == k.shape)
        if ok:
            va, vb, kind = layout.segments[-1] if hasattr(layout, "segments") else (None, None, None)
            T, lh, lw = layout.signature[1], layout.signature[2], layout.signature[3]
            th, tw = (lh + 1) // 2, (lw + 1) // 2
            ok = kind == "video" and vb == q.shape[2] and th % p == 0 and tw % p == 0 and (vb - va) == T * th * tw
        if ok and sigmas is not None:
            s = float(sigmas.flatten()[0])
            ok = patch.sigma_lo - 1e-4 <= s <= patch.sigma_hi + 1e-4
        if ok and sigmas is not None and os.path.exists(DUMP_FLAG):  # debug: dump one real attention call
            want = open(DUMP_FLAG).read().split()  # "<sigma_max> <call_no>"
            s = float(sigmas.flatten()[0])
            if s <= float(want[0]):
                patch.dump_n = getattr(patch, "dump_n", 0) + 1
                if patch.dump_n == int(want[1]):
                    torch.save({"q": q.half().cpu(), "k": k.half().cpu(), "v": v.half().cpu(), "va": va, "T": T,
                                "th": th, "tw": tw, "sigma": s}, DUMP_FLAG + ".pt")
                    os.remove(DUMP_FLAG)
                    log.info(f"PooledKV: dumped attention call {want[1]} at sigma {s:.3f}")
        if not ok or p == 1:  # pool 1 = dense through the override (overhead check)
            return call(func, q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)

        kp = torch.cat([k[:, :, :va], _pool_video(k, va, T, th, tw, p)], dim=2)
        vp = torch.cat([v[:, :, :va], _pool_video(v, va, T, th, tw, p)], dim=2)
        kw = dict(mask=None, attn_precision=attn_precision, skip_reshape=True, skip_output_reshape=False, **kwargs)
        if not patch.bias:
            out = call(func, q, kp, vp, heads, **kw)
        else:
            b, h, n, d = q.shape
            d2 = d + PAD
            # logits = q.k/sqrt(d) + bias:  q' = [q*sqrt(d2/d), 1, 0..], k' = [k, bias*sqrt(d2), 0..]
            qa = q.new_zeros(b, h, n, d2)
            qa[..., :d] = q * math.sqrt(d2 / d)
            qa[..., d] = 1.0
            ka = kp.new_zeros(b, h, kp.shape[2], d2)
            ka[..., :d] = kp
            ka[:, :, va:, d] = math.log(p * p) * math.sqrt(d2)
            vpa = vp.new_zeros(b, h, vp.shape[2], d2)
            vpa[..., :d] = vp
            out = call(func, qa, ka, vpa, heads, **kw)
            out = out.reshape(b, n, h, d2)[..., :d].reshape(b, n, h * d)
        if not patch.logged:
            patch.logged = True
            log.info(f"PooledKV: {T} latent frames x {th}x{tw} tokens, pool {p} -> {kp.shape[2]} of {k.shape[2]} keys "
                     f"(bias {'on' if patch.bias else 'off'})")
        return out

    return override


def _install(patch, transformer_options):
    cur = transformer_options.get("optimized_attention_override")
    if cur in patch.installed:
        return
    ov = _make_override(patch, cur)
    patch.installed.add(ov)
    transformer_options["optimized_attention_override"] = ov


class H3PooledKVAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "pool": ("INT", {"default": 2, "min": 1, "max": 4,
                             "tooltip": "keys/values averaged over pool x pool video tokens (2 -> 4x fewer video keys)"}),
            "sigma_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                    "tooltip": "pool only on steps with sigma_min <= sigma <= sigma_max"}),
            "sigma_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "log_bias": ("BOOLEAN", {"default": True,
                                     "tooltip": "give each pooled key the weight of the pool^2 keys it replaces"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "H3ResCascade"
    DESCRIPTION = ("Attention speed-up for MiniMax H3 at high resolution: queries stay full resolution, video keys/values "
                   "are pooled 2x2 per latent frame, so attention cost drops ~4x without rendering at low resolution.")

    def patch(self, model, pool, sigma_max, sigma_min, log_bias):
        p = _PoolPatch(pool, sigma_max, sigma_min, log_bias)
        m = model.clone()
        _install(p, m.model_options["transformer_options"])
        m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, "h3_pooled_kv",
                                lambda mp, timestep, model_options: _install(p, model_options["transformer_options"]))
        return (m,)
