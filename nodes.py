"""H3 ResCascade: change the resolution of a MiniMax H3 render part-way through sampling.

A stage of SamplerCustomAdvanced that stops at sigma s (> 0) hands back
    output   = x_s / (1 - s)            (CONST flow model, inverse noise scaling)
    denoised = x0 predicted at the stage's last model call
With euler, x_s = (1 - s) * x0 + s * eps exactly, so eps = (output - denoised) * (1 - s) / s.
ResSwitch resizes x0 and eps separately and recomposes them at the new size, still at sigma s,
so the next stage (DisableNoise, sigmas starting at s) carries on along the same trajectory.

eps is resized so it stays unit-variance white noise:
  down by k: block-mean * k
  up by k:   nearest(eps) / k + (z - blockmean(z)) for fresh z  -> every block keeps its low-res mean
Audio (2nd stream of the AV latent) is passed through untouched.
"""
import sys
import types

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.nested_tensor
import folder_paths

VAE_DOWN = 16
LBH_FOLDER = "latent_upscale_models"


def _streams(latent):
    s = latent["samples"]
    if getattr(s, "is_nested", False):
        return list(s.unbind())
    return [s]


def _pack(streams):
    if len(streams) == 1:
        return streams[0]
    return comfy.nested_tensor.NestedTensor(streams)


def _spatial(x, size, mode):
    """resize the H, W of a [B, C, T, H, W] tensor"""
    b, c, t, h, w = x.shape
    if (h, w) == tuple(size):
        return x
    y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    if mode == "area":
        y = F.interpolate(y, size=size, mode="area")
    else:
        y = F.interpolate(y, size=size, mode=mode, align_corners=False)
    return y.reshape(b, t, c, size[0], size[1]).permute(0, 2, 1, 3, 4)


def _block_mean(x, kh, kw):
    b, c, t, h, w = x.shape
    return x.reshape(b, c, t, h // kh, kh, w // kw, kw).mean(dim=(4, 6))


def _repeat(x, kh, kw):
    return x.repeat_interleave(kh, dim=3).repeat_interleave(kw, dim=4)


def _resize_noise(eps, size, mode, gen):
    b, c, t, h, w = eps.shape
    H, W = size
    std = eps.float().std(dim=(0, 2, 3, 4), keepdim=True).to(eps.dtype)
    fresh = lambda: torch.randn((b, c, t, H, W), generator=gen, dtype=torch.float32).to(eps) * std
    if mode == "fresh":
        return fresh()
    if h % H == 0 and w % W == 0:  # integer downscale
        kh, kw = h // H, w // W
        return _block_mean(eps, kh, kw) * (kh * kw) ** 0.5
    if H % h == 0 and W % w == 0:  # integer upscale
        kh, kw = H // h, W // w
        z = fresh()
        return _repeat(eps, kh, kw) / (kh * kw) ** 0.5 + z - _repeat(_block_mean(z, kh, kw), kh, kw)
    # non-integer: interpolate, then renormalise to the original per-channel std (spatially correlated)
    y = _spatial(eps, size, "bilinear")
    return y / y.float().std(dim=(0, 2, 3, 4), keepdim=True).to(y.dtype) * std


def _lbh_module():
    for name, m in list(sys.modules.items()):
        if isinstance(m, types.ModuleType) and name.endswith("minimax_h3_latent_upscaler_3d") \
                and isinstance(getattr(m, "LatentResizer3D", None), type):
            return m
    raise RuntimeError("lbh_3d needs the Comfyui_Minimax_h3_latent_Upscaler pack (LBH-123-AI) installed")


def _lbh_upscale(x0, size, model_name):
    m = _lbh_module()
    dev = mm.get_torch_device()
    model = m.load_model(model_name, dev, "fp16")
    mean, std = m._make_norm_tensors(dev, torch.float16)
    t = x0.shape[2]
    scale = (size[0] / x0.shape[3] + size[1] / x0.shape[4]) / 2
    with torch.inference_mode():
        s = (x0.to(dev, torch.float16) - mean) / std
        out = model(s, scale=scale, target_size=(t, size[0], size[1]), enable_chunking=True)
        out = (out * std + mean).to(x0.device, x0.dtype)
    model.to("cpu")
    mm.soft_empty_cache()
    return out


class H3ResSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        lbh = folder_paths.get_filename_list(LBH_FOLDER) if LBH_FOLDER in folder_paths.folder_names_and_paths else []
        return {
            "required": {
                "noisy": ("LATENT", {"tooltip": "'output' of the stage that just stopped (sigma > 0)"}),
                "denoised": ("LATENT", {"tooltip": "'denoised_output' of the same stage"}),
                "sigmas": ("SIGMAS", {"tooltip": "that stage's sigmas; the last one is where it stopped"}),
                "width": ("INT", {"default": 1344, "min": 64, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
                "x0_resize": (["auto", "area", "bicubic", "bilinear", "lbh_3d"], {
                    "tooltip": "auto = area when shrinking, bicubic when growing. lbh_3d = LBH-123-AI's learned H3 latent "
                               "upscaler (growing only)"}),
                "noise": (["consistent", "fresh"], {
                    "tooltip": "consistent keeps the trajectory's own noise (block means preserved); fresh draws new noise"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "lbh_model": (lbh or ["(none)"],),
                "sigma_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                        "tooltip": "0 = recompose at the stage's last sigma. Higher = re-noise to this "
                                                   "sigma (next stage's sigmas must start here)"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "switch"
    CATEGORY = "H3ResCascade"

    def switch(self, noisy, denoised, sigmas, width, height, x0_resize, noise, seed, lbh_model=None, sigma_out=0.0):
        s = float(sigmas[-1])
        if s <= 0 or s >= 1:
            raise ValueError(f"ResSwitch needs a stage that stopped at 0 < sigma < 1, got {s}")
        so = sigma_out if sigma_out > 0 else s
        xs, ds = _streams(noisy), _streams(denoised)
        out_v, x0 = xs[0].float(), ds[0].float()
        size = (height // VAE_DOWN, width // VAE_DOWN)
        eps = (out_v - x0) * ((1 - s) / s)
        gen = torch.Generator().manual_seed(seed)

        mode = x0_resize
        grow = size[0] * size[1] > x0.shape[3] * x0.shape[4]
        if mode == "auto":
            mode = "bicubic" if grow else "area"
        if mode == "lbh_3d":
            if not grow:
                raise ValueError("lbh_3d only upscales")
            x0n = _lbh_upscale(x0, size, lbh_model)
        else:
            x0n = _spatial(x0, size, mode)
        epsn = _resize_noise(eps.cpu(), size, noise, gen).to(x0n)
        v = x0n + epsn * (so / (1 - so))
        print(f"[H3ResSwitch] {tuple(x0.shape[-2:])} -> {size} at sigma {s:.4f}->{so:.4f}  x0:{mode}  noise:{noise}  "
              f"eps std {eps.std().item():.3f}->{epsn.std().item():.3f}")
        return ({"samples": _pack([v.to(xs[0].dtype)] + xs[1:])},)


class H3LatentUpscaleLBH:
    """Clean H3 video latent -> LBH-123-AI's learned 3D latent upscaler at an exact pixel size (no VAE round trip)."""

    @classmethod
    def INPUT_TYPES(cls):
        lbh = folder_paths.get_filename_list(LBH_FOLDER) if LBH_FOLDER in folder_paths.folder_names_and_paths else []
        return {"required": {"samples": ("LATENT", {"tooltip": "video latent (AV latents: the video stream is used, "
                                                                "audio passed through)"}),
                             "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 32}),
                             "height": ("INT", {"default": 1088, "min": 64, "max": 8192, "step": 32}),
                             "lbh_model": (lbh or ["(none)"],)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "H3ResCascade"

    def upscale(self, samples, width, height, lbh_model):
        xs = _streams(samples)
        out = _lbh_upscale(xs[0].float(), (height // VAE_DOWN, width // VAE_DOWN), lbh_model).to(xs[0].dtype)
        return ({"samples": _pack([out] + xs[1:])},)


class H3ResScale:
    """Width/height helper: scale a size by a factor and snap to the 32 px token grid."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"width": ("INT", {"default": 1344, "min": 64, "max": 8192, "step": 32}),
                             "height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
                             "factor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 4.0, "step": 0.05})}}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "scale"
    CATEGORY = "H3ResCascade"

    def scale(self, width, height, factor):
        return (max(32, round(width * factor / 32) * 32), max(32, round(height * factor / 32) * 32))


# pooled_kv.py (H3PooledKVAttention) is a research artefact: pooled keys/values break H3 (see README); not registered

NODE_CLASS_MAPPINGS = {"H3ResSwitch": H3ResSwitch, "H3ResScale": H3ResScale, "H3LatentUpscaleLBH": H3LatentUpscaleLBH}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ResSwitch": "H3 Resolution Switch (mid-sampling)", "H3ResScale": "H3 Res Scale",
                              "H3LatentUpscaleLBH": "H3 Latent Upscale (LBH 3D net)"}
