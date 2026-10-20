"""H3 Hires Fix: one node for "render at 1 MP, finish bigger".

The 1 MP render decides layout, motion, acting, detail level and audio. This node:
  1. takes its video latent (or encodes finished frames, e.g. after Fast De-rope / FaceDetail),
  2. upscales the latent with LBH-123-AI's learned 3D latent upscaler (no VAE round trip),
  3. pins the pass-1 audio outright (noise mask 0 on the audio stream),
  4. re-noises to sigma_start and runs the tail of the native schedule (beta57, 20 steps, the model's own shift)
     below sigma_start at the target size, so only lines and texture are redrawn.
Returns the AV latent at the target size (its audio stream is the pass-1 audio).
"""
import torch

import comfy.model_management as mm
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import folder_paths
import latent_preview
from comfy_extras.nodes_custom_sampler import Guider_Basic

from .nodes import LBH_FOLDER, VAE_DOWN, _lbh_upscale, _streams


def hires_sigmas(model, sigma_start, schedule_steps, steps_override):
    if steps_override > 0:  # e.g. a turbo LoRA: n even steps from sigma_start to 0
        return torch.linspace(sigma_start, 0.0, steps_override + 1)
    ms = model.get_model_object("model_sampling")
    full = comfy.samplers.beta_scheduler(ms, schedule_steps, alpha=0.5, beta=0.7)  # = RES4LYF beta57
    # skip schedule points just under sigma_start (e.g. 0.4465 below 0.45): a near-zero step costs a full step
    tail = [float(s) for s in full if float(s) < sigma_start - 0.02]
    return torch.tensor([sigma_start] + tail, dtype=torch.float32)


def frame_to_token(f):
    """H3 video latent: every 17 frames are tokens covering (1, 4, 4, 4, 4) frames"""
    g, r = divmod(f, 17)
    return g * 5 + (0 if r == 0 else 1 + (r - 1) // 4)


def face_keep_mask(atlas, T, h, w, margin):
    """(T, h, w) latent mask, 1 where a FaceDetail face sits (in any frame of that latent frame)"""
    keep = torch.zeros(T, h, w)
    if not atlas or atlas.get("nslots", 0) == 0:
        return keep
    sx, sy = w / atlas["W"], h / atlas["H"]
    for k in range(atlas["nslots"]):
        for i, e in enumerate(atlas["slots"][k]):
            if e is None:
                continue
            t, j = e
            half = 0.5 * margin * float(t["fs"][j])
            x0, x1 = (float(t["cx"][j]) - half) * sx, (float(t["cx"][j]) + half) * sx
            y0, y1 = (float(t["cy"][j]) - half) * sy, (float(t["cy"][j]) + half) * sy
            tt = min(frame_to_token(i), T - 1)
            keep[tt, max(0, int(y0)):min(h, int(y1) + 1), max(0, int(x0)):min(w, int(x1) + 1)] = 1.0
    # snap to H3's 2x2-latent DiT tokens: the model redraws a token if ANY of its cells is marked redraw, while the
    # inpaint step pastes the original back per cell, so a half-kept token turns to mush. Keep every touched token.
    hp, wp = h + h % 2, w + w % 2
    k = torch.nn.functional.pad(keep[None], (0, wp - w, 0, hp - h))
    k = torch.nn.functional.max_pool2d(k, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    return k[0, :, :h, :w]


class H3HiresFix:
    @classmethod
    def INPUT_TYPES(cls):
        lbh = folder_paths.get_filename_list(LBH_FOLDER) if LBH_FOLDER in folder_paths.folder_names_and_paths else []
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "the pass-1 model (same LoRAs + shift); a turbo LoRA works with steps_override"}),
                "positive": ("CONDITIONING", {"tooltip": "the pass-1 conditioning (text cond does not depend on size)"}),
                "latent": ("LATENT", {"tooltip": "pass-1 AV latent (sampler output). Its audio is pinned; its video is "
                                                 "upscaled unless 'images' is wired"}),
                "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1088, "min": 64, "max": 8192, "step": 32}),
                "sigma_start": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 0.95, "step": 0.01,
                                          "tooltip": "how much is redrawn at the target size. 0.45 = crisp lines, same "
                                                     "take; 0.31 softer/cheaper; >0.56 starts redrawing small features"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "lbh_model": (lbh or ["(none)"],),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "finished 1 MP frames (e.g. after Fast De-rope / FaceDetail / Anime "
                                                "Cadence); encoded with 'vae' and used instead of the latent's video"}),
                "vae": ("VAE", {"tooltip": "video VAE for 'images' (full H3 VAE recommended)"}),
                "schedule_steps": ("INT", {"default": 20, "min": 4, "max": 100,
                                           "tooltip": "length of the native beta57 schedule whose tail is used"}),
                "steps_override": ("INT", {"default": 0, "min": 0, "max": 20,
                                           "tooltip": "0 = native tail (4 steps from 0.45). >0 = that many even steps "
                                                      "(e.g. 2 with a DMAD turbo model)"}),
                "face_atlas": ("H3_FACE_ATLAS", {"tooltip": "from H3 Face Atlas (FaceDetail): keep those faces as they are "
                                                            "(small faces would be redrawn badly at sigma 0.45)"}),
                "face_margin": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                                          "tooltip": "protected box side = face size x this"}),
            },
        }

    RETURN_TYPES = ("LATENT", "SIGMAS")
    RETURN_NAMES = ("latent", "sigmas")
    FUNCTION = "run"
    CATEGORY = "H3ResCascade"
    DESCRIPTION = ("Render at 1 MP, finish bigger: learned latent upscale + short low-sigma refine at the target size "
                   "with the pass-1 audio pinned. Keeps the 1 MP take.")

    def run(self, model, positive, latent, width, height, sigma_start, seed, lbh_model, images=None, vae=None,
            schedule_steps=20, steps_override=0, face_atlas=None, face_margin=2.0):
        xs = _streams(latent)
        if images is not None:
            if vae is None:
                raise ValueError("H3HiresFix: 'images' needs 'vae'")
            video = vae.encode(images[:, :, :, :3])
        else:
            video = xs[0]
        size = (height // VAE_DOWN, width // VAE_DOWN)
        video = _lbh_upscale(video.float(), size, lbh_model).to(torch.float32)
        keep = face_keep_mask(face_atlas, video.shape[2], size[0], size[1], face_margin)
        streams, masks = [video], [(1.0 - keep)[None, None]]
        if keep.any():
            print(f"[H3HiresFix] faces kept: {keep.mean().item():.1%} of the video latent")
        if len(xs) > 1:
            audio = xs[1].float()
            streams.append(audio)
            masks.append(torch.zeros_like(audio[:1]))
        samples = comfy.nested_tensor.NestedTensor(streams) if len(streams) > 1 else streams[0]
        noise_mask = comfy.nested_tensor.NestedTensor(masks) if len(masks) > 1 else None

        sigmas = hires_sigmas(model, sigma_start, schedule_steps, steps_override)
        print(f"[H3HiresFix] {tuple(xs[0].shape[-2:])} -> {size}, sigmas {[round(float(s), 4) for s in sigmas]}")
        guider = Guider_Basic(model)
        guider.set_conds(positive)
        noise = comfy.sample.prepare_noise(samples, seed)
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, {})
        out = guider.sample(noise, samples, comfy.samplers.sampler_object("euler"), sigmas, denoise_mask=noise_mask,
                            callback=callback, disable_pbar=False, seed=seed)
        return ({"samples": out.to(mm.intermediate_device())}, sigmas)


class H3FrontLoadedSigmas:
    """Pass-1 speed-up without a LoRA: H3 decides layout, acting and detail above sigma ~0.79, and its output heads are
    step-size aware, so keep every native step down to `keep_above` and take the rest of the schedule in fewer, bigger
    base-model steps (picked evenly from the native schedule). 20 steps -> 12 + 4: -23% sampling, same take."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "steps": ("INT", {"default": 20, "min": 4, "max": 100, "tooltip": "native beta57 schedule length"}),
                             "keep_above": ("FLOAT", {"default": 0.79, "min": 0.0, "max": 1.0, "step": 0.01,
                                                      "tooltip": "every native step with sigma >= this is kept"}),
                             "tail_steps": ("INT", {"default": 4, "min": 1, "max": 50,
                                                    "tooltip": "steps for the rest (4 = closest to native; 3/2 drift)"})}}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "run"
    CATEGORY = "H3ResCascade"

    def run(self, model, steps, keep_above, tail_steps):
        full = comfy.samplers.beta_scheduler(model.get_model_object("model_sampling"), steps, alpha=0.5, beta=0.7)
        full = [float(s) for s in full]
        # switch point: the last native sigma >= keep_above (0.7903 for 0.79); every step above it is native
        k = max(i for i, s in enumerate(full) if s >= keep_above - 1e-4)
        rest = len(full) - 1 - k
        n = min(tail_steps, rest)
        idx = sorted({k + round(j * rest / n) for j in range(n + 1)})
        sig = full[:k] + [full[i] for i in idx]
        print(f"[H3FrontLoadedSigmas] {len(full) - 1} -> {len(sig) - 1} steps: {[round(s, 4) for s in sig]}")
        return (torch.tensor(sig, dtype=torch.float32),)
