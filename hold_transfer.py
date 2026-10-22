"""H3 Hold Transfer: give a re-rendered (e.g. hires) clip exactly the holds of its source clip.

After Anime Cadence / Fast De-rope the 1 MP frames hold drawings (pixel i == pixel i-1, per region: a character can be
on 2s while the camera pans on 1s). A hires refine redraws every frame slightly differently, so the holds turn into
boil, and re-running Anime Cadence on the hires frames re-decides the cadence (it can hold the wrong drawing over an
impact frame). This copies the source's hold map instead: wherever the source repeats a pixel, the output repeats
its own previous pixel. Nothing is re-decided; impacts stay where the source put them.

Anime Cadence (pick sharpest) fills a hold with the sharpest frame of its run, which is often not the run's first
frame. When `images` is a re-render of Anime Cadence's INPUT (the hires frames are rendered before cadence, so
FaceDetail's tiles stay on their frames), connect that input as `source`: every reference frame that is an exact copy
of another source frame then takes the re-rendered version of THAT frame, so the hold shows the drawing Anime Cadence
chose (without it, the run's first frame is held: e.g. a transition frame instead of the impact flash after it).
"""
import torch
import torch.nn.functional as F


def hold_masks(ref, thr, erode):
    """ref (N,H,W,C) in 0..1 -> (N,1,H,W) float, 1 where frame i repeats frame i-1 (frame 0 never held)"""
    d = (ref[1:] - ref[:-1]).abs().amax(dim=-1)  # (N-1,H,W)
    m = (d <= thr).float()[:, None]
    if erode > 0:  # shrink held regions so their edges never sit on a changing pixel
        m = -F.max_pool2d(-m, 2 * erode + 1, stride=1, padding=erode)
    return torch.cat([torch.zeros_like(m[:1]), m], dim=0)


def source_frames(reference, source, window=8, exact=0.002, tie=0.002):
    """index of the source frame each reference frame copies (identity where none): for every reference frame, the
    best match among the source frames within +-window, used only when it is a copy (mean abs difference <= exact at
    1/8 size) and clearly better than the frame itself (ties keep the identity, e.g. 2s pairs Fast De-rope already
    held). Frames that copy nothing (e.g. Anime Cadence camera=follow warps) keep their own index."""
    n = min(reference.shape[0], source.shape[0])

    def small(x, size=None):
        x = x[:n, :, :, :3].float().movedim(-1, 1)
        size = size or (max(8, x.shape[2] // 8), max(8, x.shape[3] // 8))
        return F.interpolate(x, size=size, mode="area")

    r = small(reference)
    s = small(source, r.shape[-2:])
    idx = list(range(n))
    for t in range(n):
        lo, hi = max(0, t - window), min(n, t + window + 1)
        d = (r[t:t + 1] - s[lo:hi]).abs().mean(dim=(1, 2, 3))
        k = lo + int(d.argmin())
        if k != t and d[k - lo] <= exact and d[t - lo] > d[k - lo] + tie:
            idx[t] = k
    return idx


class H3HoldTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference": ("IMAGE", {"tooltip": "the clip whose holds are the truth (e.g. 1 MP output of Anime Cadence)"}),
            "images": ("IMAGE", {"tooltip": "the re-rendered clip (e.g. hires decode), same frame count, any size"}),
            "threshold": ("FLOAT", {"default": 0.004, "min": 0.0, "max": 0.1, "step": 0.001,
                                    "tooltip": "max per-pixel change (0..1) that still counts as a hold in the reference"}),
            "erode": ("INT", {"default": 2, "min": 0, "max": 16, "tooltip": "reference px to shrink held regions by"}),
            "feather": ("INT", {"default": 4, "min": 0, "max": 64, "tooltip": "output px of soft edge on held regions"}),
        }, "optional": {
            "source": ("IMAGE", {"tooltip": "the frames `reference` was made from (Anime Cadence's input) when `images` "
                                            "re-renders THOSE frames: each hold then shows the re-render of the frame "
                                            "Anime Cadence actually held (pick sharpest), not the run's first frame"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "H3ResCascade"

    def run(self, reference, images, threshold, erode, feather, source=None):
        n = min(reference.shape[0], images.shape[0])
        if reference.shape[0] != images.shape[0]:
            print(f"[H3HoldTransfer] frame counts differ ({reference.shape[0]} vs {images.shape[0]}), using {n}")
        m = hold_masks(reference[:n].float(), threshold, erode)
        H, W = images.shape[1], images.shape[2]
        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        if feather > 0:
            m = F.avg_pool2d(m, 2 * feather + 1, stride=1, padding=feather, count_include_pad=False)
        moved = 0
        if source is not None:
            idx = source_frames(reference[:n], source[:n])
            idx += list(range(len(idx), n))
            moved = sum(1 for t, k in enumerate(idx) if k != t)
            out = images[torch.tensor(idx)] if moved else images[:n].clone()
        else:
            out = images[:n].clone()
        for i in range(1, n):  # sequential: a run of holds propagates the first drawing of the run
            mi = m[i, 0, :, :, None].to(out.dtype)
            if mi.max() > 0:
                out[i] = mi * out[i - 1] + (1 - mi) * out[i]
        full = (m[:, 0].mean(dim=(1, 2)) > 0.98).sum().item()
        report = f"held area {m.mean().item():.1%} of all pixels; {full} of {n} frames fully held"
        if source is not None:
            report += f"; {moved} frames show another source frame (the drawing Anime Cadence held)"
        print(f"[H3HoldTransfer] {report}")
        return (out, report)
