"""H3 Hold Transfer: give a re-rendered (e.g. hires) clip exactly the holds of its source clip.

After Anime Cadence / Fast De-rope the 1 MP frames hold drawings (pixel i == pixel i-1, per region: a character can be
on 2s while the camera pans on 1s). A hires refine redraws every frame slightly differently, so the holds turn into
boil, and re-running Anime Cadence on the hires frames re-decides the cadence (it can hold the wrong drawing over an
impact frame). This copies the source's hold map instead: wherever the source repeats a pixel, the output repeats
its own previous pixel. Nothing is re-decided; impacts stay where the source put them.
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
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "H3ResCascade"

    def run(self, reference, images, threshold, erode, feather):
        n = min(reference.shape[0], images.shape[0])
        if reference.shape[0] != images.shape[0]:
            print(f"[H3HoldTransfer] frame counts differ ({reference.shape[0]} vs {images.shape[0]}), using {n}")
        m = hold_masks(reference[:n].float(), threshold, erode)
        H, W = images.shape[1], images.shape[2]
        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        if feather > 0:
            m = F.avg_pool2d(m, 2 * feather + 1, stride=1, padding=feather, count_include_pad=False)
        out = images[:n].clone()
        for i in range(1, n):  # sequential: a run of holds propagates the first drawing of the run
            mi = m[i, 0, :, :, None].to(out.dtype)
            if mi.max() > 0:
                out[i] = mi * out[i - 1] + (1 - mi) * out[i]
        full = (m[:, 0].mean(dim=(1, 2)) > 0.98).sum().item()
        report = f"held area {m.mean().item():.1%} of all pixels; {full} of {n} frames fully held"
        print(f"[H3HoldTransfer] {report}")
        return (out, report)
