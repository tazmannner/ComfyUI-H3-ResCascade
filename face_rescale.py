"""H3 Face Atlas Rescale: reuse FaceDetail's redrawn faces on a hires version of the clip.

FaceDetail redraws every small face in a 512 px tile (a ~35 px face at 1 MP is magnified ~10x), far more resolution
than 1920x1088 needs. A hires refine would redraw those faces from ~2 latent tokens again (badly), and the LBH latent
upscale turns a small face into a smudge. Instead: rescale the atlas's frame-space coordinates to the hires size and
run FaceDetail's own stitch (colour match, geometry lock, fidelity gate) onto the hires frames with the same tiles.
"""
import numpy as np

FRAME_KEYS = {"cx": "x", "cy": "y", "fs": "s", "side": "s"}  # "speed" is relative to face size: unitless


class H3FaceAtlasRescale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"atlas": ("H3_FACE_ATLAS",),
                             "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 8}),
                             "height": ("INT", {"default": 1088, "min": 64, "max": 8192, "step": 8})}}

    RETURN_TYPES = ("H3_FACE_ATLAS",)
    FUNCTION = "run"
    CATEGORY = "H3ResCascade"
    DESCRIPTION = ("Scale an H3 Face Atlas (FaceDetail) to another frame size, so H3 Face Atlas Stitch can put the same "
                   "redrawn face tiles onto the hires frames.")

    def run(self, atlas, width, height):
        if atlas.get("nslots", 0) == 0:
            return (dict(atlas, W=width, H=height),)
        sx, sy = width / atlas["W"], height / atlas["H"]
        sc = {"x": sx, "y": sy, "s": (sx * sy) ** 0.5}
        copies = {}

        def scaled(t):
            if id(t) not in copies:
                u = dict(t)
                for k, axis in FRAME_KEYS.items():
                    if k in u and u[k] is not None:
                        u[k] = np.asarray(u[k], dtype=np.float64) * sc[axis]
                copies[id(t)] = u
            return copies[id(t)]

        out = dict(atlas, W=width, H=height)
        out["slots"] = [[None if e is None else (scaled(e[0]), e[1]) for e in slot] for slot in atlas["slots"]]
        return (out,)


class H3FaceAtlasStitchChunked:
    """FaceDetail's H3 Face Atlas Stitch, run on chunks of frames (long hires clips: 15 s at 1920x1088 is ~9 GB of
    float32 frames, and the stitch makes several full copies). Calls FaceDetail's own node per chunk with the atlas
    sliced to that chunk, so the stitch logic itself is unchanged; colour/fidelity stats become per chunk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "refined": ("IMAGE",), "atlas": ("H3_FACE_ATLAS",),
                             "chunk": ("INT", {"default": 96, "min": 8, "max": 4096}),
                             "mask_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                             "feather": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 32.0, "step": 0.5}),
                             "color_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "geometry_lock": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "H3ResCascade"

    def run(self, images, refined, atlas, chunk, mask_scale, feather, color_match, blend, geometry_lock):
        import nodes as comfy_nodes
        import torch
        stitch = comfy_nodes.NODE_CLASS_MAPPINGS["H3FaceAtlasStitch"]()
        if atlas.get("nslots", 0) == 0:
            return (images,)
        n = images.shape[0]
        outs = []
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            sub = dict(atlas, n=b - a, slots=[slot[a:b] for slot in atlas["slots"]])
            outs.append(stitch.run(images[a:b], refined[a:b], sub, mask_scale, feather, color_match, blend,
                                   geometry_lock=geometry_lock)[0].cpu())
        return (torch.cat(outs, dim=0),)
