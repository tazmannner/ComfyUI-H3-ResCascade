from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .hires import H3FrontLoadedSigmas, H3HiresFix
from .hold_transfer import H3HoldTransfer
from .face_rescale import H3FaceAtlasRescale, H3FaceAtlasStitchChunked

NODE_CLASS_MAPPINGS["H3HiresFix"] = H3HiresFix
NODE_DISPLAY_NAME_MAPPINGS["H3HiresFix"] = "H3 Hires Fix (1 MP -> bigger, audio pinned)"
NODE_CLASS_MAPPINGS["H3FrontLoadedSigmas"] = H3FrontLoadedSigmas
NODE_DISPLAY_NAME_MAPPINGS["H3FrontLoadedSigmas"] = "H3 Front-Loaded Sigmas (pass-1 speed, same take)"
NODE_CLASS_MAPPINGS["H3HoldTransfer"] = H3HoldTransfer
NODE_DISPLAY_NAME_MAPPINGS["H3HoldTransfer"] = "H3 Hold Transfer (keep the source's 2s/3s holds)"
NODE_CLASS_MAPPINGS["H3FaceAtlasRescale"] = H3FaceAtlasRescale
NODE_DISPLAY_NAME_MAPPINGS["H3FaceAtlasRescale"] = "H3 Face Atlas Rescale (FaceDetail faces onto hires)"
NODE_CLASS_MAPPINGS["H3FaceAtlasStitchChunked"] = H3FaceAtlasStitchChunked
NODE_DISPLAY_NAME_MAPPINGS["H3FaceAtlasStitchChunked"] = "H3 Face Atlas Stitch (chunked, long hires clips)"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
