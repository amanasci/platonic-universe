import torch
from typing import Any, Dict, Iterable
from pu.models.base import ModelAdapter
from pu.preprocess import PreprocessSAM2
from pu.models.registry import register_adapter

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


class SAM2Adapter(ModelAdapter):
    """
    Adapter for SAM2 (Segment Anything Model 2) models.
    Uses SAM2's image encoder to extract image embeddings.
    
    Note: SAM2 must be installed separately. Install with:
        pip install git+https://github.com/facebookresearch/sam2.git
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not installed. Please install it with:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                "or:\n"
                "  cd /path/to/sam2 && SAM2_BUILD_CUDA=0 pip install -e ."
            )
        self.model = None
        self.predictor = None

    def load(self) -> None:
        # Build SAM2 model using the config and checkpoint
        # model_name is expected to be a Hugging Face model ID like "facebook/sam2.1-hiera-large"
        # We'll use the from_pretrained method to load from HuggingFace
        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
        self.model = self.predictor.model
        self.model.to("cuda").eval()

    def get_preprocessor(self, modes: Iterable[str]):
        # Return a callable compatible with datasets.Dataset.map
        return PreprocessSAM2(modes, self.predictor._transforms, resize=False)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        """
        Given a batch from the DataLoader and the mode name,
        return embeddings for that batch using SAM2's image encoder.
        """
        # batch contains preprocessed images under f"{mode}" key
        inputs = batch[f"{mode}"].to("cuda")

        with torch.no_grad():
            # Forward through the image encoder
            backbone_out = self.model.forward_image(inputs)
            _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

            # Add no_mem_embed if needed (following SAM2ImagePredictor logic)
            if self.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

            # Use the highest resolution features (last in the list)
            # These are the image embeddings from the encoder
            emb = vision_feats[-1]

            # Pool spatially to get a fixed-size embedding per image
            # Average pool over spatial dimensions (H, W)
            emb = emb.mean(dim=0).detach()

        return emb


# Register the adapter only if SAM2 is available
if SAM2_AVAILABLE:
    register_adapter("sam2", SAM2Adapter)