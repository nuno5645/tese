from .base_model import SegmentationModel
from .deeplabv3 import DeepLabV3
from .segformer import SegFormer
from .enhanced_unet import EnhancedUNet

MODELS = {
    'deeplabv3': DeepLabV3,
    'segformer': SegFormer,
    'enhanced_unet': EnhancedUNet
}

def get_model(name, **kwargs):
    """Factory function to get a model by name"""
    if name not in MODELS:
        raise ValueError(f"Model {name} not found. Available models: {list(MODELS.keys())}")
    return MODELS[name](**kwargs)
