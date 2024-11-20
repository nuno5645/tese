import torch.nn as nn

class SegmentationModel(nn.Module):
    """Base class for all segmentation models"""
    def __init__(self):
        super().__init__()
        
    def get_name(self):
        return self.__class__.__name__
        
    def get_parameters(self):
        """Return model parameters for logging"""
        return {
            "name": self.get_name(),
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
