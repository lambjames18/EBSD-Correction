"""
models - ROMA model components

Contains the core model architecture for ROMA inference:
- matcher.py: RegressionMatcher, Decoder, ConvRefiner, GP
- encoders.py: CNNandDinov2, VGG19, ResNet50
- transformer/: Transformer components (DINOv2, attention layers)
"""

# Import key components for convenience
from .matcher import RegressionMatcher, Decoder, ConvRefiner, GP, CosKernel
from .encoders import CNNandDinov2

__all__ = [
    'RegressionMatcher',
    'Decoder',
    'ConvRefiner',
    'GP',
    'CosKernel',
    'CNNandDinov2',
]