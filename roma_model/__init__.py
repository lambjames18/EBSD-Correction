"""
roma_model - Standalone ROMA (Regression Matcher with Augmentation) implementation

This package contains a minimal, self-contained implementation of ROMA
for inference only. All training and evaluation code has been removed.

Essential components:
- models/: Core model architecture (matcher, encoders, transformer)
- utils/: Utility functions for inference
- model_factory.py: Factory functions for creating models
"""

import os

# Global variables required by ROMA internals
DEBUG_MODE = False
RANK = int(os.environ.get('RANK', default=0))
GLOBAL_STEP = 0
STEP_SIZE = 1
LOCAL_RANK = -1

# Version info
__version__ = "1.0.0"
__author__ = "Extracted from MatchAnything ROMA"

# Lazy imports to avoid requiring torch at package import time
# Users should import directly from submodules or use these functions
def __getattr__(name):
    """Lazy import for torch-dependent components."""
    if name == 'get_model':
        from roma_model.model_factory import get_model
        return get_model
    elif name == 'get_roma_model':
        from roma_model.model_factory import get_roma_model
        return get_roma_model
    elif name == 'RegressionMatcher':
        from roma_model.models.matcher import RegressionMatcher
        return RegressionMatcher
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'get_model',
    'get_roma_model',
    'RegressionMatcher',
    'DEBUG_MODE',
    'RANK',
    'GLOBAL_STEP',
    'STEP_SIZE',
    'LOCAL_RANK',
]