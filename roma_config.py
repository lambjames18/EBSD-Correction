"""
roma_config.py - Configuration for ROMA Matcher

This module provides configuration settings for the ROMA model,
following the MatchAnything configuration structure.
"""

from typing import Tuple


def get_roma_config(
    confidence_threshold: float = 0.1,
    num_samples: int = 5000,
    resize_by_stretch: bool = True,
    normalize_images: bool = False,
    coarse_resolution: Tuple[int, int] = (560, 560),
    upsample_resolution: Tuple[int, int] = (864, 864),
    use_symmetric_matching: bool = True,
    use_certainty_attenuation: bool = True,
) -> dict:
    """
    Generate ROMA model configuration.

    Args:
        confidence_threshold: Minimum confidence for accepting matches (0.0-1.0)
        num_samples: Maximum number of matches to sample
        resize_by_stretch: If True, resize by stretching; if False, pad to square
        normalize_images: Whether to normalize images to [0, 1] range
        coarse_resolution: Resolution for coarse matching (must be divisible by 14)
        upsample_resolution: Resolution for upsampling (must be divisible by 8)
        use_symmetric_matching: Whether to use symmetric matching (both directions)
        use_certainty_attenuation: Whether to attenuate certainty scores

    Returns:
        Configuration dictionary for ROMA model
    """
    # Validate resolutions
    if coarse_resolution[0] % 14 != 0 or coarse_resolution[1] % 14 != 0:
        raise ValueError(f"Coarse resolution must be divisible by 14, got {coarse_resolution}")

    if upsample_resolution[0] % 8 != 0 or upsample_resolution[1] % 8 != 0:
        raise ValueError(f"Upsample resolution must be divisible by 8, got {upsample_resolution}")

    config = {
        # Match confidence threshold
        'match_thresh': confidence_threshold,

        # Image preprocessing
        'resize_by_stretch': resize_by_stretch,
        'normalize_img': normalize_images,

        # Model configuration
        'model': {
            'coarse_backbone': 'DINOv2_large',  # Backbone architecture
            'coarse_feat_dim': 1024,             # Feature dimension from DINOv2
            'medium_feat_dim': 512,              # Medium-level feature dimension
            'coarse_patch_size': 14,             # Patch size for DINOv2
            'amp': True,                         # Use automatic mixed precision (FP16)
        },

        # Test-time configuration
        'test_time': {
            'coarse_res': coarse_resolution,           # Resolution for coarse matching
            'upsample': True,                          # Enable upsampling
            'upsample_res': upsample_resolution,       # Resolution for upsampling
            'symmetric': use_symmetric_matching,       # Use symmetric matching
            'attenutate_cert': use_certainty_attenuation,  # Attenuate certainty scores
        },

        # Sampling configuration
        'sample': {
            'method': 'threshold_balanced',  # Sampling method
            'n_sample': num_samples,         # Number of matches to sample
            'thresh': 0.05,                  # Threshold for balanced sampling
        },
    }

    return config


# Preset configurations for common use cases

def get_fast_config() -> dict:
    """
    Get configuration optimized for speed (lower resolution, fewer samples).

    Returns:
        Fast configuration dictionary
    """
    return get_roma_config(
        confidence_threshold=0.15,
        num_samples=2000,
        coarse_resolution=(448, 448),
        upsample_resolution=(672, 672),
    )


def get_accurate_config() -> dict:
    """
    Get configuration optimized for accuracy (higher resolution, more samples).

    Returns:
        Accurate configuration dictionary
    """
    return get_roma_config(
        confidence_threshold=0.05,
        num_samples=10000,
        coarse_resolution=(672, 672),
        upsample_resolution=(1120, 1120),
    )


def get_balanced_config() -> dict:
    """
    Get balanced configuration (default settings).

    Returns:
        Balanced configuration dictionary
    """
    return get_roma_config()


# Configuration presets for different data types

def get_ebsd_config() -> dict:
    """
    Get configuration optimized for EBSD (Electron Backscatter Diffraction) data.

    EBSD images typically have:
    - High spatial resolution
    - Similar modalities (e.g., IQ, CI, phase maps)
    - Need for accurate registration

    Returns:
        EBSD-optimized configuration dictionary
    """
    return get_roma_config(
        confidence_threshold=0.08,
        num_samples=5000,
        resize_by_stretch=True,
        normalize_images=False,  # EBSD images often have specific intensity ranges
        coarse_resolution=(560, 560),
        upsample_resolution=(896, 896),
    )


def get_multimodal_config() -> dict:
    """
    Get configuration optimized for cross-modality registration.

    Cross-modality images (e.g., SEM vs. EBSD) have:
    - Different appearance characteristics
    - Need for robust feature matching
    - Benefit from symmetric matching

    Returns:
        Multi-modal configuration dictionary
    """
    return get_roma_config(
        confidence_threshold=0.12,
        num_samples=5000,
        resize_by_stretch=True,
        normalize_images=True,  # Normalize to reduce appearance differences
        use_symmetric_matching=True,
        use_certainty_attenuation=True,
    )


def get_microscopy_config() -> dict:
    """
    Get configuration optimized for general microscopy images.

    Returns:
        Microscopy-optimized configuration dictionary
    """
    return get_roma_config(
        confidence_threshold=0.1,
        num_samples=5000,
        resize_by_stretch=True,
        coarse_resolution=(560, 560),
        upsample_resolution=(864, 864),
    )


# Configuration summary
AVAILABLE_PRESETS = {
    'fast': get_fast_config,
    'accurate': get_accurate_config,
    'balanced': get_balanced_config,
    'ebsd': get_ebsd_config,
    'multimodal': get_multimodal_config,
    'microscopy': get_microscopy_config,
}


def get_preset_config(preset_name: str) -> dict:
    """
    Get a preset configuration by name.

    Args:
        preset_name: Name of the preset configuration
                    Options: 'fast', 'accurate', 'balanced', 'ebsd', 'multimodal', 'microscopy'

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If preset_name is not recognized
    """
    if preset_name not in AVAILABLE_PRESETS:
        available = ', '.join(AVAILABLE_PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )

    return AVAILABLE_PRESETS[preset_name]()


def print_config(config: dict) -> None:
    """
    Pretty-print a configuration dictionary.

    Args:
        config: Configuration dictionary to print
    """
    import json

    print("ROMA Configuration:")
    print("=" * 60)
    print(json.dumps(config, indent=2))
    print("=" * 60)
