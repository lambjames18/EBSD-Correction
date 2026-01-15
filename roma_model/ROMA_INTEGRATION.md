# ROMA Integration for EBSD-Correction GUI

This document describes the integration of the MatchAnything ROMA (Regression Matcher with Augmentation) model into the EBSD-Correction GUI for automatic control point detection.

## Overview

ROMA is a state-of-the-art deep learning model for detecting corresponding points between images. It uses:
- **DINOv2** transformer backbone for robust feature extraction
- **Multi-scale refinement** for accurate point localization
- **Symmetric matching** for bidirectional correspondence
- **Certainty estimation** for confidence-based filtering

This makes it significantly more robust than traditional methods like SIFT, especially for:
- Cross-modality registration (e.g., SEM ↔ EBSD)
- Images with repetitive patterns
- Low-contrast or noisy images
- Large deformations

## Files Created

### 1. `roma_matcher.py`
Main wrapper class providing a clean interface to ROMA:
- **`RomaMatcher`**: Main class for point detection
- **`detect_points_roma()`**: Convenience function for one-time use

### 2. `roma_config.py`
Configuration management:
- **`get_roma_config()`**: Generate custom configurations
- **Preset configurations**: Fast, accurate, balanced, EBSD-specific, etc.
- **`get_preset_config()`**: Load predefined presets by name

### 3. `roma_example.py`
Comprehensive examples demonstrating:
- Basic usage
- Preset configurations
- Metadata extraction
- Visualization
- Integration with PointAutoIdentifier

### 4. `models.py` (Updated)
Integrated ROMA into existing `PointAutoIdentifier` class:
- Added "matchanything" and "roma" engines
- Comprehensive error handling
- Automatic checkpoint downloading instructions

## Installation

### Prerequisites

1. **PyTorch with CUDA** (if using GPU):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Required dependencies**:
   ```bash
   pip install kornia einops loguru timm transformers pillow scikit-image scipy
   ```

3. **Download ROMA checkpoint**:
   - Download from: [Google Drive Link](https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view)
   - Place in: `MatchAnything/weights/matchanything_roma.ckpt`

### Directory Structure

```
EBSD-Correction/
├── models.py                    # Updated with ROMA integration
├── roma_matcher.py              # ROMA wrapper class (NEW)
├── roma_config.py               # Configuration presets (NEW)
├── roma_example.py              # Usage examples (NEW)
├── ROMA_INTEGRATION.md          # This file (NEW)
└── MatchAnything/
    ├── weights/
    │   └── matchanything_roma.ckpt  # Download and place here
    ├── third_party/ROMA/
    │   └── ...                  # ROMA source code
    └── src/
        └── ...                  # Supporting code
```

## Usage

### Basic Usage

```python
from roma_matcher import RomaMatcher

# Create matcher
matcher = RomaMatcher(
    checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt",
    confidence_threshold=0.1,
    num_samples=5000,
)

# Detect points
src_points, dst_points, confidences = matcher.detect_points(
    source_image,      # numpy array (H, W) or (H, W, C)
    destination_image, # numpy array (H, W) or (H, W, C)
    ransac_filter=True,
    ransac_threshold=5.5,
)

print(f"Detected {len(src_points)} matches")
```

### Integration with PointAutoIdentifier

```python
from models import PointAutoIdentifier

# Use ROMA through the auto-identifier
src_points, dst_points = PointAutoIdentifier.detect_points(
    source_image,
    destination_image,
    method='matchanything',  # or 'roma'
    confidence_threshold=0.1,
    ransac_filter=True,
)
```

### Using Preset Configurations

```python
from roma_matcher import RomaMatcher
from roma_config import get_preset_config

# Option 1: Load preset as dict and manually set parameters
ebsd_config = get_preset_config('ebsd')
matcher = RomaMatcher(
    checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt",
    confidence_threshold=ebsd_config['match_thresh'],
    num_samples=ebsd_config['sample']['n_sample'],
    coarse_resolution=ebsd_config['test_time']['coarse_res'],
    upsample_resolution=ebsd_config['test_time']['upsample_res'],
)

# Option 2: Use in PointAutoIdentifier
src_points, dst_points = PointAutoIdentifier.detect_points(
    source_image,
    destination_image,
    method='matchanything',
    confidence_threshold=0.08,  # EBSD-optimized
    coarse_resolution=(560, 560),
    upsample_resolution=(896, 896),
)
```

### Available Presets

| Preset       | Use Case                          | Confidence | Samples | Resolution      |
|--------------|-----------------------------------|------------|---------|-----------------|
| `fast`       | Quick processing, lower accuracy  | 0.15       | 2000    | 448×448 → 672   |
| `balanced`   | Default, good balance             | 0.10       | 5000    | 560×560 → 864   |
| `accurate`   | High accuracy, slower             | 0.05       | 10000   | 672×672 → 1120  |
| `ebsd`       | EBSD data (high resolution)       | 0.08       | 5000    | 560×560 → 896   |
| `multimodal` | Cross-modality (SEM ↔ EBSD)       | 0.12       | 5000    | 560×560 → 864   |
| `microscopy` | General microscopy                | 0.10       | 5000    | 560×560 → 864   |

## Configuration Parameters

### RomaMatcher Parameters

| Parameter                  | Type           | Default         | Description                                      |
|----------------------------|----------------|-----------------|--------------------------------------------------|
| `checkpoint_path`          | str            | Required        | Path to .ckpt file                               |
| `confidence_threshold`     | float          | 0.1             | Minimum confidence (0.0-1.0)                     |
| `num_samples`              | int            | 5000            | Maximum matches to sample                        |
| `device`                   | str            | "cuda"          | Device: "cuda" or "cpu"                          |
| `resize_by_stretch`        | bool           | True            | Stretch vs. pad to square                        |
| `normalize_images`         | bool           | False           | Normalize to [0, 1]                              |
| `coarse_resolution`        | Tuple[int,int] | (560, 560)      | Coarse matching resolution (÷14)                 |
| `upsample_resolution`      | Tuple[int,int] | (864, 864)      | Upsampling resolution (÷8)                       |
| `use_symmetric_matching`   | bool           | True            | Bidirectional matching                           |
| `use_certainty_attenuation`| bool           | True            | Attenuate confidence scores                      |

### detect_points() Parameters

| Parameter          | Type  | Default | Description                           |
|--------------------|-------|---------|---------------------------------------|
| `ransac_filter`    | bool  | False   | Apply RANSAC outlier filtering        |
| `ransac_threshold` | float | 5.5     | RANSAC inlier threshold (pixels)      |

## Performance Considerations

### GPU Memory
- **Minimum**: 4GB VRAM for (560×560) resolution
- **Recommended**: 8GB+ VRAM for (864×864) upsampling
- **High accuracy**: 12GB+ VRAM for (1120×1120)

### Speed
- **Fast preset**: ~0.5-1 second per pair (GPU)
- **Balanced preset**: ~1-2 seconds per pair (GPU)
- **Accurate preset**: ~2-4 seconds per pair (GPU)
- **CPU mode**: 10-20× slower than GPU

### Optimization Tips

1. **Reuse matcher instance**: Don't create new matcher for each pair
   ```python
   # Good: Create once, reuse
   matcher = RomaMatcher(...)
   for img_pair in pairs:
       src_pts, dst_pts, _ = matcher.detect_points(img_pair[0], img_pair[1])

   # Bad: Creates new matcher each time
   for img_pair in pairs:
       matcher = RomaMatcher(...)  # Slow!
       src_pts, dst_pts, _ = matcher.detect_points(img_pair[0], img_pair[1])
   ```

2. **Adjust resolution for speed**: Lower resolution = faster inference
   ```python
   # Fast but less accurate
   matcher = RomaMatcher(coarse_resolution=(448, 448))
   ```

3. **Reduce num_samples**: Fewer samples = faster post-processing
   ```python
   matcher = RomaMatcher(num_samples=2000)  # vs. default 5000
   ```

4. **Use CPU for small batches**: GPU has initialization overhead
   ```python
   matcher = RomaMatcher(device='cpu')  # For 1-2 image pairs
   ```

## Troubleshooting

### Common Issues

#### 1. FileNotFoundError: Checkpoint not found
```
Solution: Download weights from Google Drive and place in:
  MatchAnything/weights/matchanything_roma.ckpt
```

#### 2. CUDA out of memory
```
Solutions:
- Reduce coarse_resolution: (560, 560) → (448, 448)
- Disable upsampling: Set upsample_resolution to same as coarse_resolution
- Reduce num_samples: 5000 → 2000
- Use CPU: device='cpu'
```

#### 3. ImportError: No module named 'roma_matcher'
```
Solution: Ensure roma_matcher.py is in the same directory as models.py
```

#### 4. Few or no matches detected
```
Solutions:
- Lower confidence_threshold: 0.1 → 0.05
- Increase num_samples: 5000 → 10000
- Disable RANSAC: ransac_filter=False
- Try different preset: 'multimodal' for cross-modality
```

#### 5. Matches are inaccurate
```
Solutions:
- Enable RANSAC: ransac_filter=True
- Increase confidence_threshold: 0.1 → 0.15
- Use accurate preset: get_preset_config('accurate')
- Adjust ransac_threshold: Default 5.5 pixels
```

## Comparison with SIFT

| Feature                  | SIFT                    | ROMA                           |
|--------------------------|-------------------------|--------------------------------|
| **Algorithm**            | Hand-crafted features   | Deep learning (DINOv2)         |
| **Cross-modality**       | Poor                    | Excellent                      |
| **Low contrast**         | Struggles               | Robust                         |
| **Speed (GPU)**          | Fast (CPU)              | Fast (GPU), slow (CPU)         |
| **Memory**               | Low                     | High (4-12GB VRAM)             |
| **Accuracy**             | Moderate                | High                           |
| **Setup**                | Built-in (scikit-image) | Requires checkpoint download   |

## Advanced Usage

### Custom Configuration

```python
from roma_matcher import RomaMatcher

# Custom configuration for specific use case
matcher = RomaMatcher(
    checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt",
    confidence_threshold=0.08,
    num_samples=8000,
    coarse_resolution=(672, 672),
    upsample_resolution=(1008, 1008),
    use_symmetric_matching=True,
    normalize_images=True,  # Helpful for cross-modality
)
```

### Batch Processing

```python
from roma_matcher import RomaMatcher
from pathlib import Path

matcher = RomaMatcher(checkpoint_path="...")

image_pairs = [
    ("image1_src.tif", "image1_dst.tif"),
    ("image2_src.tif", "image2_dst.tif"),
    # ...
]

results = []
for src_path, dst_path in image_pairs:
    src_img = load_image(src_path)
    dst_img = load_image(dst_path)

    result = matcher.detect_points_with_metadata(src_img, dst_img)
    results.append(result)

    print(f"{src_path}: {result['num_matches']} matches, "
          f"confidence: {result['mean_confidence']:.3f}")
```

### Caching in PointAutoIdentifier

For GUI applications, consider caching the matcher instance:

```python
class PointAutoIdentifier:
    def __init__(self):
        self._roma_matcher = None

    def _get_roma_matcher(self, **kwargs):
        """Get or create cached ROMA matcher."""
        if self._roma_matcher is None:
            from roma_matcher import RomaMatcher
            self._roma_matcher = RomaMatcher(**kwargs)
        return self._roma_matcher

    def detect_points_matchanything(self, src_img, dst_img, **kwargs):
        matcher = self._get_roma_matcher(**kwargs)
        return matcher.detect_points(src_img, dst_img)
```

## References

- **MatchAnything Paper**: [Link to paper if available]
- **ROMA Architecture**: Uses DINOv2 + ConvRefiner for dense matching
- **Original Repository**: Check MatchAnything folder for full implementation
- **Model Weights**: [Google Drive](https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view)

## License

ROMA integration follows the licensing of the MatchAnything project. Please refer to the MatchAnything repository for license details.

## Support

For issues specific to the integration:
1. Check this documentation
2. Run `roma_example.py` to verify installation
3. Check logs for detailed error messages

For issues with the underlying ROMA model:
- Refer to the MatchAnything repository
- Check the original ROMA implementation

---

**Last Updated**: 2025-01-07
**Version**: 1.0
