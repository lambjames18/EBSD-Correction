# ROMA Quick Start Guide

Quick reference for using ROMA in your EBSD-Correction GUI.

## Installation (5 minutes)

1. **Install dependencies**:
   ```bash
   pip install torch torchvision kornia einops loguru pillow scikit-image scipy
   ```

2. **Download checkpoint** (1.5 GB):
   - Link: https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view
   - Place in: `MatchAnything/weights/matchanything_roma.ckpt`

3. **Verify installation**:
   ```bash
   python test_roma_installation.py
   ```

## Basic Usage (3 lines of code)

```python
from roma_matcher import RomaMatcher

matcher = RomaMatcher(checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt")
src_points, dst_points, confidences = matcher.detect_points(source_image, dest_image)
```

## Use in Existing Code

### With PointAutoIdentifier

```python
from models import PointAutoIdentifier

# Just change the method parameter!
src_pts, dst_pts = PointAutoIdentifier.detect_points(
    source_image,
    destination_image,
    method='matchanything',  # or 'roma'
)
```

### Direct Usage

```python
from roma_matcher import detect_points_roma

# One-line convenience function
src_pts, dst_pts, conf = detect_points_roma(source_image, dest_image)
```

## Common Parameters

```python
matcher = RomaMatcher(
    confidence_threshold=0.1,    # Lower = more matches (0.05-0.15)
    num_samples=5000,            # More = slower but more matches
    device="cuda",               # "cuda" or "cpu"
)

src_pts, dst_pts, _ = matcher.detect_points(
    source_image,
    dest_image,
    ransac_filter=True,          # Remove outliers
    ransac_threshold=5.5,        # Pixels (lower = stricter)
)
```

## Presets for Different Data

### EBSD Data
```python
matcher = RomaMatcher(
    checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt",
    confidence_threshold=0.08,
    coarse_resolution=(560, 560),
    upsample_resolution=(896, 896),
)
```

### Cross-Modality (SEM ↔ EBSD)
```python
matcher = RomaMatcher(
    checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt",
    confidence_threshold=0.12,
    normalize_images=True,       # Important for different modalities!
)
```

### Fast Processing
```python
matcher = RomaMatcher(
    checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt",
    confidence_threshold=0.15,
    num_samples=2000,
    coarse_resolution=(448, 448),
)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Lower `coarse_resolution` to (448, 448) or use `device="cpu"` |
| Too few matches | Lower `confidence_threshold` to 0.05 or disable `ransac_filter` |
| Inaccurate matches | Enable `ransac_filter=True` or increase `confidence_threshold` |
| Slow on CPU | Use GPU or lower `coarse_resolution` and `num_samples` |
| Import error | Run `test_roma_installation.py` to diagnose |

## Performance

| Mode | Speed (GPU) | Memory | Accuracy |
|------|-------------|--------|----------|
| Fast | 0.5-1 sec | 4 GB | Good |
| Balanced | 1-2 sec | 6 GB | Very Good |
| Accurate | 2-4 sec | 12 GB | Excellent |

## Full Documentation

- **Detailed guide**: `ROMA_INTEGRATION.md`
- **Examples**: `roma_example.py`
- **Test script**: `test_roma_installation.py`

## Support

1. Run `python test_roma_installation.py` to diagnose issues
2. Check `ROMA_INTEGRATION.md` for detailed troubleshooting
3. Review examples in `roma_example.py`
