"""
roma_example.py - Example usage of the ROMA matcher

This script demonstrates how to use the ROMA matcher for automatic
control point detection between image pairs.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from roma_matcher import RomaMatcher, detect_points_roma
from roma_config import get_preset_config, print_config


def example_basic_usage():
    """
    Basic example: Detect points between two images.
    """
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Load your images (replace with actual image paths)
    # For this example, we'll create dummy images
    source_image = np.random.rand(512, 512).astype(np.float32)
    dest_image = np.random.rand(512, 512).astype(np.float32)

    # Option 1: Create a matcher instance (reusable)
    matcher = RomaMatcher(
        checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt",
        confidence_threshold=0.1,
        num_samples=5000,
    )

    # Detect points
    src_points, dst_points, confidences = matcher.detect_points(
        source_image,
        dest_image,
        ransac_filter=True,  # Apply RANSAC to filter outliers
        ransac_threshold=5.5,
    )

    print(f"Detected {len(src_points)} matches")
    print(f"Mean confidence: {np.mean(confidences):.3f}")
    print(f"First 5 source points:\n{src_points[:5]}")
    print(f"First 5 destination points:\n{dst_points[:5]}")

    # Option 2: One-time convenience function
    src_pts, dst_pts, conf = detect_points_roma(
        source_image,
        dest_image,
        confidence_threshold=0.1,
        ransac_filter=True,
    )

    print(f"\nUsing convenience function: {len(src_pts)} matches")


def example_preset_config():
    """
    Example: Using preset configurations for different data types.
    """
    print("\n" + "=" * 60)
    print("Example 2: Using Preset Configurations")
    print("=" * 60)

    # Get EBSD-optimized configuration
    ebsd_config = get_preset_config('ebsd')
    print_config(ebsd_config)

    # Load images (dummy data for example)
    source_image = np.random.rand(512, 512).astype(np.float32)
    dest_image = np.random.rand(512, 512).astype(np.float32)

    # Create matcher with EBSD preset
    matcher = RomaMatcher(
        checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt",
        **ebsd_config,  # Unpack config as kwargs (note: need to map keys correctly)
    )

    # Or manually set important parameters
    matcher_manual = RomaMatcher(
        checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt",
        confidence_threshold=0.08,  # From ebsd_config
        num_samples=5000,
        resize_by_stretch=True,
        coarse_resolution=(560, 560),
        upsample_resolution=(896, 896),
    )

    src_points, dst_points, _ = matcher_manual.detect_points(source_image, dest_image)
    print(f"\nEBSD preset: Detected {len(src_points)} matches")


def example_with_metadata():
    """
    Example: Get detailed metadata about the matches.
    """
    print("\n" + "=" * 60)
    print("Example 3: Getting Detailed Metadata")
    print("=" * 60)

    source_image = np.random.rand(512, 512).astype(np.float32)
    dest_image = np.random.rand(512, 512).astype(np.float32)

    matcher = RomaMatcher(
        checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt",
        confidence_threshold=0.1,
    )

    # Get matches with metadata
    result = matcher.detect_points_with_metadata(
        source_image,
        dest_image,
        ransac_filter=True,
        ransac_threshold=5.5,
    )

    print("\nMatch metadata:")
    print(f"  Number of matches: {result['num_matches']}")
    print(f"  Mean confidence: {result['mean_confidence']:.3f}")
    print(f"  RANSAC applied: {result['ransac_applied']}")
    print(f"  Source points shape: {result['source_points'].shape}")
    print(f"  Destination points shape: {result['destination_points'].shape}")


def example_visualization():
    """
    Example: Visualize the detected matches.
    """
    print("\n" + "=" * 60)
    print("Example 4: Visualizing Matches")
    print("=" * 60)

    # Create synthetic test images with known transformation
    h, w = 512, 512
    source_image = np.random.rand(h, w).astype(np.float32)

    # Create destination image with simple affine transformation
    from scipy.ndimage import affine_transform
    matrix = [[1.1, 0.05], [0.05, 1.1]]
    offset = [10, 15]
    dest_image = affine_transform(source_image, matrix, offset=offset, order=1)

    matcher = RomaMatcher(
        checkpoint_path="MatchAnything/weights/matchanything_roma.ckpt",
        confidence_threshold=0.1,
    )

    src_points, dst_points, confidences = matcher.detect_points(
        source_image,
        dest_image,
        ransac_filter=True,
    )

    print(f"Detected {len(src_points)} matches")

    # Visualize matches
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Source image with points
    axes[0].imshow(source_image, cmap='gray')
    axes[0].scatter(src_points[:, 0], src_points[:, 1], c='red', s=10, alpha=0.5)
    axes[0].set_title(f'Source Image ({len(src_points)} points)')
    axes[0].axis('off')

    # Destination image with points
    axes[1].imshow(dest_image, cmap='gray')
    axes[1].scatter(dst_points[:, 0], dst_points[:, 1], c='blue', s=10, alpha=0.5)
    axes[1].set_title(f'Destination Image ({len(dst_points)} points)')
    axes[1].axis('off')

    plt.tight_layout()
    output_path = Path("roma_matches_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    # Also plot matches as lines between images
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Concatenate images horizontally
    combined = np.hstack([source_image, dest_image])
    ax.imshow(combined, cmap='gray')

    # Draw lines connecting matches
    for i in range(min(50, len(src_points))):  # Plot first 50 matches
        x1, y1 = src_points[i]
        x2, y2 = dst_points[i]
        x2 += w  # Offset for concatenated image

        # Color by confidence
        color = plt.cm.hot(confidences[i])
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5, alpha=0.5)

    ax.set_title(f'Matches (first 50 of {len(src_points)}) - Color by confidence')
    ax.axis('off')

    plt.tight_layout()
    output_path = Path("roma_matches_lines.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Line visualization saved to: {output_path}")


def example_integration_with_gui():
    """
    Example: How to integrate ROMA with the PointAutoIdentifier in models.py
    """
    print("\n" + "=" * 60)
    print("Example 5: Integration with PointAutoIdentifier")
    print("=" * 60)

    print("To integrate ROMA into your GUI's PointAutoIdentifier:")
    print("""
    1. Add to models.py:

    from roma_matcher import RomaMatcher

    class PointAutoIdentifier:
        ENGINES = {
            "sift": "detect_points_sift",
            "matchanything": "detect_points_matchanything",  # Add this
        }

        @staticmethod
        def detect_points_matchanything(
            source_image: np.ndarray,
            destination_image: np.ndarray,
            **kwargs
        ) -> Tuple[np.ndarray, np.ndarray]:
            '''Detect matching points between images using ROMA.

            Args:
                source_image: Source image as numpy array
                destination_image: Destination image as numpy array
                **kwargs: Additional parameters:
                    - confidence_threshold: Minimum confidence (default: 0.1)
                    - ransac_filter: Whether to apply RANSAC (default: True)
                    - ransac_threshold: RANSAC threshold in pixels (default: 5.5)
                    - checkpoint_path: Path to model checkpoint

            Returns:
                Tuple of (source_points, destination_points) as numpy arrays
            '''
            # Extract parameters
            confidence_threshold = kwargs.get('confidence_threshold', 0.1)
            ransac_filter = kwargs.get('ransac_filter', True)
            ransac_threshold = kwargs.get('ransac_threshold', 5.5)
            checkpoint_path = kwargs.get(
                'checkpoint_path',
                'MatchAnything/weights/matchanything_roma.ckpt'
            )

            try:
                # Create matcher (consider caching this for reuse)
                matcher = RomaMatcher(
                    checkpoint_path=checkpoint_path,
                    confidence_threshold=confidence_threshold,
                )

                # Detect points
                src_points, dst_points, _ = matcher.detect_points(
                    source_image,
                    destination_image,
                    ransac_filter=ransac_filter,
                    ransac_threshold=ransac_threshold,
                )

                logger.info(f"ROMA detected {len(src_points)} matches")

                if len(src_points) < 4:
                    logger.warning("Insufficient matches found")
                    return np.array([]), np.array([])

                return src_points.astype(int), dst_points.astype(int)

            except Exception as e:
                logger.error(f"ROMA point detection failed: {e}")
                return np.array([]), np.array([])

    2. Then use in your GUI:
       src_pts, dst_pts = PointAutoIdentifier.detect_points(
           source_image,
           destination_image,
           method='matchanything',  # Use ROMA
           confidence_threshold=0.1,
           ransac_filter=True,
       )
    """)


if __name__ == "__main__":
    print("\nROMA Matcher Examples")
    print("=" * 60)
    print("Note: These examples use dummy data for demonstration.")
    print("Replace with actual image loading for real use cases.")
    print("=" * 60)

    # Run examples
    try:
        example_basic_usage()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_preset_config()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_with_metadata()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_visualization()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    example_integration_with_gui()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
