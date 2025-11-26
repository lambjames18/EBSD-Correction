"""
presenter.py - MVP Presenter for Distortion Correction Application

This module acts as the intermediary between the model and view layers.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
from enum import Enum
import numpy as np

from models import (
    Point,
    PointManager,
    ImageLoader,
    ImageWriter,
    TransformManager,
    ImageProcessor,
    ProjectManager,
    ImageData,
    TransformType,
    DataFormat,
)

# Configure logging
logger = logging.getLogger(__name__)


class ViewMode(Enum):
    """Enumeration of view modes."""

    SPLIT_HORIZONTAL = "horizontal"
    SPLIT_VERTICAL = "vertical"
    OVERLAY = "overlay"
    SIDE_BY_SIDE = "side_by_side"


class CropMode(Enum):
    """Enumeration of crop modes."""

    SOURCE = "src"
    DESTINATION = "dst"
    NONE = "none"


class ApplicationPresenter:
    """Main presenter that coordinates between model and view components."""

    def __init__(self):
        # Model components
        self.point_manager = PointManager()
        self.transform_manager = TransformManager()
        self.project_manager = ProjectManager()
        self.image_processor = ImageProcessor()
        self.image_writer = ImageWriter()

        # Data storage
        self.source_image: Optional[ImageData] = None
        self.destination_image: Optional[ImageData] = None
        self.current_slice = 0
        self.current_source_mode = "Intensity"
        self.current_dest_mode = "Intensity"

        # State flags
        self.is_3d_mode = False
        self.clahe_active_source = False
        self.clahe_active_dest = False
        self.match_resolutions = False

        # View reference (will be set by view)
        self.view = None

        # Paths
        self.source_points_path: Optional[Path] = None
        self.dest_points_path: Optional[Path] = None

        logger.info("ApplicationPresenter initialized")

    def set_view(self, view) -> None:
        """Set the view reference."""
        self.view = view

    # ========== Data Loading ==========

    def load_source_image(
        self, path: Union[str, Path, List[Path], List[str]], resolution: float = 1.0
    ) -> bool:
        """Load source (distorted) image."""
        try:
            self.source_image = ImageLoader.load(path, resolution)
            self.is_3d_mode = self.source_image.shape[0] > 1
            if self.destination_image and self.is_3d_mode:
                if self.destination_image.shape[0] != self.source_image.shape[0]:
                    self.source_image = None
                    self.is_3d_mode = False
                    raise ValueError(
                        "Source and destination images must have the same number of slices for 3D mode."
                    )

            # Set default points path if not set
            if self.source_points_path is None:
                if type(path) in {list, tuple}:
                    path = Path(path[0])
                self.source_points_path = path.parent / "distorted_pts.txt"

            if type(path) in {list, tuple}:
                logger.info(f"Loaded source image stack: {len(path)} files")
            else:
                logger.info(f"Loaded source image: {path}")
            self._notify_view_data_loaded()
            return True

        except Exception as e:
            logger.error(f"Failed to load source image: {e} ({parse_error()})")
            self._notify_view_error(f"Failed to load source image: {str(e)}")
            return False

    def load_destination_image(
        self, path: Union[Path, List[Path]], resolution: float = 1.0
    ) -> bool:
        """Load destination (control) image."""
        try:
            self.destination_image = ImageLoader.load(path, resolution)
            self.is_3d_mode = self.destination_image.shape[0] > 1
            if self.source_image and self.is_3d_mode:
                if self.source_image.shape[0] != self.destination_image.shape[0]:
                    self.destination_image = None
                    self.is_3d_mode = False
                    raise ValueError(
                        "Source and destination images must have the same number of slices for 3D mode."
                    )

            # Set default points path if not set
            if self.dest_points_path is None:
                if type(path) in {list, tuple}:
                    path = Path(path[0])
                self.dest_points_path = path.parent / "control_pts.txt"

            if type(path) in {list, tuple}:
                logger.info(f"Loaded destination image stack: {len(path)} files")
            else:
                logger.info(f"Loaded destination image: {path}")
            self._notify_view_data_loaded()
            return True

        except Exception as e:
            logger.error(f"Failed to load destination image: {e} ({parse_error()})")
            self._notify_view_error(f"Failed to load destination image: {str(e)}")
            return False

    def load_points(
        self, source_path: Optional[Path] = None, dest_path: Optional[Path] = None
    ) -> bool:
        """Load control points from files."""
        try:
            src_path = source_path or self.source_points_path
            dst_path = dest_path or self.dest_points_path

            if src_path and dst_path:
                self.point_manager.load_from_file(src_path, dst_path)
                logger.info("Loaded control points")
                self._notify_view_points_changed()
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to load points: {e}")
            self._notify_view_error(f"Failed to load points: {str(e)}")
            return False

    # ========== Point Management ==========

    def add_point(self, source: str, x: int, y: int) -> None:
        """Add a control point."""
        try:
            point = Point(x, y, self.current_slice)

            if source == "source" and self.source_image is not None:
                # Need to add corresponding point in destination
                # For now, add at same location (user will adjust)
                self.point_manager.source_points.add_point(point)
                self._notify_view_request_corresponding_point("destination")
            elif source == "destination" and self.destination_image is not None:
                # Correct for matched resolutions
                if self.match_resolutions:
                    src_res, dst_res = self.get_resolutions()
                    res_scale = src_res / dst_res
                    point = Point(
                        int(x * res_scale), int(y * res_scale), self.current_slice
                    )
                # Add destination point
                self.point_manager.destination_points.add_point(point)

            self._save_points()
            self._notify_view_points_changed()

        except Exception as e:
            logger.error(f"Failed to add point: {e}")
            self._notify_view_error(f"Failed to add point: {str(e)}")

    def remove_point(self, point_index: int) -> None:
        """Remove a control point."""
        try:
            success = self.point_manager.remove_point_pair(
                self.current_slice, point_index
            )

            if success:
                self._save_points()
                self._notify_view_points_changed()

        except Exception as e:
            logger.error(f"Failed to remove point: {e}")
            self._notify_view_error(f"Failed to remove point: {str(e)}")

    def clear_points(self, slice_only: bool = True) -> None:
        """Clear control points."""
        try:
            if slice_only:
                self.point_manager.clear_points(self.current_slice)
            else:
                self.point_manager.clear_points()

            self._save_points()
            self._notify_view_points_changed()

        except Exception as e:
            logger.error(f"Failed to clear points: {e}")
            self._notify_view_error(f"Failed to clear points: {str(e)}")

    def get_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current points for display."""
        return self.point_manager.get_point_pairs(self.current_slice)

    def undo(self) -> None:
        """Undo last point operation."""
        if self.point_manager.undo():
            self._notify_view_points_changed()

    def redo(self) -> None:
        """Redo last undone point operation."""
        if self.point_manager.redo():
            self._notify_view_points_changed()

    # ========== Image Processing ==========

    def toggle_clahe(self, source: str) -> None:
        """Toggle CLAHE for source or destination image."""
        if source == "source":
            self.clahe_active_source = not self.clahe_active_source
        else:
            self.clahe_active_dest = not self.clahe_active_dest

        self._notify_view_update_display()

    def toggle_match_resolutions(self):
        self.match_resolutions = not self.match_resolutions
        self._notify_view_update_display()

    def get_current_images(
        self,
        scale: float = None,
        src_scale: float = None,
        dst_scale: float = None,
        normalize=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get current images for display."""
        if scale is not None:
            src_scale = scale
            dst_scale = scale
        elif src_scale is None and dst_scale is not None:
            src_scale = dst_scale
            logger.warning("Source scale not provided; using destination scale")
        elif dst_scale is None and src_scale is not None:
            dst_scale = src_scale
            logger.warning("Destination scale not provided; using source scale")
        elif src_scale is None and dst_scale is None:
            src_scale = 1.0
            dst_scale = 1.0
            logger.info("Scaling factors not provided; using 1.0 for both")

        # Get the current source image
        if not self.source_image:
            src_img = None
        else:
            try:
                src_img = self.source_image.get_slice(
                    self.current_source_mode, self.current_slice
                )
                channels = src_img.shape[2]
                src_img = np.squeeze(src_img)

                # Apply CLAHE if active
                if self.clahe_active_source:
                    src_img = self.image_processor.apply_clahe(src_img)

                # Resize if needed
                if src_scale != 1.0:
                    src_img = self.image_processor.resize_image(src_img, src_scale)

                if src_img.ndim == 2:
                    src_img = src_img.reshape(src_img.shape + (channels,))

                # Normalize to uint8
                if normalize:
                    src_img = self.image_processor.normalize_to_uint8(src_img)

            except Exception as e:
                logger.error(f"Failed to get current source image: {e}")
                src_img = None

        # Get the current destination image
        if not self.destination_image:
            dst_img = None
        else:
            try:
                dst_img = self.destination_image.get_slice(
                    self.current_dest_mode, self.current_slice
                )
                channels = dst_img.shape[2]
                dst_img = np.squeeze(dst_img)

                # Reconcile image resolutions (not done for EBSD)
                if self.match_resolutions:
                    downscale = (
                        self.destination_image.resolution / self.source_image.resolution
                    )
                    dst_img = self.image_processor.resize_image(dst_img, downscale)

                # Apply CLAHE if active
                if self.clahe_active_dest:
                    dst_img = self.image_processor.apply_clahe(dst_img)

                # Resize if needed
                if dst_scale != 1.0:
                    dst_img = self.image_processor.resize_image(dst_img, dst_scale)

                if dst_img.ndim == 2:
                    dst_img = dst_img.reshape(dst_img.shape + (channels,))

                # Normalize to uint8
                if normalize:
                    dst_img = self.image_processor.normalize_to_uint8(dst_img)

            except Exception as e:
                logger.error(f"Failed to get current destination image: {e}")
                dst_img = None

        return src_img, dst_img

    # ========== Transformation ==========

    def apply_transform(
        self,
        transform_type: TransformType,
        crop_mode: CropMode = CropMode.NONE,
        normalize: bool = True,
        preview: bool = False,
        return_images: bool = False,
    ) -> Optional[np.ndarray]:
        """Apply transformation to current image/slice."""
        try:
            src_points, dst_points = self.point_manager.get_point_pairs(
                self.current_slice
            )

            if src_points.size == 0 or dst_points.size == 0:
                self._notify_view_error("No control points defined for transformation")
                return None

            # Correct points for matched resolutions
            if self.match_resolutions:
                src_res, dst_res = self.get_resolutions()
                res_scale = dst_res / src_res
                dst_points = np.array(
                    [(p[0] * res_scale, p[1] * res_scale) for p in dst_points]
                )

            # Get current source image
            src_img, dst_img = self.get_current_images(normalize=normalize)

            # Estimate transform
            output_shape = dst_img.shape[:2]
            tform = self.transform_manager.estimate_transform(
                src_points, dst_points, transform_type, size=output_shape
            )

            # Apply transform
            warped = self.transform_manager.apply_transform(
                src_img, tform, output_shape
            )

            # Apply cropping if requested
            if crop_mode != CropMode.NONE:
                warped, dst_img = self._apply_cropping(
                    warped, dst_img, src_img.shape, crop_mode
                )

            if preview:
                self._notify_view_show_preview(warped, dst_img)

            if return_images:
                return warped, src_img, dst_img

        except Exception as e:
            logger.error(f"Failed to apply transform: {e}")
            self._notify_view_error(f"Failed to apply transform: {str(e)}")
            return None

    def apply_transform_3d(
        self, transform_type: TransformType, crop_mode: CropMode = CropMode.SOURCE
    ) -> Optional[np.ndarray]:
        """Apply transformation to entire 3D stack."""
        try:
            # Get all points
            src_points, dst_points = self.point_manager.get_point_pairs()

            if src_points.size == 0 or dst_points.size == 0:
                self._notify_view_error("No control points defined for transformation")
                return None

            # Get image stacks
            src_stack = self.source_image.data[self.current_source_mode]
            dst_stack = self.destination_image.data[self.current_dest_mode]

            # Apply transformation
            output_shape = dst_stack.shape[1:3]
            warped_stack = self.transform_manager.apply_transform_stack(
                src_stack, src_points, dst_points, transform_type, output_shape
            )

            # Apply cropping if needed
            if crop_mode != CropMode.NONE:
                # Implement 3D cropping logic here
                pass

            return warped_stack

        except Exception as e:
            logger.error(f"Failed to apply 3D transform: {e}")
            self._notify_view_error(f"Failed to apply 3D transform: {str(e)}")
            return None

    def export_data(
        self,
        path: Path,
        data_format: DataFormat,
        crop_mode: CropMode,
        transform_type: TransformType,
    ) -> bool:
        """Export corrected image data."""
        try:
            # For image export, we just warp the current mode and save the result
            if data_format in {DataFormat.IMAGE, DataFormat.RAW_IMAGE}:
                warped_img, src_img, dst_img = self.apply_transform(
                    transform_type,
                    crop_mode,
                    normalize=False,
                    preview=False,
                    return_images=True,
                )
                if warped_img is None:
                    return False
                if data_format == DataFormat.IMAGE:
                    warped_img = self.image_processor.normalize_to_uint8(warped_img)
                    src_img = self.image_processor.normalize_to_uint8(src_img)
                    dst_img = self.image_processor.normalize_to_uint8(dst_img)

                self.image_writer.save_image(warped_img, path)
                self.image_writer.save_image(
                    src_img, path.with_name(path.stem + "_src" + path.suffix)
                )
                self.image_writer.save_image(
                    dst_img, path.with_name(path.stem + "_dst" + path.suffix)
                )
                return True

            elif data_format == DataFormat.ANG:
                if self.source_image.path.suffix.lower() != ".ang":
                    self._notify_view_error(
                        "Source image is not in .ang format; cannot export as .ang"
                    )
                    return False
                # loop over all modalities and export
                warped_imgs = {}
                src_imgs = {}

                for mode in self.source_image.modalities:
                    if mode.lower() == "eulerangles":
                        continue
                    self.set_source_mode(mode)
                    self._notify_view_update_display()
                    warped, src, dst_img = self.apply_transform(
                        transform_type,
                        crop_mode,
                        return_images=True,
                    )
                    if warped is None:
                        return False
                    warped_imgs[mode] = warped
                    src_imgs[mode] = src

                # Save dst image
                self.image_writer.save_image(
                    dst_img, path.with_name(path.stem + "_dst.tif")
                )

                # Save warped images as .ang
                self.image_writer.save_ang(
                    warped_imgs,
                    path.with_name(path.stem + ".ang"),
                    self.source_image.metadata["header"],
                    self.source_image.resolution,
                )
                self.image_writer.save_ang(
                    src_imgs,
                    path.with_name(path.stem + "_src.ang"),
                    self.source_image.metadata["header"],
                    self.source_image.resolution,
                )

            elif data_format == DataFormat.H5:
                pass

            elif data_format == DataFormat.DREAM3D:
                pass
            # Apply transformation
            if self.is_3d_mode:
                warped_stack = self.apply_transform_3d(transform_type, crop_mode)
                if warped_stack is None:
                    return False
                self.image_processor.export_data(warped_stack, path, data_format)
            else:
                warped_img, src_img, dst_img = self.apply_transform(
                    transform_type, crop_mode, return_images=True
                )
                if warped_img is None:
                    return False
            return True

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            self._notify_view_error(f"Failed to export data: {str(e)}")
            return False

    def export_transform(self, path: Path, transform_type: TransformType) -> bool:
        """Export transformation parameters."""
        try:
            src_points, dst_points = self.point_manager.get_point_pairs(
                self.current_slice
            )

            if src_points.size == 0 or dst_points.size == 0:
                self._notify_view_error("No control points defined for export")
                return False

            # Get destination shape for TPS
            dst_img = self.destination_image.get_slice(
                self.current_dest_mode, self.current_slice
            )
            output_shape = dst_img.shape[:2]

            # Estimate transform
            tform = self.transform_manager.estimate_transform(
                src_points, dst_points, transform_type, size=output_shape
            )

            # Export
            format = path.suffix[1:] if path.suffix else "npy"
            self.transform_manager.export_transform(tform, path, format)

            return True

        except Exception as e:
            logger.error(f"Failed to export transform: {e}")
            self._notify_view_error(f"Failed to export transform: {str(e)}")
            return False

    # ========== Project Management ==========

    def save_project(self, path: Path) -> bool:
        """Save current project."""
        try:
            settings = {
                "current_slice": self.current_slice,
                "source_mode": self.current_source_mode,
                "dest_mode": self.current_dest_mode,
                "clahe_source": self.clahe_active_source,
                "clahe_dest": self.clahe_active_dest,
                "is_3d": self.is_3d_mode,
                "match_resolutions": self.match_resolutions,
                "source_resolution": self.source_image.resolution,
                "destination_resolution": self.destination_image.resolution,
            }

            self.project_manager.save_project(
                path,
                self.point_manager,
                self.source_image.path,
                self.destination_image.path,
                settings,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            self._notify_view_error(f"Failed to save project: {str(e)}")
            return False

    def load_project(self, path: Path) -> bool:
        """Load project from file."""
        try:
            project_data = self.project_manager.load_project(path)
            settings = project_data.get("settings", {})

            # Load images
            self.load_source_image(
                Path(project_data["source_image"]),
                settings.get("source_resolution", 1.0),
            )
            self.load_destination_image(
                Path(project_data["destination_image"]),
                settings.get("destination_resolution", 1.0),
            )

            # Load points
            self.point_manager.load_from_json(project_data)

            # Load settings
            self.current_slice = settings.get("current_slice", 0)
            self.current_source_mode = settings.get("source_mode", "Intensity")
            self.current_dest_mode = settings.get("dest_mode", "Intensity")
            self.clahe_active_source = settings.get("clahe_source", False)
            self.clahe_active_dest = settings.get("clahe_dest", False)
            self.is_3d_mode = settings.get("is_3d", False)
            self.match_resolutions = settings.get("match_resolutions", False)

            self._notify_view_project_loaded()
            return True

        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            self._notify_view_error(f"Failed to load project: {str(e)}")
            return False

    # ========== Navigation ==========

    def set_current_slice(self, slice_idx: int) -> None:
        """Set current slice for 3D data."""
        if self.source_image and 0 <= slice_idx < self.source_image.shape[0]:
            self.current_slice = slice_idx
            self._notify_view_update_display()

    def set_source_mode(self, mode: str) -> None:
        """Set display mode for source image."""
        if self.source_image and mode in self.source_image.modalities:
            self.current_source_mode = mode
            self._notify_view_update_display()

    def set_destination_mode(self, mode: str) -> None:
        """Set display mode for destination image."""
        if self.destination_image and mode in self.destination_image.modalities:
            self.current_dest_mode = mode
            self._notify_view_update_display()

    def set_image_resolutions(self, src_res: float, dst_res: float) -> None:
        """Set image resolutions. This reloads the data with the correct resolutions."""
        ### TODO: Fix this
        if self.source_image:
            self.load_source_image(self.source_image.path, src_res)
        if self.destination_image:
            self.load_destination_image(self.destination_image.path, dst_res)

    def get_slice_range(self) -> Tuple[int, int]:
        """Get valid slice range."""
        if self.source_image:
            return 0, self.source_image.shape[0] - 1
        return 0, 0

    def get_source_modalities(self) -> List[str]:
        """Get available source image modalities."""
        if self.source_image:
            return self.source_image.modalities
        return []

    def get_destination_modalities(self) -> List[str]:
        """Get available destination image modalities."""
        if self.destination_image:
            return self.destination_image.modalities
        return []

    def get_resolutions(self) -> Tuple[float, float]:
        """Get image resolutions."""
        src_res = self.source_image.resolution if self.source_image else 1.0
        dst_res = self.destination_image.resolution if self.destination_image else 1.0
        return src_res, dst_res

    # ========== Private Helper Methods ==========

    def _save_points(self) -> None:
        """Save points to file."""
        if self.source_points_path and self.dest_points_path:
            try:
                self.point_manager.save_to_file(
                    self.source_points_path, self.dest_points_path
                )
            except Exception as e:
                logger.error(f"Failed to save points: {e}")

    def _apply_cropping(
        self,
        warped: np.ndarray,
        reference: np.ndarray,
        original_shape: Tuple,
        crop_mode: CropMode,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply cropping to match grids."""
        if crop_mode == CropMode.SOURCE:
            # Crop to match source grid size
            # Find centroid of warped non-zero region
            nonzero = np.where(warped > 0)
            if len(nonzero[0]) > 0:
                centroid_r = int(np.mean(nonzero[0]))
                centroid_c = int(np.mean(nonzero[1]))

                # Calculate crop region
                r_start = max(0, centroid_r - original_shape[0] // 2)
                r_end = min(warped.shape[0], r_start + original_shape[0])
                c_start = max(0, centroid_c - original_shape[1] // 2)
                c_end = min(warped.shape[1], c_start + original_shape[1])
            else:
                r_start = 0
                r_end = original_shape[0]
                c_start = 0
                c_end = original_shape[1]
            warped = warped[r_start:r_end, c_start:c_end]
            reference = reference[r_start:r_end, c_start:c_end]

        elif crop_mode == CropMode.DESTINATION:
            # Crop to destination size
            warped = warped[: reference.shape[0], : reference.shape[1]]

        return warped, reference

    # ========== View Notification Methods ==========

    def _notify_view_data_loaded(self) -> None:
        """Notify view that data has been loaded."""
        if self.view:
            self.view.on_data_loaded()

    def _notify_view_points_changed(self) -> None:
        """Notify view that points have changed."""
        if self.view:
            self.view.on_points_changed()

    def _notify_view_update_display(self) -> None:
        """Notify view to update display."""
        if self.view:
            self.view.on_display_update_needed()

    def _notify_view_error(self, message: str) -> None:
        """Notify view of an error."""
        if self.view:
            self.view.on_error(message)

    def _notify_view_show_preview(
        self, warped: np.ndarray, reference: np.ndarray
    ) -> None:
        """Notify view to show transformation preview."""
        if self.view:
            self.view.on_show_preview(warped, reference)

    def _notify_view_project_loaded(self) -> None:
        """Notify view that a project has been loaded."""
        if self.view:
            self.view.on_project_loaded()

    def _notify_view_request_corresponding_point(self, target: str) -> None:
        """Notify view to request corresponding point from user."""
        if self.view:
            self.view.on_request_corresponding_point(target)


def parse_error():
    import sys
    import os

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    return (exc_type, fname, exc_tb.tb_lineno)
