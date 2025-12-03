"""
models.py - Data Models and Business Logic for Distortion Correction

This module contains the core business logic separated from the UI.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import h5py
from skimage import io, transform
import torch
from torchvision.transforms.functional import resize as RESIZE
from torchvision.transforms import InterpolationMode
from kornia.enhance import equalize_clahe

# Configure logging
logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Enumeration of supported data formats."""

    ANG = "ang"
    H5 = "h5"
    DREAM3D = "dream3d"
    IMAGE = "image"
    RAW_IMAGE = "raw_image"


class TransformType(Enum):
    """Enumeration of available transformation types."""

    TPS = "tps"
    TPS_AFFINE = "tps_affine"


@dataclass
class Point:
    """Represents a control point."""

    x: float
    y: float
    slice_idx: int = 0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y] or [slice, x, y]."""
        if self.slice_idx == 0:
            return np.array([self.x, self.y])
        return np.array([self.slice_idx, self.x, self.y])


@dataclass
class PointSet:
    """Collection of control points for an image."""

    points: Dict[int, List[Point]] = field(default_factory=dict)

    def add_point(self, point: Point) -> None:
        """Add a point to the set."""
        if point.slice_idx not in self.points:
            self.points[point.slice_idx] = []
        self.points[point.slice_idx].append(point)

    def remove_point(self, slice_idx: int, point_idx: int) -> bool:
        """Remove a point by index."""
        try:
            if slice_idx in self.points and 0 <= point_idx < len(
                self.points[slice_idx]
            ):
                self.points[slice_idx].pop(point_idx)
                logger.debug("Point index in range for removal")
            else:
                logger.debug("Point index out of range for removal")
            return True
        except Exception as e:
            logger.error("Failed to remove point from PointSet.")
            return False

    def get_points_array(self, slice_idx: Optional[int] = None) -> np.ndarray:
        """Get points as numpy array for a specific slice or all slices."""
        if slice_idx is not None:
            if slice_idx not in self.points:
                return np.array([])
            return np.array([[p.x, p.y] for p in self.points[slice_idx]])

        # Get all points with slice information
        all_points = []
        for slice_idx, slice_points in self.points.items():
            for point in slice_points:
                all_points.append([slice_idx, point.x, point.y])
        return np.array(all_points) if all_points else np.array([])

    def clear(self, slice_idx: Optional[int] = None) -> None:
        """Clear points for a specific slice or all slices."""
        if slice_idx is not None:
            self.points.pop(slice_idx, None)
        else:
            self.points.clear()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            str(slice_idx): [[p.x, p.y] for p in points]
            for slice_idx, points in self.points.items()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PointSet":
        """Create PointSet from dictionary."""
        point_set = cls()
        for slice_idx, points in data.items():
            for x, y in points:
                point_set.add_point(Point(x, y, int(slice_idx)))
        return point_set


@dataclass
class ImageData:
    """Container for image data and metadata."""

    data: Dict[str, np.ndarray]
    resolution: float
    path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the first data array."""
        if self.data:
            first_key = next(iter(self.data.keys()))
            return self.data[first_key].shape
        return (0, 0, 0)

    @property
    def modalities(self) -> List[str]:
        """Get list of available modalities."""
        return list(self.data.keys())

    def get_slice(self, modality: str, slice_idx: int = 0) -> np.ndarray:
        """Get a specific slice of data."""
        if modality not in self.data:
            raise KeyError(f"Modality '{modality}' not found in image data")

        data = self.data[modality]
        if slice_idx >= data.shape[0]:
            raise IndexError(
                f"Slice index {slice_idx} out of range for shape {data.shape}"
            )

        return data[slice_idx]

    def add_modality(self, modality_name: str, data: np.ndarray) -> None:
        """Add a new modality to the image data."""
        if modality_name in self.data:
            raise ValueError(f"Modality '{modality_name}' already exists")

        # Verify shape compatibility (should match existing modality shapes)
        if self.data:
            first_key = next(iter(self.data.keys()))
            expected_shape = self.data[first_key].shape[:3]  # slices, height, width
            if data.shape[:3] != expected_shape:
                raise ValueError(
                    f"New modality shape {data.shape[:3]} does not match existing shape {expected_shape}"
                )

        self.data[modality_name] = data
        logger.info(f"Added modality '{modality_name}' to image data")


class PointManager:
    """Manages control points for image registration."""

    def __init__(self):
        self.source_points = PointSet()
        self.destination_points = PointSet()
        self._history: List[Tuple[str, Any]] = []
        self._history_index = -1
        self.max_history = 50
        logger.info("PointManager initialized")

    def add_point_pair(self, src_point: Point, dst_point: Point) -> None:
        """Add a pair of corresponding points."""
        self._save_state()
        self.source_points.add_point(src_point)
        self.destination_points.add_point(dst_point)
        logger.debug(f"Added point pair: src={src_point}, dst={dst_point}")

    def remove_point_pair(self, slice_idx: int, point_idx: int) -> bool:
        """Remove a pair of corresponding points."""
        self._save_state()
        src_removed = self.source_points.remove_point(slice_idx, point_idx)
        dst_removed = self.destination_points.remove_point(slice_idx, point_idx)
        logger.debug(
            f"Attempted to remove point pair at slice {slice_idx}, index {point_idx} with success {src_removed} and {dst_removed}"
        )
        success = src_removed and dst_removed

        if success:
            logger.debug(f"Removed point pair at slice {slice_idx}, index {point_idx}")
        else:
            logger.warning(
                f"Failed to remove point pair at slice {slice_idx}, index {point_idx}"
            )

        return success

    def get_point_pairs(
        self, slice_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get corresponding point pairs as numpy arrays."""
        src_points = self.source_points.get_points_array(slice_idx)
        dst_points = self.destination_points.get_points_array(slice_idx)
        return src_points, dst_points

    def clear_points(self, slice_idx: Optional[int] = None) -> None:
        """Clear all points or points for a specific slice."""
        self._save_state()
        self.source_points.clear(slice_idx)
        self.destination_points.clear(slice_idx)
        logger.info(
            f"Cleared points for slice {slice_idx if slice_idx is not None else 'all'}"
        )

    def save_to_file(self, src_path: Path, dst_path: Path) -> None:
        """Save points to text files."""
        try:
            src_array = self.source_points.get_points_array()
            dst_array = self.destination_points.get_points_array()

            if src_array.size > 0:
                np.savetxt(src_path, src_array, fmt="%d", delimiter=" ")
                logger.info(f"Saved {len(src_array)} source points to {src_path}")

            if dst_array.size > 0:
                np.savetxt(dst_path, dst_array, fmt="%d", delimiter=" ")
                logger.info(f"Saved {len(dst_array)} destination points to {dst_path}")

        except Exception as e:
            logger.error(f"Failed to save points: {e}")
            raise

    def load_from_file(self, src_path: Path, dst_path: Path) -> None:
        """Load points from text files."""
        try:
            self.source_points = self._load_points_from_file(src_path)
            self.destination_points = self._load_points_from_file(dst_path)
            logger.info(f"Loaded points from {src_path} and {dst_path}")
        except Exception as e:
            logger.error(f"Failed to load points: {e}")
            raise

    def load_from_json(self, json_data: dict) -> None:
        """Load points from JSON data."""
        try:
            self.source_points = self._load_points_from_json(
                json_data.get("source_points", {})
            )
            self.destination_points = self._load_points_from_json(
                json_data.get("destination_points", {})
            )
            logger.info("Loaded points from JSON data")
        except Exception as e:
            logger.error(f"Failed to load points from JSON: {e}")
            raise

    def _load_points_from_file(self, path: Path) -> PointSet:
        """Load points from a single file."""
        point_set = PointSet()

        if not path.exists():
            logger.warning(f"Point file does not exist: {path}")
            return point_set

        try:
            data = np.loadtxt(path, dtype=int)
            if data.size == 0:
                return point_set

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Handle 2D or 3D points
            if data.shape[1] == 2:
                # 2D points, assume slice 0
                for x, y in data:
                    point_set.add_point(Point(int(x), int(y), 0))
            elif data.shape[1] == 3:
                # 3D points with slice information
                for slice_idx, x, y in data:
                    point_set.add_point(Point(int(x), int(y), int(slice_idx)))
            else:
                raise ValueError(f"Invalid point format in {path}")

        except Exception as e:
            logger.error(f"Error reading point file {path}: {e}")
            raise

        return point_set

    def _load_points_from_json(self, points: dict) -> PointSet:
        """Load points from a numpy array."""
        point_set = PointSet()

        try:
            for slice_idx, pts in points.items():
                for x, y in pts:
                    point_set.add_point(Point(x, y, int(slice_idx)))

        except Exception as e:
            logger.error(f"Error loading points from json: {e}")
            raise

        return point_set

    def _save_state(self) -> None:
        """Save current state for undo functionality."""
        state = {
            "source": self.source_points.to_dict(),
            "destination": self.destination_points.to_dict(),
        }

        # Remove any states after current index
        self._history = self._history[: self._history_index + 1]

        # Add new state
        self._history.append(("state", state))

        # Limit history size
        if len(self._history) > self.max_history:
            self._history.pop(0)
        else:
            self._history_index += 1

    def undo(self) -> bool:
        """Undo last operation."""
        if self._history_index > 0:
            self._history_index -= 1
            state = self._history[self._history_index][1]
            self.source_points = PointSet.from_dict(state["source"])
            self.destination_points = PointSet.from_dict(state["destination"])
            logger.debug("Undid last operation")
            return True
        return False

    def redo(self) -> bool:
        """Redo last undone operation."""
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            state = self._history[self._history_index][1]
            self.source_points = PointSet.from_dict(state["source"])
            self.destination_points = PointSet.from_dict(state["destination"])
            logger.debug("Redid operation")
            return True
        return False


class ImageLoader:
    """Handles loading and preprocessing of various image formats."""

    SUPPORTED_FORMATS = {
        ".ang": "load_ang",
        ".h5": "load_h5",
        ".dream3d": "load_dream3d",
        ".tif": "load_image",
        ".tiff": "load_image",
        ".png": "load_image",
        ".jpg": "load_image",
        ".jpeg": "load_image",
    }

    @classmethod
    def load(
        cls, path: Union[str, Path, List, Tuple], resolution: float = 1.0, modality_name: str = "Intensity"
    ) -> ImageData:
        """Load image data from file."""
        if type(path) not in [list, tuple]:
            path = Path(path)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            suffix = path.suffix.lower()
            if suffix not in cls.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file format: {suffix}")

            loader_method = getattr(cls, cls.SUPPORTED_FORMATS[suffix])

        else:
            loader_method = cls.load_images
            first_path = Path(path[0])
            first_suffix = first_path.suffix.lower()
            for i in range(len(path)):
                _p = Path(path[i])

                if not _p.exists():
                    raise FileNotFoundError(f"File not found: {path}")

                suffix = _p.suffix.lower()
                if suffix not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                    raise ValueError(
                        f"When providing a list of images, all paths must be of image type"
                    )

                if suffix != suffix.lower():
                    raise ValueError(
                        f"When prividing a list of images, all images must have the same extension"
                    )

                path[i] = _p

        try:
            logger.info(f"Loading {suffix} file(s)")
            # Pass modality_name for single image files
            if loader_method == cls.load_image:
                data, res, metadata = loader_method(path, modality_name)
            else:
                data, res, metadata = loader_method(path)

            # Use provided resolution if loader didn't return one
            if res is None:
                res = resolution

            if metadata is None:
                metadata = {}

            return ImageData(data=data, resolution=res, path=path, metadata=metadata)

        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            raise

    @staticmethod
    def load_ang(path: Path) -> Tuple[Dict[str, np.ndarray], float]:
        """Load ANG file format."""
        col_names = None
        header = ""
        header_lines = 0
        ncols = nrows = 0
        res = 1.0

        with open(path, "r") as f:
            for line in f:
                header += line
                if "NCOLS_ODD" in line:
                    ncols = int(line.split(": ")[1].strip())
                elif "NROWS" in line:
                    nrows = int(line.split(": ")[1].strip())
                elif "COLUMN_HEADERS" in line:
                    col_names = (
                        line.replace("\n", "").split(": ")[1].strip().split(", ")
                    )
                elif "XSTEP" in line:
                    res = float(line.split(": ")[1].strip())
                elif "HEADER: End" in line:
                    break
                header_lines += 1

        if col_names is None:
            col_names = ["phi1", "PHI", "phi2", "x", "y", "IQ", "CI", "Phase index"]

        raw_data = np.genfromtxt(path, skip_header=header_lines, dtype=float)

        if raw_data.shape[0] != ncols * nrows:
            raise ValueError(
                f"Data size mismatch: expected {ncols*nrows}, got {raw_data.shape[0]}"
            )

        n_entries = raw_data.shape[-1]
        data = raw_data.reshape((nrows, ncols, n_entries))

        out = {}
        for i, name in enumerate(col_names[:n_entries]):
            arr = data[:, :, i]
            arr = np.fliplr(np.rot90(arr, k=3)).T
            out[name] = arr.reshape((1,) + arr.shape + (1,))

        # Create Euler angles array
        if all(k in out for k in ["phi1", "PHI", "phi2"]):
            out["EulerAngles"] = np.stack(
                [out["phi1"], out["PHI"], out["phi2"]], axis=-1
            )

        metadata = {"header": header}
        return out, res, metadata

    @staticmethod
    def load_h5(path: Path) -> Tuple[Dict[str, np.ndarray], float]:
        """Load H5 file format."""
        with h5py.File(path, "r") as h5:
            # Find the EBSD data entry
            if "1" in h5.keys():
                entry = "1"
            elif "Scan 1" in h5.keys():
                entry = "Scan 1"
            else:
                raise ValueError("Could not find EBSD data in the H5 file")

            ebsd_data = h5[f"{entry}/EBSD/Data"]
            nrows = h5[f"{entry}/EBSD/Header/nRows"][0]
            ncols = h5[f"{entry}/EBSD/Header/nColumns"][0]
            res = h5[f"{entry}/EBSD/Header/Step X"][0]

            keys = list(ebsd_data.keys())
            data = {
                key.upper()
                .replace(" ", "-"): ebsd_data[key][...]
                .reshape(1, nrows, ncols, -1)
                for key in keys
            }

            # Create Euler angles array if components exist
            if all(k in data for k in ["PHI1", "PHI", "PHI2"]):
                data["EULERANGLES"] = np.stack(
                    [data["PHI1"], data["PHI"], data["PHI2"]], axis=-1
                ).astype(float)

        return data, res, None

    @staticmethod
    def load_dream3d(path: Path) -> Tuple[Dict[str, np.ndarray], float]:
        """Load DREAM3D file format."""
        with h5py.File(path, "r") as h5:
            if "DataStructure" in h5.keys():
                ebsd_data = h5["DataStructure/DataContainer/CellData"]
                res = np.asarray(
                    h5["DataStructure/DataContainer"].attrs.get("_SPACING")
                )[1]
            elif "DataContainers" in h5.keys():
                ebsd_data = h5["DataContainers/ImageDataContainer/CellData"]
                res = h5["DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"][
                    ...
                ][1]
            else:
                raise ValueError("Could not find EBSD data in the DREAM3D file")

            ebsd_keys = list(ebsd_data.keys())
            data = {key: ebsd_data[key][...] for key in ebsd_keys}

        return data, res, None

    @staticmethod
    def load_image(path: Path, modality_name: str = "Intensity") -> Tuple[Dict[str, np.ndarray], None]:
        """Load standard image formats with optional modality name."""
        im = io.imread(path, as_gray=True).astype(np.float32)

        # Normalize to 0-255 range
        im = np.around((im - np.min(im)) / (np.max(im) - np.min(im)) * 255, 0)
        im = im.astype(np.uint8)

        if im.ndim == 2:
            im = im.reshape((im.shape[0], im.shape[1], 1))

        im = im.reshape((1,) + im.shape)

        return {modality_name: im}, None, None

    @staticmethod
    def load_images(paths: list) -> Tuple[Dict[str, np.ndarray], None]:
        """Load a list of standard image formats."""
        images = np.array(
            [
                io.imread(paths[i], as_gray=True).astype(np.float32)
                for i in range(len(paths))
            ]
        )

        # Put in a channel axis if needed
        if images.ndim == 3:
            images = images.reshape(images.shape + (1,))

        # Normalize to 0-255 range
        # mn = np.min(images, axis=(1, 2, 3), keepdims=True)
        # mx = np.max(images, axis=(1, 2, 3), keepdims=True)
        # rnge = mx - mn
        # images = np.around(255 * (images - mn) / rnge, 0)
        # images = images.astype(np.uint8)

        return {"Intensity": images}, None, None


class ImageWriter:
    """Handles saving images to disk."""

    SUPPORTED_FORMATS = {
        ".ang": "save_ang",
        ".h5": "save_h5",
        ".dream3d": "save_dream3d",
        ".tif": "save_image",
        ".tiff": "save_image",
        ".png": "save_image",
        ".jpg": "save_image",
        ".jpeg": "save_image",
    }

    @classmethod
    def save(
        cls,
        image_data: Union[np.ndarray, Dict[str, np.ndarray]],
        path: Path,
        *args,
        **kwargs,
    ) -> None:
        """Save image data to file."""
        try:
            # Determine file format from path
            ext = path.suffix.lower()
            if ext in cls.SUPPORTED_FORMATS:
                save_method = cls.SUPPORTED_FORMATS[ext]
                getattr(cls, save_method)(image_data, path, *args, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        except Exception as e:
            logger.error(f"Failed to save image to {path}: {e}")
            raise

    @staticmethod
    def save_image(image: np.ndarray, path: Path) -> None:
        """Save image to file."""
        try:
            io.imsave(path, image)
            logger.info(f"Saved image to {path}")
        except Exception as e:
            logger.error(f"Failed to save image to {path}: {e}")
            raise

    @staticmethod
    def save_dream3d(
        image_data: Dict[str, np.ndarray], path: Path, header: str
    ) -> None:
        """Save image data to DREAM3D format."""
        ### TODO: Implement saving to DREAM3D format
        raise NotImplementedError("Saving to DREAM3D format is not yet implemented.")

    @staticmethod
    def save_ang(
        image_data: Dict[str, np.ndarray],
        path: Path,
        ang_header: str,
        resolution: float,
    ) -> None:
        """Save image data to ANG format."""
        ### TODO: Implement saving to ANG format
        # Create new x/y grid
        k = next(iter(image_data.keys()))
        x, y = np.meshgrid(
            np.arange(image_data[k].shape[1]), np.arange(image_data[k].shape[0])
        )
        x = x.astype(np.float32) * resolution
        y = y.astype(np.float32) * resolution
        image_data["x" if "x" in image_data else "X"] = x
        image_data["y" if "y" in image_data else "Y"] = y

        col_names = list(image_data.keys())
        data_out = []
        for i, key in enumerate(col_names):
            data_out.append(image_data[key].reshape(-1, 1).astype(np.float32))
        data_out = np.hstack(data_out)
        np.savetxt(
            path,
            data_out,
            header=ang_header[:-1],
            fmt="%.5f",
            delimiter=" ",
            comments="",
        )

    @staticmethod
    def save_h5(
        image_data: Dict[str, np.ndarray], path: Path, resolution: float
    ) -> None:
        """Save image data to HDF5 format."""
        ### TODO: Implement saving to HDF5 format
        raise NotImplementedError("Saving to HDF5 format is not yet implemented.")
        h5 = h5py.File(SAVE_PATH_EBSD, "w")
        h5.attrs.create(name="resolution", data=res)
        h5.attrs.create(name="header", data=header_string)
        for i, key in enumerate(ebsd_keys):
            print("Saving key:", key, src_imgs[i].shape, src_imgs[i].dtype)
            h5.create_dataset(name=key, data=src_imgs[i], dtype=src_imgs[i].dtype)
            h5[key].attrs.create(name="name", data=key)
            h5[key].attrs.create(name="dtype", data=str(src_imgs[i].dtype))
            h5[key].attrs.create(name="shape", data=src_imgs[i].shape)
        for i, key in enumerate(bse_keys):
            h5.create_dataset(name=f"{key}", data=dst_imgs[i], dtype=dst_imgs[i].dtype)
            h5[f"{key}"].attrs.create(name="name", data=f"{key}")
            h5[f"{key}"].attrs.create(name="dtype", data=str(dst_imgs[i].dtype))
            h5[f"{key}"].attrs.create(name="shape", data=dst_imgs[i].shape)
        h5.close()


class TransformManager:
    """Manages image transformations and warping operations."""

    def __init__(self):
        self.supported_transforms = list(TransformType)
        self._last_transform = None
        self._last_params = None
        logger.info("TransformManager initialized")

    def estimate_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        transform_type: TransformType,
        **kwargs,
    ) -> Any:
        """Estimate transformation parameters from point correspondences."""
        if src_points.size == 0 or dst_points.size == 0:
            raise ValueError("Cannot estimate transform with empty point sets")

        if src_points.shape[0] != dst_points.shape[0]:
            raise ValueError(
                f"Point count mismatch: {src_points.shape[0]} vs {dst_points.shape[0]}"
            )

        try:
            if transform_type in [TransformType.TPS, TransformType.TPS_AFFINE]:
                # Import TPS here to avoid circular dependency
                from tps import ThinPlateSplineTransform

                affine_only = transform_type == TransformType.TPS_AFFINE
                tform = ThinPlateSplineTransform(affine_only=affine_only)
                tform.estimate(src_points, dst_points, **kwargs)
            else:
                # Use skimage for other transforms
                tform = transform.estimate_transform(
                    transform_type.value, src_points, dst_points, **kwargs
                )

            self._last_transform = tform
            logger.debug(f"Estimated {transform_type.value} transform")
            return tform

        except Exception as e:
            logger.error(f"Failed to estimate transform: {e}")
            raise

    def apply_transform(
        self,
        image: np.ndarray,
        tform: Any,
        output_shape: Optional[Tuple[int, int]] = None,
        order: int = 0,
    ) -> np.ndarray:
        """Apply transformation to an image."""
        try:
            if output_shape is None:
                output_shape = image.shape[:2]

            warped = transform.warp(
                image,
                tform,
                output_shape=output_shape,
                mode="constant",
                cval=0,
                order=order,
            )

            logger.debug(f"Applied transform to image of shape {image.shape}")
            return warped

        except Exception as e:
            logger.error(f"Failed to apply transform: {e}")
            raise

    def apply_transform_stack(
        self,
        image_stack: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        transform_type: TransformType,
        output_shape: Optional[Tuple[int, int]] = None,
        order: int = 0,
        **kwargs,
    ) -> np.ndarray:
        """Apply transformation to a stack of images with interpolation between slices."""
        if output_shape is None:
            output_shape = image_stack.shape[1:3]

        # Get unique slices with points
        slice_indices = np.unique(src_points[:, 0]).astype(int)

        # Estimate transforms for each slice with points
        transforms = {}
        for slice_idx in slice_indices:
            mask = src_points[:, 0] == slice_idx
            src_pts = src_points[mask, 1:]
            dst_pts = dst_points[mask, 1:]

            tform = self.estimate_transform(src_pts, dst_pts, transform_type, **kwargs)
            transforms[slice_idx] = tform

        # Apply transforms with interpolation
        output_stack = np.zeros(
            (image_stack.shape[0], *output_shape, image_stack.shape[-1])
        )

        for i in range(image_stack.shape[0]):
            # Find nearest transform or interpolate
            if i in transforms:
                tform = transforms[i]
            else:
                # Use nearest neighbor for now (could implement interpolation)
                nearest_idx = min(transforms.keys(), key=lambda x: abs(x - i))
                tform = transforms[nearest_idx]

            output_stack[i] = self.apply_transform(
                image_stack[i], tform, output_shape, order
            )

        logger.info(f"Applied transform to stack of {image_stack.shape[0]} images")
        return output_stack

    def export_transform(self, tform: Any, path: Path, format: str = "npy") -> None:
        """Export transformation parameters to file."""
        try:
            if hasattr(tform, "params"):
                params = tform.params
            else:
                params = np.array(tform)

            if format == "npy":
                np.save(path, params)
            elif format in ["csv", "txt"]:
                delimiter = "," if format == "csv" else " "
                np.savetxt(path, params, delimiter=delimiter)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Exported transform to {path}")

        except Exception as e:
            logger.error(f"Failed to export transform: {e}")
            raise


class ImageProcessor:
    """Handles image processing operations."""

    @staticmethod
    def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 20.0,
        kernel_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        try:
            tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            tensor = equalize_clahe(tensor, clip_limit, kernel_size)
            tensor = torch.round(
                255 * (tensor - tensor.min()) / (tensor.max() - tensor.min())
            )
            result = np.squeeze(tensor.detach().numpy().astype(np.uint8))
            result = result.reshape(image.shape)

            logger.debug(f"Applied CLAHE with clip_limit={clip_limit}")
            return result

        except Exception as e:
            logger.error(f"Failed to apply CLAHE: {e}")
            raise

    @staticmethod
    def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
        """Resize image by scale factor."""
        if scale == 1.0:
            return image

        try:
            if image.ndim == 3:
                tensor = (
                    torch.tensor(np.transpose(image, (2, 0, 1))).unsqueeze(0).float()
                )
            else:
                tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()

            new_size = (int(image.shape[0] * scale), int(image.shape[1] * scale))

            resized = RESIZE(tensor, new_size, InterpolationMode.NEAREST)
            resized = resized.detach().squeeze().numpy()

            if image.ndim == 3 and resized.ndim == 3:
                resized = np.transpose(resized, (1, 2, 0))

            logger.debug(f"Resized image by factor {scale}")
            return resized

        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            raise

    @staticmethod
    def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """Normalize image to uint8 range."""
        if image.dtype == np.uint8:
            return image

        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                if image.ndim == 3:
                    normalized = np.around(
                        255
                        * (image - image.min(axis=(0, 1)))
                        / (image.max(axis=(0, 1)) - image.min(axis=(0, 1))),
                        0,
                    ).astype(np.uint8)
                else:
                    normalized = np.around(
                        255 * (image - image.min()) / (image.max() - image.min()), 0
                    ).astype(np.uint8)

            return normalized

        except Exception as e:
            logger.error(f"Failed to normalize image: {e}")
            raise


class ProjectManager:
    """Manages project state and serialization."""

    def __init__(self):
        self.project_path: Optional[Path] = None
        self.is_modified = False
        self.metadata = {"version": "2.0", "created": None, "modified": None}
        logger.info("ProjectManager initialized")

    def mark_modified(self) -> None:
        """Mark the project as modified."""
        self.is_modified = True

    def reset(self) -> None:
        """Reset project manager to initial state."""
        self.project_path = None
        self.is_modified = False
        self.metadata = {"version": "2.0", "created": None, "modified": None}

    def save_project(
        self,
        path: Path,
        point_manager: PointManager,
        src_image_path: Path,
        dst_image_path: Path,
        settings: Dict[str, Any],
    ) -> None:
        """Save complete project state."""
        try:
            project_data = {
                "metadata": self.metadata,
                "source_image": str(src_image_path),
                "destination_image": str(dst_image_path),
                "source_points": point_manager.source_points.to_dict(),
                "destination_points": point_manager.destination_points.to_dict(),
                "settings": settings,
            }

            with open(path, "w") as f:
                json.dump(project_data, f, indent=2)

            self.project_path = path
            self.is_modified = False
            logger.info(f"Saved project to {path}")

        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            raise

    def load_project(self, path: Path) -> Dict[str, Any]:
        """Load project state from file."""
        try:
            with open(path, "r") as f:
                project_data = json.load(f)

            self.project_path = path
            self.is_modified = False
            logger.info(f"Loaded project from {path}")

            return project_data

        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            raise
