"""
view_interface.py - View Interface for Distortion Correction Application

This module defines the interface that any view implementation must follow.
"""

### TODO
# Get export working for dream3d
# Get 3D data working

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, ttk, messagebox
from presenter import ApplicationPresenter, TransformType, CropMode, DataFormat

# Configure logging
logger = logging.getLogger(__name__)


class ViewInterface(ABC):
    """Abstract base class for view implementations."""

    @abstractmethod
    def on_data_loaded(self) -> None:
        """Called when image data has been loaded."""
        pass

    @abstractmethod
    def on_points_changed(self) -> None:
        """Called when control points have changed."""
        pass

    @abstractmethod
    def on_display_update_needed(self) -> None:
        """Called when display needs to be updated."""
        pass

    @abstractmethod
    def on_error(self, message: str) -> None:
        """Called when an error occurs."""
        pass

    @abstractmethod
    def on_show_preview(self, warped: np.ndarray, reference: np.ndarray) -> None:
        """Called to show transformation preview."""
        pass

    @abstractmethod
    def on_project_loaded(self) -> None:
        """Called when a project has been loaded."""
        pass

    @abstractmethod
    def on_request_corresponding_point(self, target: str) -> None:
        """Called when a corresponding point is needed."""
        pass

    @abstractmethod
    def on_project_reset(self) -> None:
        """Called when a new project is created."""
        pass


class ModernDistortionCorrectionView(tk.Tk, ViewInterface):
    """Modern implementation of the distortion correction GUI using MVP pattern."""

    def __init__(self):
        super().__init__()

        # Create presenter
        self.presenter = ApplicationPresenter()
        self.presenter.set_view(self)

        # UI state
        self.current_src_zoom = 100  # percentage
        self.current_dst_zoom = 100  # percentage
        self.show_points = True
        self.awaiting_corresponding_point = None

        # Setup UI
        self._setup_window()
        self._setup_logging()
        self._create_menu()
        self._create_main_layout()
        self._create_controls()
        self._bind_events()

        logger.info("View initialized")

    def _style_call(self, style="dark"):
        if style == "dark":
            self.bg = "#333333"
            self.fg = "#ffffff"
            self.hl = "#229fff"
            self.hl2 = "#00bb00"
            self.tk.call("source", r"./theme/dark.tcl")
            s = ttk.Style(self)
            s.theme_use("azure-dark")
        elif style == "light":
            self.bg = "#ffffff"
            self.fg = "#000000"
            self.hl = "#007fff"
            self.hl2 = "#00bb00"
            self.tk.call("source", r"./theme/light.tcl")
            s = ttk.Style(self)
            s.theme_use("azure-light")

        s.configure("TFrame", background=self.bg)
        s.configure("TLabel", background=self.bg, foreground=self.fg)
        s.configure("TCheckbutton", background=self.bg, foreground=self.fg)
        s.configure(
            "TLabelframe",
            background=self.bg,
            foreground=self.fg,
            highlightcolor=self.hl,
            highlightbackground=self.hl,
        )
        s.configure(
            "TLabelframe.Label",
            background=self.bg,
            foreground=self.fg,
            highlightcolor=self.hl,
            highlightbackground=self.hl,
        )

    def _setup_window(self):
        """Setup main window properties."""
        self.title("Distortion Correction v2.0")
        self.geometry("1400x900")

        # Configure grid weights
        self.grid_rowconfigure(0, weight=0)  # Menu area
        self.grid_rowconfigure(1, weight=1)  # Main content
        self.grid_rowconfigure(2, weight=0)  # Status bar
        self.grid_columnconfigure(0, weight=1)

        # Set style
        self._style_call("dark")
        self.configure(background=self.bg)

    def _setup_logging(self):
        """Setup logging display."""
        # Create status bar for logging
        self.status_frame = ttk.Frame(self)
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=2)

        self.status_label = ttk.Label(self.status_frame, text="Ready", anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)

        self.progress_bar = ttk.Progressbar(
            self.status_frame,
            style="Niklas.Horizontal.TProgressbar",
            mode="indeterminate",
            length=200,
        )
        self.progress_bar.pack(side="right", padx=5)

    def _create_menu(self):
        """Create application menu."""
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)

        # File menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self._on_new_project)
        file_menu.add_separator()
        file_menu.add_command(
            label="Open Source Image...", command=self._on_open_source
        )
        file_menu.add_command(
            label="Open Destination Image...", command=self._on_open_destination
        )
        file_menu.add_separator()
        file_menu.add_command(label="Load Points...", command=self._on_load_points)
        file_menu.add_command(label="Save Points", command=self._on_save_points)
        file_menu.add_separator()
        file_menu.add_command(label="Open Project...", command=self._on_open_project)
        file_menu.add_command(label="Save Project", command=self._on_save_project)
        file_menu.add_command(
            label="Save Project As...", command=self._on_save_project_as
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Export Transform...", command=self._on_export_transform
        )
        file_menu.add_command(
            label="Export Corrected Image...", command=self._on_export_corrected
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Edit menu
        edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self._on_undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self._on_redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(
            label="Clear Points (Current Slice)", command=self._on_clear_slice
        )
        edit_menu.add_command(label="Clear All Points", command=self._on_clear_all)
        edit_menu.add_separator()
        edit_menu.add_command(label="Set Resolution", command=self._on_set_resolution)

        # View menu
        view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(
            label="Hide Points",
            variable=tk.BooleanVar(value=False),
            command=self._on_toggle_points,
        )
        view_menu.add_separator()
        view_menu.add_command(
            label="Zoom In", command=self._on_zoom_in, accelerator="Ctrl++"
        )
        view_menu.add_command(
            label="Zoom Out", command=self._on_zoom_out, accelerator="Ctrl+-"
        )
        view_menu.add_command(
            label="Zoom 100%", command=self._on_zoom_reset, accelerator="Ctrl+0"
        )

        # Transform menu
        transform_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Transform", menu=transform_menu)
        transform_menu.add_command(
            label="Preview TPS",
            command=lambda: self._on_preview_transform(TransformType.TPS),
        )
        transform_menu.add_command(
            label="Preview Affine TPS",
            command=lambda: self._on_preview_transform(TransformType.TPS_AFFINE),
        )
        transform_menu.add_separator()
        transform_menu.add_command(
            label="Apply to Current Slice", command=self._on_apply_current
        )
        transform_menu.add_command(
            label="Apply to All Slices", command=self._on_apply_all
        )
        transform_menu.add_separator()
        transform_menu.add_command(label="Test", command=self._on_progress_test)

    def _create_main_layout(self):
        """Create main layout with image viewers."""
        # Main container
        self.main_frame = ttk.Frame(self)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=0)
        self.main_frame.grid_columnconfigure(2, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=0)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=0)

        # Left viewer (source/distorted)
        self.left_frame = ttk.LabelFrame(self.main_frame, text="Source (Distorted)")
        self.left_frame.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=2)

        self.left_canvas = tk.Canvas(self.left_frame, bg=self.bg, cursor="crosshair")
        left_h_scrollbar = ttk.Scrollbar(
            self.left_frame,
            orient=tk.HORIZONTAL,
            command=self.left_canvas.xview,
            cursor="sb_h_double_arrow",
        )
        left_v_scrollbar = ttk.Scrollbar(
            self.left_frame,
            orient=tk.VERTICAL,
            command=self.left_canvas.yview,
            cursor="sb_v_double_arrow",
        )
        self.left_canvas.config(
            xscrollcommand=left_h_scrollbar.set, yscrollcommand=left_v_scrollbar.set
        )
        left_h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        left_v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.pack(fill="both", expand=True)

        # Right viewer (destination/control)
        self.right_frame = ttk.LabelFrame(self.main_frame, text="Destination (Control)")
        self.right_frame.grid(row=0, column=2, rowspan=3, sticky="nsew", padx=2)

        self.right_canvas = tk.Canvas(self.right_frame, bg=self.bg, cursor="crosshair")
        right_h_scrollbar = ttk.Scrollbar(
            self.right_frame,
            orient=tk.HORIZONTAL,
            command=self.right_canvas.xview,
            cursor="sb_h_double_arrow",
        )
        right_v_scrollbar = ttk.Scrollbar(
            self.right_frame,
            orient=tk.VERTICAL,
            command=self.right_canvas.yview,
            cursor="sb_v_double_arrow",
        )
        self.right_canvas.config(
            xscrollcommand=right_h_scrollbar.set, yscrollcommand=right_v_scrollbar.set
        )
        right_h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        right_v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_canvas.pack(fill="both", expand=True)

    def _create_controls(self):
        """Create control panel."""
        # Top controls
        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Slice selector
        ttk.Label(controls_frame, text="Slice:").pack(side="left", padx=5)
        self.slice_var = tk.IntVar(value=0)
        self.slice_spinbox = ttk.Spinbox(
            controls_frame,
            from_=0,
            to=0,
            width=7,
            textvariable=self.slice_var,
            state="disabled",
            command=self._on_slice_changed,
        )
        self.slice_spinbox.pack(side="left", padx=5)

        # Mode selectors
        ttk.Label(controls_frame, text="Mode (src):").pack(side="left", padx=(20, 5))
        self.source_mode_var = tk.StringVar(value="Intensity")
        self.source_mode_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.source_mode_var,
            state="readonly",
            width=12,
        )
        self.source_mode_combo.pack(side="left", padx=5)
        self.source_mode_combo.bind(
            "<<ComboboxSelected>>", self._on_source_mode_changed
        )

        ttk.Label(controls_frame, text="Mode (dst):").pack(side="left", padx=(20, 5))
        self.dest_mode_var = tk.StringVar(value="Intensity")
        self.dest_mode_combo = ttk.Combobox(
            controls_frame, textvariable=self.dest_mode_var, state="readonly", width=12
        )
        self.dest_mode_combo.pack(side="left", padx=5)
        self.dest_mode_combo.bind("<<ComboboxSelected>>", self._on_dest_mode_changed)

        # CLAHE toggles
        self.clahe_source_var = tk.BooleanVar(value=False)
        self.clahe_source_check = ttk.Checkbutton(
            controls_frame,
            text="CLAHE (src)",
            variable=self.clahe_source_var,
            command=lambda: self.presenter.toggle_clahe("source"),
        )
        self.clahe_source_check.pack(side="left", padx=(20, 5))

        self.clahe_dest_var = tk.BooleanVar(value=False)
        self.clahe_dest_check = ttk.Checkbutton(
            controls_frame,
            text="CLAHE (dst)",
            variable=self.clahe_dest_var,
            command=lambda: self.presenter.toggle_clahe("destination"),
        )
        self.clahe_dest_check.pack(side="left", padx=5)

        # Zoom control
        ttk.Label(controls_frame, text="Zoom (src):").pack(side="left", padx=(20, 5))
        self.zoom_src_var = tk.StringVar(value="100%")
        self.zoom_src_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.zoom_src_var,
            values=[
                "5%",
                "10%",
                "25%",
                "50%",
                "75%",
                "100%",
                "150%",
                "200%",
                "300%",
                "500%",
            ],
            state="readonly",
            width=8,
        )
        self.zoom_src_combo.pack(side="left", padx=5)
        self.zoom_src_combo.bind("<<ComboboxSelected>>", self._on_zoom_changed)
        ttk.Label(controls_frame, text="Zoom (dst):").pack(side="left", padx=(20, 5))
        self.zoom_dst_var = tk.StringVar(value="100%")
        self.zoom_dst_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.zoom_dst_var,
            values=[
                "5%",
                "10%",
                "25%",
                "50%",
                "75%",
                "100%",
                "150%",
                "200%",
                "300%",
                "500%",
            ],
            state="readonly",
            width=8,
        )
        self.zoom_dst_combo.pack(side="left", padx=5)
        self.zoom_dst_combo.bind("<<ComboboxSelected>>", self._on_zoom_changed)

        # Match resolutions control
        self.match_resolutions_var = tk.BooleanVar(value=False)
        self.match_resolutions_check = ttk.Checkbutton(
            controls_frame,
            text="Match Res",
            variable=self.match_resolutions_var,
            command=self.presenter.toggle_match_resolutions,
        )
        self.match_resolutions_check.pack(side="left", padx=(20, 5))

    def _bind_events(self):
        """Bind keyboard and mouse events."""
        # Keyboard shortcuts
        self.bind("<Control-z>", lambda e: self._on_undo())
        self.bind("<Control-y>", lambda e: self._on_redo())
        self.bind("<Control-equal>", lambda e: self._on_zoom_in())
        self.bind("<Control-minus>", lambda e: self._on_zoom_out())
        self.bind("<Control-0>", lambda e: self._on_zoom_reset())

        # Mouse events for canvases
        self.left_canvas.bind(
            "<Button-1>", lambda e: self._on_canvas_click(e, "source")
        )
        self.right_canvas.bind(
            "<Button-1>", lambda e: self._on_canvas_click(e, "destination")
        )
        if os.name == "posix":
            remove_string = "<Button 2>"
            scroll_multiplier = 1
        else:
            remove_string = "<Button 3>"
            scroll_multiplier = 120

        self.left_canvas.bind(
            remove_string, lambda e: self._on_canvas_right_click(e, "source")
        )
        self.right_canvas.bind(
            remove_string, lambda e: self._on_canvas_right_click(e, "destination")
        )
        self.left_canvas.bind(
            "<MouseWheel>",
            lambda event: self.left_canvas.yview_scroll(
                int(-1 * (event.delta / scroll_multiplier)), "units"
            ),
        )
        self.left_canvas.bind(
            "<Shift-MouseWheel>",
            lambda event: self.left_canvas.xview_scroll(
                int(-1 * (event.delta / scroll_multiplier)), "units"
            ),
        )
        self.right_canvas.bind(
            "<MouseWheel>",
            lambda event: self.right_canvas.yview_scroll(
                int(-1 * (event.delta / scroll_multiplier)), "units"
            ),
        )
        self.right_canvas.bind(
            "<Shift-MouseWheel>",
            lambda event: self.right_canvas.xview_scroll(
                int(-1 * (event.delta / scroll_multiplier)), "units"
            ),
        )

    # ========== Event Handlers ==========

    def _on_progress_test(self):
        """Test progress bar."""
        self.show_progress(True)
        self.after(2000, lambda: self.show_progress(False))

    def _on_new_project(self):
        """Handle creating a new project."""
        if self.presenter.has_unsaved_changes():
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save the current project?",
            )
            if response is None:
                return
            elif response:
                self._on_save_project()

        self.presenter.new_project()
        self.set_status("New project created")

    def _on_open_source(self):
        """Handle opening source image."""
        file_paths = filedialog.askopenfilenames(
            title="Open Source Image",
            filetypes=[
                ("All Supported", "*.ang *.h5 *.dream3d *.tif *.tiff *.png *.jpg"),
                ("EBSD Files", "*.ang *.h5 *.dream3d"),
                ("Image Files", "*.tif *.tiff *.png *.jpg"),
                ("All Files", "*.*"),
            ],
        )

        if file_paths:
            try:
                self.show_progress(True)
                if len(file_paths) == 1:
                    path = Path(file_paths[0])
                    modality_name = None

                    # Check if this is a single image file that needs a modality name
                    if path.suffix.lower() in [
                        ".tif",
                        ".tiff",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        # Ask for modality name
                        modality_name = self._get_modality_name_dialog(path.name)
                        if modality_name is None:
                            return  # User cancelled

                    self.show_progress(True)
                    self.set_status(f"Loading source image: {path.name}")
                    if self.presenter.load_source_image(
                        path, modality_name=modality_name
                    ):
                        self.set_status("Source image loaded successfully")

                else:
                    first_path = Path(file_paths[0])
                    # Check if this is a single image file that needs a modality name
                    if first_path.suffix.lower() in [
                        ".tif",
                        ".tiff",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        # Ask for modality name
                        modality_name = self._get_modality_name_dialog(first_path.name)
                        if modality_name is None:
                            return  # User cancelled

                    self.set_status(f"Loading {len(file_paths)} source images")
                    if self.presenter.load_source_image(
                        file_paths, modality_name=modality_name
                    ):
                        self.set_status("Source image stack loaded successfully")
            finally:
                self.show_progress(False)

    def _on_open_destination(self):
        """Handle opening destination image."""
        file_paths = filedialog.askopenfilenames(
            title="Open Destination Image",
            filetypes=[
                ("Image Files", "*.tif *.tiff *.png *.jpg *.dream3d"),
                ("All Files", "*.*"),
            ],
        )

        if file_paths:
            try:
                self.show_progress(True)
                if len(file_paths) == 1:
                    path = Path(file_paths[0])
                    modality_name = None

                    # Check if this is a single image file that needs a modality name
                    if path.suffix.lower() in [
                        ".tif",
                        ".tiff",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        # Ask for modality name
                        modality_name = self._get_modality_name_dialog(path.name)
                        if modality_name is None:
                            return  # User cancelled

                    self.set_status(f"Loading destination image: {path.name}")
                    if self.presenter.load_destination_image(
                        path, modality_name=modality_name
                    ):
                        self.set_status("Destination image loaded successfully")

                else:
                    first_path = Path(file_paths[0])
                    # Check if this is a single image file that needs a modality name
                    if first_path.suffix.lower() in [
                        ".tif",
                        ".tiff",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        # Ask for modality name
                        modality_name = self._get_modality_name_dialog(first_path.name)
                        if modality_name is None:
                            return  # User cancelled

                    self.set_status(f"Loading {len(file_paths)} destination images")
                    if self.presenter.load_destination_image(
                        file_paths, modality_name=modality_name
                    ):
                        self.set_status("Destination image stack loaded successfully")
            finally:
                self.show_progress(False)

    def _on_load_points(self):
        """Handle loading control points."""
        src_path = filedialog.askopenfilename(
            title="Load Source Points",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )

        if src_path:
            dst_path = filedialog.askopenfilename(
                title="Load Destination Points",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            )

            if dst_path:
                if self.presenter.load_points(Path(src_path), Path(dst_path)):
                    self.set_status("Points loaded successfully")

    def _on_save_points(self):
        """Handle saving control points."""
        # Uses the default paths set in presenter
        self.presenter._save_points()
        self.set_status("Points saved")

    def _on_open_project(self):
        """Handle opening a project."""
        file_path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("Project Files", "*.json"), ("All Files", "*.*")],
        )

        if file_path:
            self.show_progress(True)
            self.set_status("Loading project...")

            if self.presenter.load_project(Path(file_path)):
                self.set_status("Project loaded successfully")

            self.show_progress(False)

    def _on_save_project(self):
        """Handle saving current project."""
        if self.presenter.project_manager.project_path:
            if self.presenter.save_project(self.presenter.project_manager.project_path):
                self.set_status("Project saved")
        else:
            self._on_save_project_as()

    def _on_save_project_as(self):
        """Handle saving project with new name."""
        file_path = filedialog.asksaveasfilename(
            title="Save Project As",
            defaultextension=".json",
            filetypes=[("Project Files", "*.json"), ("All Files", "*.*")],
        )

        if file_path:
            if self.presenter.save_project(Path(file_path)):
                self.set_status(f"Project saved as {Path(file_path).name}")

    def _on_export_transform(self):
        """Handle exporting transformation."""
        # Get transform type
        transform_type = self._get_transform_type_dialog()
        if not transform_type:
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Transform",
            defaultextension=".npy",
            filetypes=[
                ("NumPy Array", "*.npy"),
                ("CSV File", "*.csv"),
                ("Text File", "*.txt"),
                ("All Files", "*.*"),
            ],
        )

        if file_path:
            if self.presenter.export_transform(Path(file_path), transform_type):
                self.set_status("Transform exported successfully")

    def _on_export_corrected(self):
        """Handle exporting corrected image."""
        # This would need implementation for full export functionality
        ### TODO: Get multiple output formats working
        self.set_status("Exporting corrected image")

        transform_type = self._get_transform_type_dialog()
        if transform_type is None:
            return

        data_format = self._get_export_format_dialog()
        if data_format is None:
            return

        if data_format == DataFormat.DREAM3D:
            crop_mode = CropMode.SOURCE
        else:
            crop_mode = self._get_crop_mode_dialog()
            if crop_mode is None:
                return

        if data_format == DataFormat.IMAGE:
            ftypes = [
                ("TIFF Image", "*.tif *.tiff"),
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("All Files", "*.*"),
            ]
        elif data_format == DataFormat.RAW_IMAGE:
            ftypes = [
                ("TIFF Image", "*.tif *.tiff"),
                ("All Files", "*.*"),
            ]
        elif data_format == DataFormat.ANG:
            ftypes = [
                ("ANG File", "*.ang"),
                ("All Files", "*.*"),
            ]
        elif data_format == DataFormat.DREAM3D:
            ftypes = [
                ("Dream3D File", "*.dream3d"),
                ("All Files", "*.*"),
            ]

        path = filedialog.asksaveasfilename(
            title="Export Corrected Data",
            defaultextension=ftypes[0][1].split(" ")[0].replace("*", ""),
            filetypes=ftypes,
        )

        if not path:
            return

        self.show_progress(True)
        self.set_status(
            f"Exporting {transform_type.value} corrected image as {data_format.value} cropped to {crop_mode.value}..."
        )
        self.presenter.export_data(Path(path), data_format, crop_mode, transform_type)
        self.show_progress(False)
        self.set_status(f"Data exported to '{path}'")

    def _on_canvas_click(self, event, canvas_type):
        """Handle canvas click for point placement."""
        # Get scrollbar offsets
        if canvas_type == "source":
            scale = self.current_src_zoom / 100.0
            canvas = self.left_canvas
        else:
            scale = self.current_dst_zoom / 100.0
            canvas = self.right_canvas

        # Convert canvas coordinates to image coordinates
        x = int(event.x / scale)
        y = int(event.y / scale)
        x += int(canvas.canvasx(0) / scale)
        y += int(canvas.canvasy(0) / scale)

        if self.awaiting_corresponding_point == canvas_type:
            # Add corresponding point
            self.presenter.add_point(canvas_type, x, y)
            self.awaiting_corresponding_point = None
            self.set_status("Point pair added")
        else:
            # Start new point pair
            self.presenter.add_point(canvas_type, x, y)
            self.awaiting_corresponding_point = (
                "destination" if canvas_type == "source" else None
            )
            if self.awaiting_corresponding_point:
                self.set_status("Click on destination image to add corresponding point")

    def _on_canvas_right_click(self, event, canvas_type):
        """Handle right-click for point removal."""
        # Find nearest point and remove it

        if canvas_type == "source":
            canvas = self.left_canvas
        else:
            canvas = self.right_canvas

        closest = canvas.find_closest(canvas.canvasx(event.x), canvas.canvasy(event.y))
        tag = canvas.itemcget(closest[0], "tags")
        tag = (
            tag.replace("current", "")
            .replace("text", "")
            .replace("bbox", "")
            .replace("point_", "")
            .strip()
        )
        if tag == "":
            return
        logger.debug(f"Removing point with index {int(tag)}")
        self.presenter.remove_point(int(tag))

    def _on_slice_changed(self):
        """Handle slice change."""
        self.presenter.set_current_slice(self.slice_var.get())

    def _on_source_mode_changed(self, event=None):
        """Handle source mode change."""
        self.presenter.set_source_mode(self.source_mode_var.get())

    def _on_dest_mode_changed(self, event=None):
        """Handle destination mode change."""
        self.presenter.set_destination_mode(self.dest_mode_var.get())

    def _on_zoom_changed(self, event=None):
        """Handle zoom change."""
        zoom_str_src = self.zoom_src_var.get().rstrip("%")
        self.current_src_zoom = int(zoom_str_src)
        zoom_str_dst = self.zoom_dst_var.get().rstrip("%")
        self.current_dst_zoom = int(zoom_str_dst)
        self.update_display()

    def _on_zoom_in(self):
        """Zoom in."""
        zoom_levels = [5, 10, 25, 50, 75, 100, 150, 200, 300, 500]
        current_src_idx = (
            zoom_levels.index(self.current_src_zoom)
            if self.current_src_zoom in zoom_levels
            else 3
        )
        current_dst_idx = (
            zoom_levels.index(self.current_dst_zoom)
            if self.current_dst_zoom in zoom_levels
            else 3
        )
        if current_src_idx < len(zoom_levels) - 1:
            self.current_src_zoom = zoom_levels[current_src_idx + 1]
            self.zoom_src_var.set(f"{self.current_src_zoom}%")
            self.update_display()
        if current_dst_idx < len(zoom_levels) - 1:
            self.current_dst_zoom = zoom_levels[current_dst_idx + 1]
            self.zoom_dst_var.set(f"{self.current_dst_zoom}%")
            self.update_display()

    def _on_zoom_out(self):
        """Zoom out."""
        zoom_levels = [5, 10, 25, 50, 75, 100, 150, 200, 300, 500]
        current_src_idx = (
            zoom_levels.index(self.current_src_zoom)
            if self.current_src_zoom in zoom_levels
            else 3
        )
        current_dst_idx = (
            zoom_levels.index(self.current_dst_zoom)
            if self.current_dst_zoom in zoom_levels
            else 3
        )
        if current_src_idx > 0:
            self.current_src_zoom = zoom_levels[current_src_idx - 1]
            self.zoom_src_var.set(f"{self.current_src_zoom}%")
            self.update_display()
        if current_dst_idx > 0:
            self.current_dst_zoom = zoom_levels[current_dst_idx - 1]
            self.zoom_dst_var.set(f"{self.current_dst_zoom}%")
            self.update_display()

    def _on_zoom_reset(self):
        """Reset zoom to 100%."""
        self.current_src_zoom = 100
        self.current_dst_zoom = 100
        self.zoom_src_var.set("100%")
        self.zoom_dst_var.set("100%")
        self.update_display()

    def _on_undo(self):
        """Undo last action."""
        self.presenter.undo()
        self.set_status("Undone")

    def _on_redo(self):
        """Redo last undone action."""
        self.presenter.redo()
        self.set_status("Redone")

    def _on_clear_slice(self):
        """Clear points on current slice."""
        if messagebox.askyesno("Clear Points", "Clear all points on current slice?"):
            self.presenter.clear_points(slice_only=True)
            self.set_status("Points cleared for current slice")

    def _on_clear_all(self):
        """Clear all points."""
        if messagebox.askyesno("Clear All Points", "Clear all points on all slices?"):
            self.presenter.clear_points(slice_only=False)
            self.set_status("All points cleared")

    def _on_toggle_points(self):
        """Toggle point visibility."""
        self.show_points = not self.show_points
        self.update_display()

    def _on_set_resolution(self):
        """Set image resolution."""
        ### TODO: Fix this
        src_res, dst_res = self._get_image_resolutions_dialog()
        if src_res is None or dst_res is None:
            return
        elif src_res and dst_res:
            # Update image resolutions
            self.presenter.set_image_resolutions(src_res, dst_res)

    def _on_preview_transform(self, transform_type):
        """Preview transformation."""
        crop_mode = self._get_crop_mode_dialog()
        if crop_mode is not None:
            self.show_progress(True)
            self.set_status(f"Generating {transform_type.value} preview...")
            self.presenter.apply_transform(transform_type, crop_mode, preview=True)
            self.show_progress(False)

    def _on_apply_current(self):
        """Apply transformation to current slice."""
        transform_type = self._get_transform_type_dialog()
        if transform_type:
            crop_mode = self._get_crop_mode_dialog()
            if crop_mode is not None:
                self.show_progress(True)
                self.set_status("Applying transformation...")
                self.presenter.apply_transform(transform_type, crop_mode, preview=False)
                self.show_progress(False)
                self.set_status("Transformation applied")

    def _on_apply_all(self):
        """Apply transformation to all slices."""
        if not self.presenter.is_3d_mode:
            messagebox.showinfo("Info", "This operation is only available for 3D data")
            return

        transform_type = self._get_transform_type_dialog()
        if transform_type:
            crop_mode = self._get_crop_mode_dialog()
            if crop_mode is not None:
                self.show_progress(True)
                self.set_status("Applying transformation to all slices...")
                self.presenter.apply_transform_3d(transform_type, crop_mode)
                self.show_progress(False)
                self.set_status("Transformation applied to all slices")

    # ========== ViewInterface Implementation ==========

    def on_data_loaded(self):
        """Called when image data has been loaded."""
        # Update UI elements based on loaded data
        if self.presenter.source_image:
            modes = self.presenter.get_source_modalities()
            self.source_mode_combo["values"] = modes
            if modes:
                # Keep current mode if it still exists, otherwise use presenter's current mode
                current_mode = self.presenter.current_source_mode
                if current_mode in modes:
                    self.source_mode_var.set(current_mode)
                else:
                    # Use the first mode if current doesn't exist
                    self.source_mode_var.set(modes[0])
                    self.presenter.current_source_mode = modes[0]

        if self.presenter.destination_image:
            modes = self.presenter.get_destination_modalities()
            self.dest_mode_combo["values"] = modes
            if modes:
                # Keep current mode if it still exists, otherwise use presenter's current mode
                current_mode = self.presenter.current_dest_mode
                if current_mode in modes:
                    self.dest_mode_var.set(current_mode)
                else:
                    # Use the first mode if current doesn't exist
                    self.dest_mode_var.set(modes[0])
                    self.presenter.current_dest_mode = modes[0]

        # Update slice control
        min_slice, max_slice = self.presenter.get_slice_range()
        self.slice_spinbox.config(from_=min_slice, to=max_slice)
        self.slice_spinbox.config(state="normal" if max_slice > 0 else "disabled")

        # Update match resolutions checkbox
        self.match_resolutions_var.set(self.presenter.match_resolutions)

        self.update_display()

    def on_points_changed(self):
        """Called when control points have changed."""
        self.update_display()

    def on_display_update_needed(self):
        """Called when display needs to be updated."""
        self.update_display()

    def on_error(self, message: str):
        """Called when an error occurs."""
        messagebox.showerror("Error", message)
        self.set_status(f"Error: {message}")

    def on_show_preview(self, warped: np.ndarray, reference: np.ndarray):
        """Called to show transformation preview."""
        # Create preview window
        Viewer = Interactive2DViewer(self, warped, reference, "Transformation Preview")
        self.set_status("Preview window opened")
        Viewer.root.wait_window()
        self.set_status("Preview window closed")

    def on_project_loaded(self):
        """Called when a project has been loaded."""
        self.on_data_loaded()
        self.set_status("Project loaded")

    def on_request_corresponding_point(self, target: str):
        """Called when a corresponding point is needed."""
        self.awaiting_corresponding_point = target
        self.set_status(f"Click on {target} image to add corresponding point")

    def on_project_reset(self):
        """Called when a new project is created."""
        # Clear canvases
        self.left_canvas.delete("all")
        self.right_canvas.delete("all")

        # Reset UI state
        self.current_src_zoom = 100
        self.current_dst_zoom = 100
        self.zoom_src_var.set("100%")
        self.zoom_dst_var.set("100%")
        self.show_points = True
        self.awaiting_corresponding_point = None

        # Reset slice control
        self.slice_var.set(0)
        self.slice_spinbox.config(from_=0, to=0, state="disabled")

        # Reset mode selectors
        self.source_mode_var.set("Intensity")
        self.dest_mode_var.set("Intensity")
        self.source_mode_combo["values"] = []
        self.dest_mode_combo["values"] = []

        # Reset CLAHE toggles
        self.clahe_source_var.set(False)
        self.clahe_dest_var.set(False)

        # Reset match resolutions
        self.match_resolutions_var.set(False)

        self.set_status("Ready")

    # ========== Helper Methods ==========

    def update_display(self):
        """Update the image display."""
        # if not self.presenter.source_image or not self.presenter.destination_image:
        #     return

        # Get current images

        src_scale = self.current_src_zoom / 100.0
        dst_scale = self.current_dst_zoom / 100.0
        src_img, dst_img = self.presenter.get_current_images(
            src_scale=src_scale, dst_scale=dst_scale
        )

        if src_img is not None:
            # Clear canvases
            self.left_canvas.delete("all")

            # Convert to PhotoImage and display
            self._display_image(self.left_canvas, src_img)

            # Update scroll region
            self.left_canvas.config(
                scrollregion=(0, 0, src_img.shape[1], src_img.shape[0])
            )

        if dst_img is not None:
            # Clear canvases
            self.right_canvas.delete("all")

            # Convert to PhotoImage and display
            self._display_image(self.right_canvas, dst_img)

            # Update scroll region
            self.right_canvas.config(
                scrollregion=(0, 0, dst_img.shape[1], dst_img.shape[0])
            )

        # Draw points if enabled
        if self.show_points:
            self._draw_points()

    def _display_image(self, canvas, image):
        """Display image on canvas."""
        # This would need proper implementation to convert numpy array to PhotoImage
        # For now, just create a placeholder
        image = self._photo_image(image)
        canvas.image = image  # Keep a reference to avoid garbage collection
        canvas.create_image(0, 0, anchor="nw", image=image)

    def _photo_image(self, image: np.ndarray):
        """Creates a PhotoImage object that plays nicely with a tkinter canvas for viewing purposes."""
        height, width, channels = image.shape
        if channels == 1:
            data = (
                f"P5 {width} {height} 255 ".encode() + image.astype(np.uint8).tobytes()
            )
        else:
            ppm_header = f"P6 {width} {height} 255 ".encode()
            data = ppm_header + image.tobytes()
        return tk.PhotoImage(width=width, height=height, data=data, format="PPM")

    def _draw_points(self):
        """Draw control points on canvases."""
        # Scale points for current zoom
        src_points, dst_points = self.presenter.get_points()
        src_scale = self.current_src_zoom / 100.0
        dst_scale = self.current_dst_zoom / 100.0

        # Scale destination points if resolutions are matched
        if self.presenter.match_resolutions:
            src_res, dst_res = self.presenter.get_resolutions()
            res_scale = dst_res / src_res
            dst_points = [(p[0] * res_scale, p[1] * res_scale) for p in dst_points]

        # Draw source points
        for i, point in enumerate(src_points):
            x, y = point[0] * src_scale, point[1] * src_scale
            self.left_canvas.create_oval(
                x - 4,
                y - 4,
                x + 4,
                y + 4,
                fill="white",
                outline="red",
                tags=f"point_{i}",
            )
            self.left_canvas.create_text(
                x + 5,
                y - 5,
                text=str(i),
                fill="red",
                anchor="sw",
                tags=f"point_{i}",
                font=("Arial", 11, "bold"),
            )
            self.left_canvas.create_text(
                x + 5,
                y - 5,
                text=str(i),
                fill="white",
                anchor="sw",
                tags=f"point_{i}",
                font=("Arial", 10),
            )

        # Draw destination points
        for i, point in enumerate(dst_points):
            x, y = point[0] * dst_scale, point[1] * dst_scale
            self.right_canvas.create_oval(
                x - 4,
                y - 4,
                x + 4,
                y + 4,
                fill="white",
                outline="green",
                tags=f"point_{i}",
            )
            self.right_canvas.create_text(
                x + 5,
                y - 5,
                text=str(i),
                fill="green",
                anchor="sw",
                tags=f"point_{i}",
                font=("Arial", 11, "bold"),
            )
            self.right_canvas.create_text(
                x + 5,
                y - 5,
                text=str(i),
                fill="white",
                anchor="sw",
                tags=f"point_{i}",
                font=("Arial", 10),
            )

    def set_status(self, message: str):
        """Update status bar."""
        self.status_label.config(text=message)
        self.update_idletasks()
        logger.info(message)

    def show_progress(self, show: bool):
        """Show or hide progress indicator."""
        if show:
            self.progress_bar.start(1)
        else:
            self.progress_bar.stop()

    def _get_image_resolutions_dialog(self) -> Tuple:
        """Show dialog to select transformation type."""
        dialog = tk.Toplevel(self)
        dialog.title("Enter Resolution (µm)")
        dialog.geometry("250x100")
        dialog.transient(self)
        dialog.grab_set()
        dialog.rowconfigure([0, 1, 2], weight=1)
        dialog.columnconfigure([0, 1], weight=1)

        default_src_res, default_dst_res = self.presenter.get_resolutions()
        src_res = tk.StringVar(value=default_src_res)
        dst_res = tk.StringVar(value=default_dst_res)
        result = [default_src_res, default_dst_res]

        sl = ttk.Label(dialog, text="Source:")
        sl.grid(row=0, column=0, sticky="nse", padx=3, pady=3)
        se = ttk.Entry(dialog, textvariable=src_res, width=10)
        se.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)

        dl = ttk.Label(dialog, text="Destination:")
        dl.grid(row=1, column=0, sticky="nse", padx=3, pady=3)
        se = ttk.Entry(dialog, textvariable=dst_res, width=10)
        se.grid(row=1, column=1, sticky="nsew", padx=3, pady=3)

        def on_ok():
            result[0] = float(src_res.get())
            result[1] = float(dst_res.get())
            dialog.destroy()

        def on_cancel():
            dialog.destroy()
            result[0] = None
            result[1] = None

        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=2, column=0, columnspan=2, padx=3, pady=3)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        dialog.wait_window()
        return result[0], result[1]

    def _get_transform_type_dialog(self) -> Optional[TransformType]:
        """Show dialog to select transformation type."""
        dialog = tk.Toplevel(self)
        dialog.title("Select Transform Type")
        dialog.geometry("300x130")
        dialog.transient(self)
        dialog.grab_set()

        selected = tk.StringVar(value="TPS")

        for transform_type in TransformType:
            tk.Radiobutton(
                dialog,
                text=transform_type.value.replace("_", " ").title(),
                variable=selected,
                value=transform_type.value,
            ).pack(anchor="w", padx=20, pady=5)

        result = [None]

        def on_ok():
            result[0] = TransformType(selected.get())
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(side="bottom", pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        dialog.wait_window()
        return result[0]

    def _get_crop_mode_dialog(self) -> Optional[CropMode]:
        """Show dialog to select crop mode."""
        dialog = tk.Toplevel(self)
        dialog.title("Select Crop Mode")
        dialog.geometry("300x150")
        dialog.transient(self)
        dialog.grab_set()

        selected = tk.StringVar(value="none")

        for crop_mode in CropMode:
            tk.Radiobutton(
                dialog,
                text=crop_mode.value.replace("_", " ").title(),
                variable=selected,
                value=crop_mode.value,
            ).pack(anchor="w", padx=20, pady=5)

        result = [None]

        def on_ok():
            result[0] = CropMode(selected.get())
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(side="bottom", pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        dialog.wait_window()
        return result[0]

    def _get_export_format_dialog(self) -> Optional[DataFormat]:
        """Show dialog to select export format."""
        dialog = tk.Toplevel(self)
        dialog.title("Select Export Format")
        dialog.geometry("250x260")
        dialog.transient(self)

        selected_format = tk.StringVar(value=DataFormat.IMAGE.value)

        ttk.Label(dialog, text="Data Format:").pack(anchor="w", padx=20, pady=5)
        for data_format in DataFormat:
            tk.Radiobutton(
                dialog,
                text=data_format.value.replace("_", " ").title(),
                variable=selected_format,
                value=data_format.value,
            ).pack(anchor="w", padx=20, pady=5)

        result = [None]

        def on_ok():
            result[0] = DataFormat(selected_format.get())
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(side="bottom", pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        dialog.wait_window()
        return result[0]

    def _get_modality_name_dialog(self, filename: str) -> Optional[str]:
        """Show dialog to enter a modality name for an image."""
        dialog = tk.Toplevel(self)
        dialog.title(f"Loading {filename}")
        dialog.geometry("350x150")
        dialog.transient(self)
        dialog.grab_set()

        # Main frame
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Label
        ttk.Label(
            main_frame, text=f"Enter a name for this image modality:", wraplength=300
        ).pack(pady=(0, 10))

        # Entry field
        modality_var = tk.StringVar(value="")
        entry = ttk.Entry(main_frame, textvariable=modality_var, width=25)
        entry.pack(pady=5)
        entry.focus()

        result = [None]

        def on_ok():
            name = modality_var.get().strip()
            if name:
                result[0] = name
                dialog.destroy()
            else:
                messagebox.showwarning(
                    "Invalid Name", "Please enter a valid modality name."
                )

        def on_cancel():
            dialog.destroy()

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        # Bind Enter key to OK
        entry.bind("<Return>", lambda e: on_ok())
        dialog.bind("<Escape>", lambda e: on_cancel())

        dialog.wait_window()
        return result[0]


class Interactive2DViewer:
    """Tkinter implementation of an interactive image overlay viewer with slider controls"""

    def __init__(self, master, im0, im1, title="Interactive View"):
        """
        Initialize the interactive viewer

        Parameters:
        -----------
        im0 : numpy.ndarray
            First image (overlay image) - should be grayscale or RGB
        im1 : numpy.ndarray
            Second image (background image) - should be grayscale or RGB
        title : str
            Window title
        """
        self.master = master
        self.im0_original = self._normalize_image(im0)
        self.im1_original = self._normalize_image(im1)
        self.title = title

        # Get dimensions
        self.max_r = im0.shape[0]
        self.max_c = im0.shape[1]

        # Initialize alpha mask
        self.alphas = np.ones((self.max_r, self.max_c))

        # Current slider values
        self.current_row_val = self.max_r // 2
        self.current_col_val = self.max_c // 2

        # Setup the GUI
        self.setup_gui()

        # Force row/col update
        self.update_row(self.current_row_val)

    def _normalize_image(self, img):
        """Normalize image to 0-255 range and ensure it's in the right format"""
        # Handle different image formats
        if img.dtype == np.float64 or img.dtype == np.float32:
            # Assume values are in [0, 1] range
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            # Convert to uint8
            img = img.astype(np.uint8)

        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)

        return img

    def _get_window_size(self):
        """Calculate appropriate window size based on image dimensions and screen size"""
        ratio = self.max_c / self.max_r
        display_height = self.root.winfo_screenheight()
        display_width = self.root.winfo_screenwidth()
        if ratio >= 1:
            width = min(display_width, int(display_height * ratio * 0.8))
            height = int(display_height * 0.8)
        else:
            width = int(display_width * 0.8)
            height = min(display_height, int(display_width * 0.8 / ratio))
        return width, height

    def setup_gui(self):
        """Setup the tkinter GUI"""
        # Create main window
        self.root = tk.Toplevel(self.master)
        self.root.title(self.title)
        width, height = self._get_window_size()
        self.root.geometry(f"{width}x{height}")

        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Create canvas for image display
        self.canvas = tk.Canvas(main_frame, bg="gray")
        self.canvas.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create vertical slider (Y position)
        self.y_slider_frame = ttk.Frame(main_frame)
        self.y_slider_frame.grid(row=0, column=0, sticky=(tk.N, tk.S))

        y_label = ttk.Label(self.y_slider_frame, text="Y pos")
        y_label.pack(side=tk.TOP)

        self.y_slider = ttk.Scale(
            self.y_slider_frame,
            from_=self.max_r,
            to=0,
            orient=tk.VERTICAL,
            value=self.max_r,
            command=self.update_row,
        )
        self.y_slider.pack(side=tk.TOP, fill=tk.Y, expand=True)

        self.y_value_label = ttk.Label(self.y_slider_frame, text=str(self.max_r))
        self.y_value_label.pack(side=tk.TOP)

        # Create horizontal slider (X position)
        self.x_slider_frame = ttk.Frame(main_frame)
        self.x_slider_frame.grid(row=1, column=1, sticky=(tk.W, tk.E))

        x_label = ttk.Label(self.x_slider_frame, text="X pos: ")
        x_label.pack(side=tk.LEFT)

        self.x_slider = ttk.Scale(
            self.x_slider_frame,
            from_=0,
            to=self.max_c,
            orient=tk.HORIZONTAL,
            value=0,
            command=self.update_col,
        )
        self.x_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.x_value_label = ttk.Label(self.x_slider_frame, text="0")
        self.x_value_label.pack(side=tk.LEFT)

        # Add info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        info_text = f"Image dimensions: {self.max_c} x {self.max_r}"
        info_label = ttk.Label(info_frame, text=info_text)
        info_label.pack(side=tk.LEFT, padx=5)

        # Add reset button
        reset_button = ttk.Button(info_frame, text="Reset", command=self.reset_sliders)
        reset_button.pack(side=tk.RIGHT, padx=5)

        # Initial image display
        self.update_display()

        # Bind canvas resize event
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def reset_sliders(self):
        """Reset sliders to initial positions"""
        self.y_slider.set(self.max_r)
        self.x_slider.set(0)
        self.update_display()

    def update_row(self, val):
        """Update function for Y position slider"""
        val = int(float(val))
        self.current_row_val = val
        self.y_value_label.config(text=str(val))

        # Update alpha mask
        new_alphas = np.ones_like(self.alphas)
        new_alphas[:val, :] = 0
        # Flip to match matplotlib behavior
        self.alphas = new_alphas[::-1]

        self.update_display()

    def update_col(self, val):
        """Update function for X position slider"""
        val = int(float(val))
        self.current_col_val = val
        self.x_value_label.config(text=str(val))

        # Update alpha mask based on current row value
        new_alphas = np.ones_like(self.alphas)

        # First apply row mask
        row_val = self.current_row_val
        new_alphas[:row_val, :] = 0
        new_alphas = new_alphas[::-1]

        # Then apply column mask
        new_alphas[:, :val] = 0
        self.alphas = new_alphas

        self.update_display()

    def blend_images(self):
        """Blend the two images based on current alpha mask"""
        # Create RGBA versions of both images
        im0_rgba = np.zeros((self.max_r, self.max_c, 4), dtype=np.uint8)
        im0_rgba[:, :, :3] = self.im0_original
        im0_rgba[:, :, 3] = (self.alphas * 255).astype(np.uint8)

        im1_rgb = self.im1_original.copy()

        # Create PIL images
        overlay = Image.fromarray(im0_rgba, "RGBA")
        background = Image.fromarray(im1_rgb, "RGB")

        # Convert background to RGBA
        background = background.convert("RGBA")

        # Composite images
        blended = Image.alpha_composite(background, overlay)

        return blended

    def update_display(self):
        """Update the displayed image"""
        # Blend images
        blended = self.blend_images()

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            # Calculate scaling to fit canvas while maintaining aspect ratio
            scale_x = canvas_width / self.max_c
            scale_y = canvas_height / self.max_r
            scale = min(scale_x, scale_y)

            new_width = int(self.max_c * scale)
            new_height = int(self.max_r * scale)

            # Resize image
            blended = blended.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(blended)

        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2 if canvas_width > 1 else 200,
            canvas_height // 2 if canvas_height > 1 else 200,
            image=self.photo,
            anchor=tk.CENTER,
        )

    def on_canvas_resize(self, event):
        """Handle canvas resize event"""
        self.update_display()

    def run(self):
        """Start the GUI main loop"""
        # Schedule initial update after window is fully loaded
        self.root.after(100, self.update_display)
        self.root.mainloop()


# ========== Main Entry Point ==========


def setup_logging():
    """Setup application logging."""
    logging.basicConfig(
        level=logging.DEBUG,
        # level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("distortion_correction.log"),
            logging.StreamHandler(),
        ],
    )


def main():
    """Main application entry point."""
    setup_logging()

    app = ModernDistortionCorrectionView()
    app.mainloop()


if __name__ == "__main__":
    main()
