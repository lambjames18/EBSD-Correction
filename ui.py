"""
Author: James Lamb

UI for running distortion correction
"""

# Python packages
import os
import shutil
import tkinter as tk
from tkinter import filedialog, ttk, ALL, EventType

# 3rd party packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
from rich import print
import imageio

# Local files
import core
import ZoomWidget as Zoom


class App(tk.Tk):
    def __init__(self, screenName=None, baseName=None):
        super().__init__(screenName, baseName)
        self._style_call("dark")
        # handle main folder
        self.update_idletasks()
        self.withdraw()
        self.folder = os.getcwd()
        self.deiconify()
        # Setup structure of window
        # frames
        # frame_w = 1920
        # frame_h = 1080
        # self.geometry(f"{frame_w}x{frame_h}")
        # self.resizable(False, False)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=3)
        self.rowconfigure(2, weight=1)
        self.top = ttk.Frame(self)
        self.viewer_left = ttk.Frame(self)
        self.viewer_right = ttk.Frame(self)
        self.bot = ttk.Frame(self)
        self.top.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.viewer_left.grid(row=1, column=0, sticky="nsew")
        self.viewer_right.grid(row=1, column=1, sticky="nsew")
        self.bot.grid(row=2, column=0, columnspan=2, sticky="nsew")
        # self.viewer.rowconfigure(0, weight=1)
        # self.viewer.columnconfigure(0, weight=1)
        # self.viewer.columnconfigure(1, weight=1)
        #
        # setup menubar
        self.menu = tk.Menu(self)
        filemenu = tk.Menu(self.menu, tearoff=0)
        filemenu.add_command(label="Open 3D", command=self.select_3d_data_popup)
        filemenu.add_command(label="Export 3D", command=lambda: self.apply_correction_to_h5("TPS"))
        filemenu.add_command(label="Open 2D", command=self.select_2d_data_popup)
        filemenu.add_command(label="Export 2D", command=lambda: self.apply_correction_to_tif("TPS"))
        filemenu.add_command(label="Easy start", command=self._easy_start)
        self.menu.add_cascade(label="File", menu=filemenu)
        self.config(menu=self.menu)
        # setup top
        self.show_points = tk.IntVar()
        self.show_points.set(1)
        view_pts = ttk.Checkbutton(
            self.top,
            text="Show points",
            variable=self.show_points,
            onvalue=1,
            offvalue=0,
            command=self._show_points,
        )
        view_pts.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.slice_num = tk.IntVar()
        self.slice_options = np.arange(0, 2, 1)
        self.slice_num.set(self.slice_options[0])
        self.slice_picker = ttk.Combobox(
            self.top,
            textvariable=self.slice_num,
            values=list(self.slice_options),
            height=10,
            width=5,
        )
        self.slice_picker["state"] = "disabled"
        self.slice_picker.bind("<<ComboboxSelected>>", self._update_viewers)
        self.slice_picker.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.ebsd_mode = tk.StringVar()
        self.ebsd_mode_options = ["Intensity"]
        self.ebsd_mode.set(self.ebsd_mode_options[0])
        self.ebsd_picker = ttk.Combobox(
            self.top,
            textvariable=self.ebsd_mode,
            values=self.ebsd_mode_options,
            height=10,
            width=20,
        )
        self.ebsd_picker["state"] = "disabled"
        self.ebsd_picker.bind("<<ComboboxSelected>>", self._update_viewers)
        self.ebsd_picker.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
        ex_ctr_pt_ims = ttk.Button(
            self.top, text="Export control point images", command=self.export_CP_imgs
        )
        ex_ctr_pt_ims.grid(row=0, column=3, sticky="ew", padx=5, pady=5)
        inherit_label = ttk.Label(self.top, text="Inherit from slice:")
        inherit_label.grid(row=0, column=5, sticky="ew", padx=5, pady=5)
        self.inherit_select = tk.StringVar()
        self.inherit_select_options = ["None"]
        self.inherit_select.set(self.inherit_select_options[0])
        self.inherit_picker = ttk.Combobox(
            self.top,
            textvariable=self.inherit_select,
            values=self.inherit_select_options,
            height=10,
            width=10)
        self.inherit_picker["state"] = "disabled"
        self.inherit_picker.bind("<<ComboboxSelected>>", self.inherit_action)
        self.inherit_picker.grid(row=0, column=6, sticky="ew", padx=5, pady=5)
        #
        # setup dragging
        self._drag_data = {"item": None}
        # setup viewer_left
        self.ebsd = tk.Canvas(self.viewer_left, highlightbackground=self.fg, bg=self.fg, bd=1, highlightthickness=0.2, cursor='tcross')
        self.ebsd.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        # self.ebsd.bind("<Button 1>", lambda arg: self.coords("ebsd", arg))
        self.ebsd.bind("<Button 3>", lambda arg: self.remove_coords("ebsd", arg))
        self.ebsd.bind("<ButtonPress-1>", lambda arg: self.move_point("start", "ebsd", arg))
        self.ebsd.bind("<ButtonRelease-1>", lambda arg: self.move_point("stop", "ebsd", arg))
        self.ebsd.bind("<B1-Motion>", lambda arg: self.move_point("move", "ebsd", arg))
        """
        self.ebsd = Zoom.CanvasImage(self.viewer_left, self.ebsd_img)
        """
        # self.ebsd.bind("<MouseWheel>", lambda event: self._zoom(event, "ebsd"))
        # self.ebsd.bind("<ButtonPress-3>", lambda event: self.ebsd.scan_mark(event.x, event.y))
        # self.ebsd.bind("<B3-Motion>", lambda event: self.ebsd.scan_dragto(event.x, event.y, gain=1))
        #
        # setup viewer right
        self.bse = tk.Canvas(self.viewer_right, highlightbackground=self.fg, bg=self.fg, bd=1, highlightthickness=0.2, cursor='tcross')
        self.bse.grid(row=0, column=1, pady=20, padx=20, sticky="nsew")
        # self.bse.bind("<Button 1>", lambda arg: self.coords("bse", arg))
        self.bse.bind("<Button 3>", lambda arg: self.remove_coords("bse", arg))
        self.bse.bind("<ButtonPress-1>", lambda arg: self.move_point("start", "bse", arg))
        self.bse.bind("<ButtonRelease-1>", lambda arg: self.move_point("stop", "bse", arg))
        self.bse.bind("<B1-Motion>", lambda arg: self.move_point("move", "bse", arg))
        # self.bse.bind("<MouseWheel>", lambda event: self._zoom(event, "bse"))
        # self.bse.bind("<ButtonPress-3>", lambda event: self.bse.scan_mark(event.x, event.y))
        # self.bse.bind("<B3-Motion>", lambda event: self.bse.scan_dragto(event.x, event.y, gain=1))
        #
        # handle points
        self.all_points = {}
        self.current_points = {"ebsd": [], "bse": []}
        #
        # setup bot
        tps_l = ttk.Label(self.bot, text="Thin-Plate Spline Correction:")
        tps_l.grid(row=0, column=0, sticky="e", padx=5, pady=5)
        tps = ttk.Button(self.bot, text="View slice", command=lambda: self.apply("TPS"))
        tps.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.tps_stack = ttk.Button(
            self.bot,
            text="Apply to stack",
            command=lambda: self.apply_3D("TPS"),
            state="disabled",
        )
        self.tps_stack.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
        # fixh5TPS = ttk.Button(
        #     self.bot,
        #     text="Save TPS correction in DREAM3D file",
        #     command=lambda: self.apply_correction_to_h5("TPS"),
        # )
        # fixh5TPS.grid(row=0, column=5, columnspan=2, sticky='ew', padx=5, pady=5)
        # lr_l = tk.Label(self.bot, text="Linear Regression Correction:")
        # lr_l.grid(row=1, column=0, sticky="e")
        # lr = tk.Button(self.bot, text="View slice", command=lambda: self.apply("LR"), fg="black")
        # lr.grid(row=1, column=1, sticky="ew")
        # lr_stack = tk.Button(
        #     self.bot,
        #     text="Apply to stack",
        #     fg="black",
        #     command=lambda: self.apply_3D("LR"),
        # )
        # lr_stack.grid(row=1, column=2, sticky="ew")
        # lr_deg_l = tk.Label(self.bot, text="Polynomial order:")
        # lr_deg_l.grid(row=1, column=3, sticky="e")
        # self.lr_degree = tk.Entry(self.bot)
        # self.lr_degree.insert(0, "3")
        # self.lr_degree.grid(row=1, column=4, sticky="ew")
        # fixh5LR = tk.Button(
        #     self.bot,
        #     text="Save LR correction in DREAM3D file",
        #     command=lambda: self.apply_correction_to_h5("LR"),
        # )
        # fixh5LR.grid(row=1, column=5, columnspan=2)

    def select_3d_data_popup(self):
        self.w = tk.Toplevel(self)
        self.w.rowconfigure(0, weight=1)
        self.w.columnconfigure(0, weight=1)
        master = ttk.Frame(self.w)
        master.grid(row=0, column=0, sticky="nsew")
        frame_w = 1920 // 6
        frame_h = 1080 // 5
        self.w.geometry(f"{frame_w}x{frame_h}")
        self.resizable(False, False)
        for i in range(5):
            master.rowconfigure(i, weight=1)
        master.columnconfigure(0, weight=1)
        des = ttk.Label(master, text="Select relevant folders/files", justify='center')
        des.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        folder = ttk.Button(
            master,
            text="Select folder for control points",
            command=self._get_FOLDER_dir,
        )
        folder.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.w.d3d = ttk.Button(
            master, text="Select Dream3d file", command=self._get_EBSD_dir,
        )
        self.w.d3d.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.w.d3d["state"] = "disabled"
        self.w.bse = ttk.Button(
            master, text="Select BSE folder", command=self._get_BSE_dir,
        )
        self.w.bse.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.w.bse["state"] = "disabled"
        self.w.close = ttk.Button(
            master,
            text="Close (reading in data will take a few moments)",
            command=self._startup_items,
        )
        self.w.close["state"] = "disabled"
        self.w.close.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        self.wait_window(self.w)
    
    def select_2d_data_popup(self):
        bse_path = filedialog.askopenfilename(title="Select control image", filetypes=[("TIF", "*.tif"), ("TIFF", "*.tiff"), ("All files", "*.*")], initialdir=self.folder)
        bse_path_folder = os.path.dirname(bse_path)
        ebsd_path = filedialog.askopenfilename(title="Select distorted image", filetypes=[("TIF", "*.tif"), ("TIFF", "*.tiff"), ("All files", "*.*")], initialdir=bse_path_folder)
        bse_im = io.imread(bse_path, as_gray=True)
        self.ebsd_im = io.imread(ebsd_path, as_gray=True)
        self.ebsd_mode_options = ["Intensity"]
        self.ebsd_mode = tk.StringVar()
        self.ebsd_mode.set(self.ebsd_mode_options[0])
        self.slice_min = 0
        self.slice_max = 0
        self.slice_num = tk.IntVar()
        self.slice_num.set(self.slice_min)
        self.bse_imgs = np.array(bse_im).reshape(1, bse_im.shape[0], bse_im.shape[1])
        self.ebsd_data = {"Intensity": np.array(self.ebsd_im).reshape(1, self.ebsd_im.shape[0], self.ebsd_im.shape[1], 1)}
        # Configure UI
        self.tps_stack["state"] = "disabled"
        self.slice_picker["state"] = "disabled"
        self.ebsd_picker["state"] = "disabled"
        # Update the viewers
        self._update_viewers()

    def coords(self, pos, event):
        """Responds to a click on an image. Redraws the images after the click. Also saves the click location in a file."""
        i = self.slice_num.get()
        self.current_points[pos].append([event.x, event.y])
        self.all_points[i] = self.current_points
        path = os.path.join(self.folder, f"ctr_pts_{i}_{pos}.txt")
        with open(path, "a", encoding="utf8") as output:
            output.write(f"{event.x} {event.y}\n")
        self._update_inherit_options()
        self._show_points()
    
    def remove_coords(self, pos, event):
        """Remove the point closes to the clicked location, the point should be removed from both images"""
        if pos == 'bse':
            closest = self.bse.find_closest(event.x, event.y)[0]
            tag = self.bse.itemcget(closest, "tags")
        elif pos == 'ebsd':
            closest = self.ebsd.find_closest(event.x, event.y)[0]
            tag = self.ebsd.itemcget(closest, "tags")
        if "current" in tag:
            tag = tag.replace("current", "").strip()
        if tag == "":
            print("No point to remove")
            return
        self.current_points[pos].pop(int(tag))
        self.all_points[self.slice_num.get()] = self.current_points
        path = os.path.join(self.folder, f"ctr_pts_{self.slice_num.get()}_{pos}.txt")
        with open(path, "w", encoding="utf8") as output:
            for i in range(len(self.current_points[pos])):
                x, y = self.current_points[pos][i]
                output.write(f"{int(x)} {int(y)}\n")
        self._update_inherit_options()
        self._update_viewers()

    def move_point(self, state, pos, event):
        if pos == 'ebsd':
            alias = self.ebsd
        elif pos == 'bse':
            alias = self.bse
        if event.state % 2 == 1:
            alias.config(cursor="fleur")
            if state == 'start':
                print("Starting movement")
                closest = alias.find_closest(event.x, event.y, halo=10)[0]
                tag = alias.itemcget(closest, "tags")
                if "current" in tag:
                    tag = tag.replace("current", "").strip()
                if tag == "": return
                self._drag_data["item"] = tag
            elif state == 'stop':
                print("Stopping movement")
                self._drag_data["item"] = None
                alias.config(cursor="tcross")
                self._update_inherit_options()
            elif state == 'move':
                if self._drag_data["item"] is None: return
                self.current_points[pos][int(self._drag_data["item"])][0] = event.x
                self.current_points[pos][int(self._drag_data["item"])][1] = event.y
                self._update_viewers()
        else:
            alias.config(cursor="tcross")
            if state == 'start':
                self.coords(pos, event)

    def inherit_action(self, *args):
        i = self.slice_num.get()
        selection = self.inherit_select.get()
        if selection == "None":
            return
        self.current_points = {"ebsd": self.all_points[int(selection)]["ebsd"].copy(),
                               "bse": self.all_points[int(selection)]["bse"].copy()}
        self.all_points[i] = self.current_points
        path = os.path.join(self.folder, f"ctr_pts_{self.slice_num.get()}_ebsd.txt")
        with open(path, "w", encoding="utf8") as output:
            for i in range(len(self.current_points['ebsd'])):
                x, y = self.current_points['ebsd'][i]
                output.write(f"{int(x)} {int(y)}\n")
        path = os.path.join(self.folder, f"ctr_pts_{self.slice_num.get()}_bse.txt")
        with open(path, "w", encoding="utf8") as output:
            for i in range(len(self.current_points['bse'])):
                x, y = self.current_points['bse'][i]
                output.write(f"{int(x)} {int(y)}\n")
        self._update_inherit_options()
        self._update_viewers()

    def _update_inherit_options(self):
        i = self.slice_num.get()
        if i not in self.inherit_select_options:
            self.inherit_select_options.append(i)
            self.inherit_select_options[1:] = sorted(self.inherit_select_options[1:])
            self.inherit_picker['values'] = self.inherit_select_options

    def apply(self, algo="TPS"):
        """Applies the correction algorithm and calls the interactive view"""
        referencePoints = np.array(self.current_points["bse"])
        distortedPoints = np.array(self.current_points["ebsd"])
        align = core.Alignment(referencePoints, distortedPoints, algorithm=algo)
        if algo == "TPS":
            save_name = os.path.join(self.folder, "TPS_mapping.npy")
            align.get_solution(size=self.bse_im.shape, solutionFile=save_name, saveSolution=False)
        elif algo == "LR":
            save_name = os.path.join(self.folder, "LR_mapping.npy")
            align.get_solution(
                degree=int(self.lr_degree.get()), solutionFile=save_name, saveSolution=False
            )
        save_name = os.path.join(self.folder, f"{algo}_out.tif")
        im1 = align.apply(self.ebsd_im, out="array")
        print("Creating interactive view")
        self._interactive_view(algo, im1)
        plt.close("all")

    def apply_3D(self, algo="LR"):
        """Applies the correction algorithm and calls the interactive view"""
        points = self.all_points
        ebsd_stack = np.sqrt(np.sum(self.ebsd_data[self.ebsd_mode.get()][...], axis=3))
        referencePoints = np.array(self.current_points["bse"])
        distortedPoints = np.array(self.current_points["ebsd"])
        print("Aligning the full ESBSD stack in mode {}".format(self.ebsd_mode.get()))
        align = core.Alignment(referencePoints, distortedPoints, algorithm=algo)
        if algo == "TPS":
            self.ebsd_cStack = align.TPS_apply_3D(points, ebsd_stack)
        elif algo == "LR":
            self.ebsd_cStack = align.LR_3D_Apply(points, ebsd_stack, deg=3)
        print("Creating interactive view")
        self._interactive_view(algo, self.ebsd_cStack, True)
        plt.close("all")

    def apply_correction_to_h5(self, algo):
        dtypes = {b"DataArray<uint8_t>": np.uint8,
                  b"DataArray<int8_t>": np.int8,
                  b"DataArray<uint16_t>": np.uint16,
                  b"DataArray<int16_t>": np.int16,
                  b"DataArray<uint32_t>": np.uint32,
                  b"DataArray<int32_t>": np.int32,
                  b"DataArray<uint64_t>": np.uint64,
                  b"DataArray<int64_t>": np.int64,
                  b"DataArray<float>": np.float32,
                  b"DataArray<double>": np.float64,
                  b"DataArray<bool>": bool}
        points = self.all_points
        if len(points.keys()) == 0:
            print("[bold red]Error:[/bold red] No points have been selected!")
            return
        elif len(points.keys()) == 1:
            print("[bold Orange]Warning:[/bold orange] Only one slice has been selected! Applying it to all slices...")
        referencePoints = np.array(self.current_points["bse"])
        distortedPoints = np.array(self.current_points["ebsd"])
        align = core.Alignment(referencePoints, distortedPoints, algorithm=algo)
        # Grab the h5 file
        print("Generating a new DREAM3D file...")
        EBSD_DIR_CORRECTED = (w := os.path.splitext(self.EBSD_DIR))[0] + "_corrected" + w[1]
        shutil.copyfile(self.EBSD_DIR, EBSD_DIR_CORRECTED)
        h5 = h5py.File(EBSD_DIR_CORRECTED, "r+")
        # Actually apply it here
        keys = list(h5["DataContainers/ImageDataContainer/CellData"])
        print(f"[bold green]Success![/bold green] Applying to volume ({len(keys)} modes)")
        for key in keys:
            # Get stack of one mode and determine characteristics
            ebsd_stack = h5["DataContainers/ImageDataContainer/CellData/" + key]
            dtype = dtypes[ebsd_stack.attrs["ObjectType"]]
            ebsd_stack = np.array(ebsd_stack[...])
            n_dims = ebsd_stack.shape[-1]
            # Loop over all dimensions
            print(f"  -> Correcting {key} ({n_dims} components of type {dtype})")
            for i in range(ebsd_stack.shape[-1]):
                if algo == "TPS":
                    # Isolate one dimension and correct
                    stack = np.copy(ebsd_stack[:, :, :, i])
                    c_stack = align.TPS_apply_3D(points, stack)
                    if dtype == np.uint8:
                        c_stack = np.around(255 * c_stack / c_stack.max(), 0).astype(np.uint8)
                    elif dtype == np.uint16:
                        c_stack = np.around(65535 * c_stack / c_stack.max(), 0).astype(np.uint16)
                    elif dtype == np.uint32:
                        c_stack = np.around(4294967295 * c_stack / c_stack.max(), 0).astype(np.uint32)
                    elif dtype == np.uint64:
                        c_stack = np.around(18446744073709551615 * c_stack / c_stack.max(), 0).astype(np.uint64)
                    else:
                        c_stack = c_stack.astype(dtype)
                    
                    # Fill original stack
                    ebsd_stack[:, :, :, i] = c_stack
                elif algo == "LR":
                    raise ValueError("algo must be TPS at this time. LR is not supported")
            # Write new stack to the h5
            h5["DataContainers/ImageDataContainer/CellData/" + key][...] = ebsd_stack
        h5.close()
        print("[bold green]Correction complete![/bold green]")

    def apply_correction_to_tif(self, algo):
        # Get the control points
        referencePoints = np.array(self.current_points["bse"])
        distortedPoints = np.array(self.current_points["ebsd"])
        align = core.Alignment(referencePoints, distortedPoints, algorithm=algo)
        # Create the output filename
        EBSD_DIR_CORRECTED = (w := os.path.splitext(self.EBSD_DIR))[0] + "_corrected" + w[1]
        if algo == "TPS":
            # Align the image
            aligned = align.TPS(self.bse_im.shape)
            # Correct dtype
            if aligned.dtype != np.uint16:
                aligned = (aligned / aligned.max() * 65535).astype(np.uint16)
        elif algo == "LR":
            raise ValueError("algo must be TPS at this time. LR is not supported")
        imageio.mimsave(EBSD_DIR_CORRECTED, aligned)
        print("[bold green]Correction complete![/bold green]")

    def export_CP_imgs(self):
        i = self.slice_num.get()
        pts = np.array(self.current_points["ebsd"])
        self._save_CP_img(f"{i}_ebsd", self.ebsd_im, pts, "gray", "#c2344e")
        pts = np.array(self.current_points["bse"])
        self._save_CP_img(f"{i}_bse", self.bse_im, pts, "gray", "#34c295")
        print("Control point images exported successfully.")

    def _zoom(self, event, view):
        """Zooms in or out on the image in the specified view"""
        print("Zooming not supported yet")
        # factor = 1.001 * event.delta
        # if view == 'ebsd':
        #     x = self.ebsd.canvasx(event.x)
        #     y = self.ebsd.canvasy(event.y)
        #     self.ebsd.scale(ALL, x, y, factor, factor)
        # elif view == 'bse':
        #     x = self.bse.canvasx(event.x)
        #     y = self.bse.canvasy(event.y)
        #     self.bse.scale(ALL, x, y, factor, factor)
            
    def _update_viewers(self, *args):
        i = self.slice_num.get()
        key = self.ebsd_mode.get()
        bse_im = self.bse_imgs[i]
        ebsd_im = self.ebsd_data[key][i]
        self.inherit_select.set(self.inherit_select_options[0])
        if ebsd_im.shape[-1] == 1:
            ebsd_im = ebsd_im[:, :, 0]
        else:
            ebsd_im = np.sqrt(np.sum(ebsd_im ** 2, axis=2))

        if ebsd_im.dtype == np.uint8:
            self.ebsd_im = ebsd_im
        else:
            self.ebsd_im = np.around(255 * ebsd_im / ebsd_im.max(), 0).astype(np.uint8)

        if bse_im.dtype != np.uint8:
            bse_im = np.around(255 * bse_im / bse_im.max(), 0).astype(np.uint8)
        self.bse_im = bse_im
        # Change current points dict by either reading in one or creating a new one
        if self.slice_num.get() in self.all_points.keys():
            self.current_points = self.all_points[self.slice_num.get()]
        else:
            self.current_points = {"ebsd": [], "bse": []}
        # Update the images and draw points
        self._update_imgs()
        self._show_points()

    def _show_points(self):
        """Either turns on or turns off control point viewing"""
        if self.show_points.get() == 1:
            pc = "#FEBC11"
            for i, p in enumerate(self.current_points["ebsd"]):
                self.ebsd.create_oval(p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1, width=2, outline=pc, tags=str(i))
                self.ebsd.create_text(
                    p[0] + 3, p[1] + 3, anchor=tk.NW, text=i, fill=pc, font=("", 10, "bold"), tags=str(i)
                )
            pc = "#EF5645"
            for i, p in enumerate(self.current_points["bse"]):
                self.bse.create_oval(p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1, width=2, outline=pc, tags=str(i))
                self.bse.create_text(
                    p[0] + 3, p[1] + 3, anchor=tk.NW, text=i, fill=pc, font=("", 10, "bold"), tags=str(i)
                )
        else:
            self.ebsd.delete("all")
            self.bse.delete("all")
            self._update_imgs()

    def _update_imgs(self):
        """Updates the images in the viewers"""
        self.ebsd.delete("all")
        self.bse.delete("all")
        # BSE
        self.bse["width"] = self.bse_im.shape[1]
        self.bse["height"] = self.bse_im.shape[0]
        self.bse_im_ppm = self._photo_image(self.bse_im)
        self.bse.create_image(0, 0, anchor="nw", image=self.bse_im_ppm)
        # EBSD
        self.ebsd["width"] = self.ebsd_im.shape[1]
        self.ebsd["height"] = self.ebsd_im.shape[0]
        self.ebsd_im_ppm = self._photo_image(self.ebsd_im)
        self.ebsd.create_image(0, 0, anchor="nw", image=self.ebsd_im_ppm)

    def _photo_image(self, image: np.ndarray):
        """Creates a PhotoImage object that plays nicely with a tkinter canvas for viewing purposes."""
        height, width = image.shape
        data = f"P5 {width} {height} 255 ".encode() + image.astype(np.uint8).tobytes()
        return tk.PhotoImage(width=width, height=height, data=data, format="PPM")

    def _read_points(self):
        """Reads a set of control points"""
        ebsd_files = [
            os.path.join(self.folder, f)
            for f in os.listdir(self.folder)
            if "ebsd" in f and "txt" in f
        ]
        bse_files = [
            os.path.join(self.folder, f)
            for f in os.listdir(self.folder)
            if "bse" in f and "txt" in f
        ]
        for i in range(len(bse_files)):
            base = os.path.splitext(os.path.basename(bse_files[i]))[0]
            key = int(base.split("_")[-2])
            bse_pts = list(np.loadtxt(bse_files[i], dtype=int, delimiter=' ').reshape(-1, 2))
            ebsd_pts = list(np.loadtxt(ebsd_files[i], dtype=int, delimiter=' ').reshape(-1, 2))
            self.all_points[key] = {"ebsd": ebsd_pts, "bse": bse_pts}
        if self.slice_num.get() in self.all_points.keys():
            self.current_points = self.all_points[self.slice_num.get()]
        else:
            self.current_points = {"ebsd": [], "bse": []}
        self.inherit_select_options = list(self.all_points.keys())
        self.inherit_select_options.insert(0, "None")
        self.inherit_picker["values"] = self.inherit_select_options
        self.inherit_picker["state"] = "enabled"

    def _startup_items(self):
        print("Running startup")
        self._open_BSE_stack(self.BSE_DIR)
        self._read_h5(self.EBSD_DIR)
        self.ebsd_mode_options = list(self.ebsd_data.keys())
        self.ebsd_mode.set(self.ebsd_mode_options[0])
        self.slice_min = 0
        self.slice_max = self.ebsd_data[self.ebsd_mode_options[0]].shape[0] - 1
        self.slice_num.set(self.slice_min)
        try:
            self.w.destroy()
        except AttributeError:
            pass
        # Configure UI
        self.tps_stack["state"] = "enabled"
        self.slice_picker["state"] = "readonly"
        self.slice_picker["values"] = list(np.arange(self.slice_min, self.slice_max + 1))
        self.ebsd_picker["state"] = "readonly"
        self.ebsd_picker["values"] = self.ebsd_mode_options
        # Read points and setup viewers
        self._read_points()
        self._update_viewers()
        print("Startup complete")

    def _save_CP_img(self, name, im, pts, cmap, tc="red"):
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        ax.imshow(im, cmap=cmap)
        for i in range(pts.shape[0]):
            ax.scatter(pts[i, 0], pts[i, 1], c=tc, s=1)
            ax.text(pts[i, 0] + 2, pts[i, 1] + 2, i, c=tc, fontweight="bold")
        ax.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.tight_layout()
        fig.savefig(f"{os.path.join(self.folder,name)}_points.png")
        plt.close()

    def _get_FOLDER_dir(self):
        """Gets the folder where all control points and other images will be saved."""
        self.folder = filedialog.askdirectory(
            initialdir=self.folder, title="Select control points folder"
        )
        self.w.d3d["state"] = "active"

    def _get_EBSD_dir(self):
        """Gets the H5 file and saves its location"""
        self.EBSD_DIR = filedialog.askopenfilename(
            initialdir=self.folder, title="Select Dream3D file"
        )
        self.w.bse["state"] = "active"

    def _get_BSE_dir(self):
        """Gets the folder containing the BSE images and saves the location"""
        self.BSE_DIR = filedialog.askdirectory(initialdir=self.EBSD_DIR, title="Select BSE folder")
        self.w.close["state"] = "active"

    def _read_h5(self, path: str):
        """Reads a Dream3D file"""
        h5 = h5py.File(path, "r")
        self.ebsd_data = h5["DataContainers/ImageDataContainer/CellData"]
        key = list(self.ebsd_data.keys())[0]
        self.ebsd_cStack = np.zeros(self.ebsd_data[key].shape)

    def _open_BSE_stack(self, imgs_path: str):
        """Reads a stack of BSE images into one numpy array"""
        paths_tiff = sorted(
            [path for path in os.listdir(imgs_path) if os.path.splitext(path)[1] == ".tiff"],
            key=lambda x: int(x.replace(".tiff", "")),
        )
        paths_tif = sorted(
            [path for path in os.listdir(imgs_path) if os.path.splitext(path)[1] == ".tif"],
            key=lambda x: int(x.replace(".tif", "")),
        )
        paths = []
        for p in [paths_tiff, paths_tif]:
            if len(p) > len(paths):
                paths = p
        if len(paths) == 0:
            raise ValueError("BSE images must be either tiff or tif. No other format is supported.")
        bse_imgs = []
        for i in range(len(paths)):
            p = os.path.join(imgs_path, paths[i])
            im = io.imread(p, as_gray=True).astype(np.float32)
            im = np.around(255 * im / im.max(), 0).astype(np.uint8)
            bse_imgs.append(im)
        self.bse_imgs = np.array(bse_imgs, dtype=np.uint8)
        print(f"{self.bse_imgs.shape[0]} BSE images imported!")

    def _interactive_view(self, algo, im1, stack=False):
        """Creates an interactive view of the overlay created from the control points and the selected correction algorithm"""
        if len(im1.shape) == 3:
            im1 = self.ebsd_cStack[self.slice_num.get()]
        elif len(im1.shape) > 3:
            raise IOError("im1 must be a 3D volume or a 2D image.")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        im0 = self.bse_imgs[self.slice_num.get()]
        max_r = im0.shape[0]
        max_c = im0.shape[1]
        max_s = self.slice_max
        ax.set_title(f"{algo} Output (Slice {self.slice_num.get()})")
        alphas = np.ones(im0.shape)
        # Show images
        im_ebsd = ax.imshow(im1, cmap="gray")
        im = ax.imshow(im0, alpha=alphas, cmap="gray")
        # Put slider on
        plt.subplots_adjust(left=0.15, bottom=0.15)
        left = ax.get_position().x0
        bot = ax.get_position().y0
        width = ax.get_position().width
        height = ax.get_position().height
        axrow = plt.axes([left - 0.15, bot, 0.05, height])
        axcol = plt.axes([left, bot - 0.15, width, 0.05])
        row_slider = Slider(
            ax=axrow,
            label="Y pos",
            valmin=0,
            valmax=max_r,
            valinit=max_r,
            valstep=1,
            orientation="vertical",
        )
        col_slider = Slider(
            ax=axcol,
            label="X pos",
            valmin=0,
            valmax=max_c,
            valinit=max_c,
            valstep=1,
            orientation="horizontal",
        )

        # Define update functions
        def update_row(val):
            val = int(np.around(val, 0))
            new_alphas = np.copy(alphas)
            new_alphas[:val, :] = 0
            im.set(alpha=new_alphas[::-1])
            fig.canvas.draw_idle()

        def update_col(val):
            val = int(np.around(val, 0))
            new_alphas = np.copy(alphas)
            new_alphas[:, :val] = 0
            im.set(alpha=new_alphas)
            fig.canvas.draw_idle()

        def change_image(val):
            val = int(np.around(val, 0))
            im1 = self.ebsd_cStack[val]
            im0 = self.bse_imgs[val]
            im.set_data(im0)
            im_ebsd.set_data(im1)
            ax.set_title(f"{algo} Output (Slice {val})")
            im.axes.figure.canvas.draw()
            im_ebsd.axes.figure.canvas.draw()
            fig.canvas.draw_idle()

        # Enable update functions
        row_slider.on_changed(update_row)
        col_slider.on_changed(update_col)
        # Create slice slider if need be
        if stack:
            axslice = plt.axes([left + 0.65, bot, 0.05, height])
            slice_slider = Slider(
                ax=axslice,
                label="Slice #",
                valmin=0,
                valmax=max_s,
                valinit=self.slice_num.get(),
                valstep=1,
                orientation="vertical",
            )
            slice_slider.on_changed(change_image)
        plt.show()
    
    def _style_call(self, style='dark'):
        if style == 'dark':
            self.bg = "#333333"
            self.fg = "#ffffff"
            self.hl = "#007fff"
            self.tk.call('source', r"./theme/dark.tcl")
            s = ttk.Style(self)
            s.theme_use("azure-dark")
            s.configure("TFrame", background=self.bg)
            s.configure("TLabel", background=self.bg, foreground=self.fg)
            s.configure("TCheckbutton", background=self.bg, foreground=self.fg)
        elif style == 'light':
            self.bg = "#ffffff"
            self.fg = "#000000"
            self.hl = "#007fff"
            self.tk.call('source', r"./theme/light.tcl")
            s = ttk.Style(self)
            s.theme_use("azure-light")
            s.configure("TFrame", background=self.bg)
            s.configure("TLabel", background=self.bg, foreground=self.fg)
            s.configure("TCheckbutton", background=self.bg, foreground=self.fg)

    def _easy_start(self):
        print("Running easy start...")
        self.BSE_DIR = "D:/Research/CoNi_16/Data/3D/BSE/small/"
        # self.BSE_DIR = "D:/Research/Ta_AM-Spalled/Data/3D/BSE/small/"
        self.EBSD_DIR = "D:/Research/CoNi_16/Data/3D/CoNi16_aligned.dream3d"
        # self.EBSD_DIR = "D:/Research/ta_AM-Spalled/Data/3D/Ta_AM-Spalled_aligned.dream3d"
        self.folder = "D:/Research/scripts/Alignment/CoNi16_3D/"
        # self.folder = "D:/Research/scripts/Alignment/Ta_AM-Spalled_3D/"
        self._startup_items()


if __name__ == "__main__":
    # s = ttk.Style()
    # print(s.theme_names())
    # s.theme_use("xpnative")
    # root = tk.Tk()
    # root.tk.call("source", "azure.tcl")
    # root.tk.call("set_theme", "dark")
    app = App()
    app.mainloop()
