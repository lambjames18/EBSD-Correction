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
from skimage import io, exposure
from rich import print
import imageio

# Local files
import core
import ZoomWidget as Zoom
import IO


class App(tk.Tk):
    def __init__(self, screenName=None, baseName=None):
        super().__init__(screenName, baseName)
        self._style_call("dark")
        # handle main folder
        self.update_idletasks()
        self.withdraw()
        self.folder = os.getcwd()
        self.deiconify()
        self.title("Distortion Correction")
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
        filemenu.add_command(label="Open 3D", command=lambda: self.select_data_popup("3D"))
        filemenu.add_command(label="Export 3D", command=lambda: self.apply_correction_to_h5("TPS"))
        filemenu.add_command(label="Open 2D", command=lambda: self.select_data_popup("2D"))
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
        self.clahe_active = False
        self.clahe_b = ttk.Button(self.top, text="Apply CLAHE to BSE", command=self.clahe)
        self.clahe_b.grid(row=0, column=7, sticky="ew", padx=5, pady=5)
        #
        # setup dragging
        self._drag_data = {"item": None}
        # setup viewer_left
        self.ebsd = tk.Canvas(self.viewer_left, highlightbackground=self.fg, bg=self.fg, bd=1, highlightthickness=0.2, cursor='tcross')
        self.ebsd.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        if os.name == 'posix':
            self.ebsd.bind("<Button 2>", lambda arg: self.remove_coords("ebsd", arg))
        else:
            self.ebsd.bind("<Button 3>", lambda arg: self.remove_coords("ebsd", arg))
        self.ebsd.bind("<ButtonPress-1>", lambda arg: self.add_coords("ebsd", arg))
        #
        # setup viewer right
        self.bse = tk.Canvas(self.viewer_right, highlightbackground=self.fg, bg=self.fg, bd=1, highlightthickness=0.2, cursor='tcross')
        self.bse.grid(row=0, column=1, pady=20, padx=20, sticky="nsew")
        # Set button 3 for unix/windows to be remove coords, button 2 for mac
        if os.name == 'posix':
            self.bse.bind("<Button 2>", lambda arg: self.remove_coords("bse", arg))
        else:
            self.bse.bind("<Button 3>", lambda arg: self.remove_coords("bse", arg))
        self.bse.bind("<ButtonPress-1>", lambda arg: self.add_coords("bse", arg))
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

        ### TODO: Add remove all points button
        ### TODO: Test 3D IO and everything else
        ### TODO: Fix 2D and 3D output stuff (prompt for file save, saving options, etc.)
        ### Additional things added
        self.points = {"ebsd": [], "bse": []}
        self.points_path = {"ebsd": "", "bse": ""}

    ### IO
    def select_data_popup(self, mode):
        self.w = IO.DataInput(self, mode)
        self.wait_window(self.w.w)
        if self.w.clean_exit:
            ebsd_path, bse_path = self.w.ebsd_path, self.w.bse_path
            ebsd_points_path, bse_points_path = self.w.ebsd_points_path, self.w.bse_points_path
            ebsd_res, bse_res = self.w.ebsd_res, self.w.bse_res
            e_d, b_d, e_pts, b_pts = IO.read_data(ebsd_path, bse_path, ebsd_points_path, bse_points_path)
            self.w = IO.DataSummary(self, e_d, b_d, e_pts, b_pts, ebsd_res, bse_res)
            self.wait_window(self.w.w)
            if self.w.clean_exit:
                if e_pts is None:
                    e_pts = {0: []}
                    ebsd_points_path = os.path.dirname(ebsd_path) + "/distorted_pts.txt"
                    with open(ebsd_points_path, "w", encoding="utf8") as output:
                        output.write("")
                if b_pts is None:
                    b_pts = {0: []}
                    bse_points_path = os.path.dirname(ebsd_path) + "/control_pts.txt"
                    with open(bse_points_path, "w", encoding="utf8") as output:
                        output.write("")
                if self.w.rescale:
                    b_d = IO.rescale_control(b_d, bse_res, ebsd_res)
                if self.w.flip:
                    b_d = np.flip(b_d, axis=1).copy(order='C')
                if self.w.crop:
                    ### TODO: Add multiple control images
                    self.w = IO.CropControl(self, b_d[0, :, :])
                    self.wait_window(self.w.w)
                    if self.w.clean_exit:
                        s, e = self.w.start, self.w.end
                        ### TODO: Add multiple control images
                        b_d = b_d[:, s[0]:e[0], s[1]:e[1]]
            else:
                return
            # Set the data
            self.points_path = {"ebsd": ebsd_points_path, "bse": bse_points_path}
            self.ebsd_data, self.bse_imgs = e_d, b_d
            self.ebsd_res, self.bse_res = ebsd_res, bse_res
            self.points["ebsd"], self.points["bse"] = e_pts, b_pts
            # Set the UI stuff
            self.ebsd_mode_options = list(self.ebsd_data.keys())
            self.ebsd_mode.set(self.ebsd_mode_options[0])
            self.slice_min = 0
            self.slice_max = self.ebsd_data[self.ebsd_mode_options[0]].shape[0] - 1
            self.slice_num.set(self.slice_min)
            self.ebsd_cStack = np.zeros(self.ebsd_data[self.ebsd_mode_options[0]].shape)
            # Configure UI
            self.tps_stack["state"] = "enabled"
            self.slice_picker["state"] = "readonly"
            self.slice_picker["values"] = list(np.arange(self.slice_min, self.slice_max + 1))
            self.ebsd_picker["state"] = "readonly"
            self.ebsd_picker["values"] = self.ebsd_mode_options
            # Finish
            self.folder = os.path.dirname(ebsd_path)
            self._update_viewers()
    
    ### Coords stuff
    def add_coords(self, pos, event):
        """Responds to a click on an image. Redraws the images after the click. Also saves the click location in a file."""
        i = self.slice_num.get()
        self.points[pos][i].append([event.x, event.y])
        path = self.points_path[pos]
        with open(path, "a", encoding="utf8") as output:
            output.write(f"{i} {event.x} {event.y}\n")
        self._update_inherit_options()
        self._show_points()
    
    def write_coords(self):
        for mode in ["ebsd", "bse"]:
            path = self.points_path[mode]
            data = []
            pts = self.points[mode]
            for key in pts.keys():
                s = np.hstack((np.ones((len(pts[key]), 1)) * key, pts[key]))
                data.append(s)
            data = np.vstack(data)
            np.savetxt(path, data, fmt="%i", delimiter=" ")
    
    def remove_coords(self, pos, event):
        """Remove the point closes to the clicked location, the point should be removed from both images"""
        ### TODO: Fix this to use new coords convention
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
        base_tag = tag.split(" ")[0]
        path = os.path.join(self.folder, f"ctr_pts_{self.slice_num.get()}_{pos}.txt")
        with open(path, "w", encoding="utf8") as output:
            for i in range(len(self.current_points[pos])):
                x, y = self.current_points[pos][i]
                output.write(f"{int(x)} {int(y)}\n")
        self._update_inherit_options()
        self._update_viewers()

    def _show_points(self):
        """Either turns on or turns off control point viewing"""
        if self.show_points.get() == 1:
            j = self.slice_num.get()
            pc = {"ebsd": "#FEBC11", "bse": "#EF5645"}
            viewers = {"ebsd": self.ebsd, "bse": self.bse}
            for mode in ["ebsd", "bse"]:
                pts = np.array(self.points[mode][j])
                for i, p in enumerate(pts):
                    o_item = viewers[mode].create_oval(p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1, width=2, outline=pc[mode], tags=str(i))
                    t_item = viewers[mode].create_text(
                        p[0] + 3, p[1] + 3, anchor=tk.NW, text=i, fill=pc[mode], font=("", 10, "bold"), tags=str(i) + "text"
                    )
                    bbox = viewers[mode].bbox(t_item)
                    r_item = viewers[mode].create_rectangle(bbox, fill="#FFFFFF", outline=pc[mode], tags=str(i) + "bbox")
                    viewers[mode].tag_raise(t_item, r_item)
        else:
            self.ebsd.delete("all")
            self.bse.delete("all")
            self._update_imgs()

    ### Apply stuff
    def apply(self, algo="TPS"):
        """Applies the correction algorithm and calls the interactive view"""
        ### TODO: Fix this to use new coords convention
        i = self.slice_num.get()
        referencePoints = np.array(self.points["bse"][i])
        distortedPoints = np.array(self.points["ebsd"][i])
        align = core.Alignment(referencePoints, distortedPoints, algorithm=algo)
        if algo == "TPS":
            save_name = os.path.join(self.folder, "TPS_solution.npy")
            align.get_solution(size=self.bse_im.shape, solutionFile=save_name, saveSolution=False)
        elif algo == "LR":
            raise NotImplementedError
        # save_name = os.path.join(self.folder, f"{algo}_out.tif")
        im1 = align.apply(self.ebsd_im, out="array")
        print("Creating interactive view")
        self._interactive_view(algo, im1)
        plt.close("all")

    def apply_3D(self, algo="LR"):
        """Applies the correction algorithm and calls the interactive view"""
        ### TODO: Fix this to use new coords convention
        self.config(cursor="watch")
        points = self.points
        ebsd_stack = np.sqrt(np.sum(self.ebsd_data[self.ebsd_mode.get()][...], axis=3))
        referencePoints = np.array(self.current_points["bse"])
        distortedPoints = np.array(self.current_points["ebsd"])
        print("Aligning the full ESBSD stack in mode {}".format(self.ebsd_mode.get()))
        align = core.Alignment(referencePoints, distortedPoints, algorithm=algo)
        if algo == "TPS":
            self.ebsd_cStack = align.TPS_apply_3D(points, ebsd_stack, self.bse_imgs)
        elif algo == "LR":
            raise NotImplementedError
        print("Creating interactive view")
        self.config(cursor="tcross")
        self._interactive_view(algo, self.ebsd_cStack, True)
        plt.close("all")

    def apply_correction_to_h5(self, algo):
        ### TODO: Fix this to use new coords convention
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
        points = self.points
        if len(points["ebsd"].keys()) == 0:
            print("[bold red]Error:[/bold red] No points have been selected!")
            return
        elif len(points["ebsd"].keys()) == 1:
            print("[bold Orange]Warning:[/bold orange] Only one slice has been selected! Applying it to all slices...")
        i = self.slice_num.get()
        referencePoints = np.array(self.points["bse"][i])
        distortedPoints = np.array(self.points["ebsd"][i])
        align = core.Alignment(referencePoints, distortedPoints, algorithm=algo)
        # Grab the h5 file
        print("Generating a new DREAM3D file...")
        EBSD_DIR_CORRECTED = (w := os.path.splitext(self.ebsd_path))[0] + "_corrected" + w[1]
        shutil.copyfile(self.ebsd_path, EBSD_DIR_CORRECTED)
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
                    c_stack = align.TPS_apply_3D(points, stack, self.bse_imgs)
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
        ### TODO: Fix this to use new coords convention
        # Get the control points
        i = self.slice_num.get()
        referencePoints = np.array(self.points["bse"][i])
        distortedPoints = np.array(self.points["ebsd"][i])
        align = core.Alignment(referencePoints, distortedPoints, algorithm=algo)
        # Create the output filename
        if self.slice_max == 1:
            extension = os.path.splitext(self.ebsd_path)[1]
        else:
            extension = ".tiff"
        SAVE_PATH_BSE = os.path.splitext(self.ebsd_path)[0] + "_BSE-out" + extension
        SAVE_PATH_EBSD = os.path.splitext(self.ebsd_path)[0] + "_EBSD-out" + extension
        if algo == "TPS":
            # Align the image
            align.TPS(self.bse_im.shape)
            aligned = align.TPS_apply(self.ebsd_im, out="array")
            # Correct dtype
            if aligned.dtype != np.uint16:
                aligned = (aligned / aligned.max() * 65535).astype(np.uint16)
            # Save the image
            self._mask = (slice(None), slice(None))
            _ebsd_stack = np.sqrt(np.sum(self.ebsd_data[self.ebsd_mode.get()][...], axis=3))
            size_diff = np.array(self.bse_imgs.shape) - np.array(_ebsd_stack.shape[:3])
            if size_diff[1] > 0:
                print(f"{size_diff[1]=}")
                start = size_diff[1] // 2
                end = -(size_diff[1] - start)
                self._mask = (slice(start, end), self._mask[1])
            if size_diff[2] > 0:
                print(f"{size_diff[2]=}")
                start = size_diff[2] // 2
                end = -(size_diff[2] - start)
                self._mask = (self._mask[0], slice(start, end))
            imageio.imwrite(SAVE_PATH_EBSD, aligned[self._mask])
            if self.clahe_active:
                im = exposure.equalize_adapthist(self.bse_imgs[0][self._mask], clip_limit=0.03)
            else:
                im = self.bse_imgs[0][self._mask]
            im = ((im - im.min()) / (im.max() - im.min()) * 65535).astype(np.uint16)
            imageio.imwrite(SAVE_PATH_BSE, im)
        elif algo == "LR":
            raise ValueError("algo must be TPS at this time. LR is not supported")
        print("[bold green]Correction complete![/bold green]")

    def export_CP_imgs(self):
        i = self.slice_num.get()
        pts = np.array(self.points["ebsd"][i])
        self._save_CP_img(f"{i}_ebsd", self.ebsd_im, pts, "gray", "#c2344e")
        pts = np.array(self.points["bse"][i])
        self._save_CP_img(f"{i}_bse", self.bse_im, pts, "gray", "#34c295")
        print("Control point images exported successfully.")

    def clahe(self):
        if self.clahe_active:
            self.clahe_active = False
            self.clahe_b["text"] = "Apply CLAHE to BSE"
            self._update_viewers()
        else:
            self.clahe_active = True
            self.clahe_b["text"] = "Remove CLAHE from BSE"
            self._update_viewers()

    ### UI stuff
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
        print(f"Updating viewers for slice {i} and mode {key}")
        bse_im = self.bse_imgs[i]
        ebsd_im = self.ebsd_data[key][i]
        if self.clahe_active:
            bse_im = exposure.equalize_adapthist(bse_im, clip_limit=0.03)
        self.inherit_select.set(self.inherit_select_options[0])

        # Check if there are 3 dimensions in which the last one is not needed
        if len(ebsd_im.shape) == 3 and ebsd_im.shape[-1] == 1:
            ebsd_im = ebsd_im[:, :, 0]
        if len(bse_im.shape) == 3 and bse_im.shape[-1] == 1:
            bse_im = bse_im[:, :, 0]
        # Check if there are 4 dimensions in which we just take the sum of the squares               
        if len(ebsd_im.shape) == 4:
            ebsd_im = np.sum(ebsd_im ** 2, axis=-1)
        # Check the dtype of the EBSD image, if they are a float, convert to uint8
        if ebsd_im.dtype == np.uint8:
            self.ebsd_im = ebsd_im
        else:
            ebsd_im = ebsd_im - ebsd_im.min()
            self.ebsd_im = np.around(255 * ebsd_im / ebsd_im.max(), 0).astype(np.uint8)
        # Check the dtype of the BSE image, if they are a float, convert to uint8
        if bse_im.dtype == np.uint8:
            self.bse_im = bse_im
        else:
            bse_im = bse_im - bse_im.min()
            self.bse_im = np.around(255 * bse_im / bse_im.max(), 0).astype(np.uint8)

        # Change current points dict by either reading in one or creating a new one
        # Update the images and draw points
        self._update_imgs()
        self._show_points()

    def _update_imgs(self):
        """Updates the images in the viewers"""
        self.ebsd.delete("all")
        self.bse.delete("all")
        # BSE
        self.bse["width"] = self.bse_im.shape[1]
        self.bse["height"] = self.bse_im.shape[0]
        self.bse_im_ppm = self._photo_image(self.bse_im, channels=1)
        self.bse.create_image(0, 0, anchor="nw", image=self.bse_im_ppm)
        # EBSD
        self.ebsd["width"] = self.ebsd_im.shape[1]
        self.ebsd["height"] = self.ebsd_im.shape[0]
        channels = 3 if self.ebsd_im.ndim == 3 else 1
        self.ebsd_im_ppm = self._photo_image(self.ebsd_im, channels=channels)
        self.ebsd.create_image(0, 0, anchor="nw", image=self.ebsd_im_ppm)

    def _photo_image(self, image: np.ndarray, channels: int = 1):
        """Creates a PhotoImage object that plays nicely with a tkinter canvas for viewing purposes."""
        ### TODO: There is an error here with rgb data, not sure why "truncated PPM data", probably need to scale values
        if channels == 1:
            height, width = image.shape
            data = f"P5 {width} {height} 255 ".encode() + image.astype(np.uint8).tobytes()
        else:
            height, width = image.shape[:2]
            ppm_header = f"P6 {width} {height} 255 ".encode()
            data = ppm_header + image.tobytes()
        return tk.PhotoImage(width=width, height=height, data=data, format="PPM")

    def _read_points(self, bse_path=None, ebsd_path=None):
        """Reads a set of control points"""
        if bse_path is None and ebsd_path is None:
            bse_files = [bse_path]
            ebsd_files = [ebsd_path]
        elif bse_path is None and ebsd_path is not None:
            raise ValueError("Must provide both BSE and EBSD paths")
        elif bse_path is not None and ebsd_path is None:
            raise ValueError("Must provide both BSE and EBSD paths")
        else:
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
            print(key, len(bse_pts), len(ebsd_pts))
            self.all_points[key] = {"ebsd": ebsd_pts, "bse": bse_pts}
        if self.slice_num.get() in self.all_points.keys():
            self.current_points = self.all_points[self.slice_num.get()]
        else:
            self.current_points = {"ebsd": [], "bse": []}
        self.inherit_select_options = list(self.all_points.keys())
        self.inherit_select_options.insert(0, "None")
        self.inherit_picker["values"] = self.inherit_select_options
        self.inherit_picker["state"] = "enabled"

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

    def _interactive_view(self, algo, im1, stack=False):
        """Creates an interactive view of the overlay created from the control points and the selected correction algorithm"""
        if len(im1.shape) == 3:
            im1 = self.ebsd_cStack[self.slice_num.get()]
        elif len(im1.shape) > 3:
            raise IOError("im1 must be a 3D volume or a 2D image.")
        # Correct for cropped EBSD data
        self._bse_mask = (slice(None), slice(None))
        _ebsd_stack = np.sqrt(np.sum(self.ebsd_data[self.ebsd_mode.get()][...], axis=3))
        size_diff = np.array(self.bse_imgs.shape) - np.array(_ebsd_stack.shape[:3])
        if size_diff[1] > 0:
            print(f"{size_diff[1]=}")
            start = size_diff[1] // 2
            end = -(size_diff[1] - start)
        if size_diff[2] > 0:
            print(f"{size_diff[2]=}")
            start = size_diff[2] // 2
            end = -(size_diff[2] - start)
        if not stack:
            im1 = im1[self._bse_mask]
        # Generate the figure
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
            s.configure("TLabelframe", background=self.bg, foreground=self.fg, highlightcolor=self.hl, highlightbackground=self.hl)
            s.configure("TLabelframe.Label", background=self.bg, foreground=self.fg, highlightcolor=self.hl, highlightbackground=self.hl)
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
            s.configure("TLabelframe", background=self.bg, foreground=self.fg, highlightcolor=self.hl, highlightbackground=self.hl)
            s.configure("TLabelframe.Label", background=self.bg, foreground=self.fg, highlightcolor=self.hl, highlightbackground=self.hl)

    def _easy_start(self):
        print("Running easy start...")
        # self.BSE_DIR = "D:/Research/CoNi_16/Data/3D/BSE/small/"
        # self.BSE_DIR = "D:/Research/Ta/Data/3D/AMSpall/BSE/small/"
        self.BSE_DIR = "/Users/jameslamb/Downloads/BSE/"
        # self.EBSD_DIR = "D:/Research/CoNi_16/Data/3D/CoNi16_aligned.dream3d"
        # self.EBSD_DIR = "D:/Research/Ta/Data/3D/AMSpall/TaAMS_Stripped.dream3d"
        self.EBSD_DIR = "/Users/jameslamb/Downloads/5842WCu_basic.dream3d"
        # self.folder = "D:/Research/scripts/Alignment/CoNi16_3D/"
        # self.folder = "D:/Research/scripts/Alignment/TaAMSpalled/"
        self.folder = "/Users/jameslamb/Documents/Research/scripts/EBSD-Correction/WCu/"
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
