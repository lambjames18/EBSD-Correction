"""
Author: James Lamb

UI for running distortion correction
"""

# Python packages
import os
import shutil
import tkinter as tk
from tkinter import filedialog, ttk

# 3rd party packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io, exposure, transform
import imageio

# Local files
import core
import IO
import InteractiveView as IV

### TODO: Fix cropping in output
### TODO: Test 3D output
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
        self.columnconfigure((0, 2), weight=5)
        self.columnconfigure(1, weight=1)
        self.rowconfigure((0, 4), weight=5)
        self.rowconfigure((1, 4), weight=1)
        self.rowconfigure(2, weight=10)
        self.top = ttk.Frame(self)
        separator1 = ttk.Separator(self, orient=tk.HORIZONTAL)
        self.viewer_left = ttk.Frame(self)
        separator2 = ttk.Separator(self, orient=tk.VERTICAL)
        self.viewer_right = ttk.Frame(self)
        separator3 = ttk.Separator(self, orient=tk.HORIZONTAL)
        self.bot_left = ttk.Frame(self)
        separator4 = ttk.Separator(self, orient=tk.VERTICAL)
        self.bot_right = ttk.Frame(self)
        self.top.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.viewer_left.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.viewer_right.grid(row=2, column=2, sticky="nsew", padx=5, pady=5)
        self.bot_left.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        self.bot_right.grid(row=4, column=2, sticky="nsew", padx=5, pady=5)
        separator1.grid(row=1, column=0, columnspan=3, sticky="ew")
        separator2.grid(row=2, column=1, sticky="ns")
        separator3.grid(row=3, column=0, columnspan=3, sticky="ew")
        separator4.grid(row=4, column=1, sticky="ns")
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
        applymenu = tk.Menu(self.menu, tearoff=0)
        applymenu.add_command(label="TPS", command=lambda: self.apply("TPS"))
        applymenu.add_command(label="TPS 3D", command=lambda: self.apply_3D("TPS"))
        self.menu.add_cascade(label="Apply", menu=applymenu, state="disabled")
        self.config(menu=self.menu)
        # setup top
        self.show_points = tk.IntVar()
        self.show_points.set(1)
        self.view_pts = ttk.Checkbutton(
            self.top,
            text="Show points",
            variable=self.show_points,
            onvalue=1,
            offvalue=0,
            command=self._show_points,
            state="disabled",
        )
        self.view_pts.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        sep = ttk.Separator(self.top, orient=tk.VERTICAL)
        sep.grid(row=0, column=1, sticky="ns")
        self.slice_num = tk.IntVar()
        self.slice_options = np.arange(0, 2, 1)
        self.slice_num.set(self.slice_options[0])
        slice_num_label = ttk.Label(self.top, text="Slice number:")
        slice_num_label.grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.slice_picker = ttk.Combobox(
            self.top,
            textvariable=self.slice_num,
            values=list(self.slice_options),
            height=10,
            width=5,
        )
        self.slice_picker["state"] = "disabled"
        self.slice_picker.bind("<<ComboboxSelected>>", self._update_viewers)
        self.slice_picker.grid(row=0, column=3, sticky="ew", padx=5, pady=5)
        sep = ttk.Separator(self.top, orient=tk.VERTICAL)
        sep.grid(row=0, column=4, sticky="ns")
        self.ex_ctr_pt_ims = ttk.Button(
            self.top, text="Export control point images", command=self.export_CP_imgs, state="disabled"
        )
        self.ex_ctr_pt_ims.grid(row=0, column=5, sticky="ew", padx=5, pady=5)
        #
        # setup viewer_left
        l = ttk.Label(self.viewer_left, text="EBSD/Distorted image", anchor=tk.CENTER)
        l.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.ebsd = tk.Canvas(self.viewer_left, highlightbackground=self.bg, bg=self.bg, bd=1, highlightthickness=0.2, cursor='tcross', width=600, height=600)
        self.ebsd.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        if os.name == 'posix':
            self.ebsd.bind("<Button 2>", lambda arg: self.remove_coords("ebsd", arg))
        else:
            self.ebsd.bind("<Button 3>", lambda arg: self.remove_coords("ebsd", arg))
        self.ebsd.bind("<ButtonPress-1>", lambda arg: self.add_coords("ebsd", arg))
        self.ebsd_hscroll = ttk.Scrollbar(self.viewer_left, orient=tk.HORIZONTAL, command=self.ebsd.xview, cursor="sb_h_double_arrow")
        self.ebsd_hscroll.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.ebsd_vscroll = ttk.Scrollbar(self.viewer_left, orient=tk.VERTICAL, command=self.ebsd.yview, cursor="sb_v_double_arrow")
        self.ebsd_vscroll.grid(row=1, column=1, sticky="ns", padx=5, pady=5)
        self.ebsd.config(xscrollcommand=self.ebsd_hscroll.set, yscrollcommand=self.ebsd_vscroll.set)
        #
        # setup viewer right
        l = ttk.Label(self.viewer_right, text="BSE/Control image", anchor=tk.CENTER)
        l.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.bse = tk.Canvas(self.viewer_right, highlightbackground=self.bg, bg=self.bg, bd=1, highlightthickness=0.2, cursor='tcross', width=600, height=600)
        self.bse.grid(row=1, column=0, pady=5, padx=5, sticky="nsew")
        if os.name == 'posix':
            self.bse.bind("<Button 2>", lambda arg: self.remove_coords("bse", arg))
        else:
            self.bse.bind("<Button 3>", lambda arg: self.remove_coords("bse", arg))
        self.bse.bind("<ButtonPress-1>", lambda arg: self.add_coords("bse", arg))
        self.bse_hscroll = ttk.Scrollbar(self.viewer_right, orient=tk.HORIZONTAL, command=self.bse.xview, cursor="sb_h_double_arrow")
        self.bse_hscroll.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.bse_vscroll = ttk.Scrollbar(self.viewer_right, orient=tk.VERTICAL, command=self.bse.yview, cursor="sb_v_double_arrow")
        self.bse_vscroll.grid(row=1, column=1, sticky="ns", padx=5, pady=5)
        self.bse.config(xscrollcommand=self.bse_hscroll.set, yscrollcommand=self.bse_vscroll.set)
        #
        # setup bottom left
        self.clear_ebsd_points = ttk.Button(self.bot_left, text="Clear points", command=lambda: self.clear_points("ebsd"), state="disabled")
        self.clear_ebsd_points.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        sep = ttk.Separator(self.bot_left, orient=tk.VERTICAL)
        sep.grid(row=0, column=1, sticky="ns")
        ebsd_mode_label = ttk.Label(self.bot_left, text="EBSD mode:")
        ebsd_mode_label.grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.ebsd_mode = tk.StringVar()
        self.ebsd_mode_options = ["Intensity"]
        self.ebsd_mode.set(self.ebsd_mode_options[0])
        self.ebsd_picker = ttk.Combobox(
            self.bot_left,
            textvariable=self.ebsd_mode,
            values=self.ebsd_mode_options,
            height=10,
            width=20,
        )
        self.ebsd_picker["state"] = "disabled"
        self.ebsd_picker.bind("<<ComboboxSelected>>", self._update_viewers)
        self.ebsd_picker.grid(row=0, column=3, sticky="ew", padx=5, pady=5)
        sep = ttk.Separator(self.bot_left, orient=tk.VERTICAL)
        sep.grid(row=0, column=4, sticky="ns")

        # setup bottom right
        self.clear_bse_points = ttk.Button(self.bot_right, text="Clear points", command=lambda: self.clear_points("bse"), state="disabled")
        self.clear_bse_points.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.clahe_active = False
        self.clahe_b = ttk.Button(self.bot_right, text="Apply CLAHE", command=self.clahe, state="disabled")
        self.clahe_b.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        sep = ttk.Separator(self.bot_right, orient=tk.VERTICAL)
        sep.grid(row=0, column=2, sticky="ns")

        # Setup resizing
        self.resize_options = [25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500]
        self.resize_var_ebsd = tk.StringVar()
        self.resize_var_bse = tk.StringVar()
        self.resize_var_ebsd.set(self.resize_options[3])
        self.resize_var_bse.set(self.resize_options[3])
        self.resize_var_ebsd.trace("w", lambda *args: self._resize("ebsd", self.resize_vars["ebsd"].get()))
        self.resize_var_bse.trace("w", lambda *args: self._resize("bse", self.resize_vars["bse"].get()))
        ebsd_resize_label = ttk.Label(self.bot_left, text="Zoom:")
        bse_resize_label = ttk.Label(self.bot_right, text="Zoom:")
        ebsd_resize_label.grid(row=0, column=5, sticky="e", padx=5, pady=5)
        bse_resize_label.grid(row=0, column=3, sticky="e", padx=5, pady=5)
        self.ebsd_resize_dropdown = ttk.Combobox(self.bot_left, textvariable=self.resize_var_ebsd, values=self.resize_options, state="readonly", width=5)
        self.bse_resize_dropdown = ttk.Combobox(self.bot_right, textvariable=self.resize_var_bse, values=self.resize_options, state="readonly", width=5)
        self.ebsd_resize_dropdown["state"] = "disabled"
        self.bse_resize_dropdown["state"] = "disabled"
        self.ebsd_resize_dropdown.grid(row=0, column=6, sticky="ew")
        self.bse_resize_dropdown.grid(row=0, column=4, sticky="ew")
        self.resize_vars = {"ebsd": self.resize_var_ebsd, "bse": self.resize_var_bse}

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
            self.ebsd_path, self.bse_path = ebsd_path, bse_path
            self.ebsd_data, self.bse_imgs = e_d, b_d
            self.ebsd_res, self.bse_res = ebsd_res, bse_res
            self.points["ebsd"], self.points["bse"] = e_pts, b_pts
            # Set the UI stuff
            self.ebsd_mode_options = list(self.ebsd_data.keys())
            self.ebsd_mode.set(self.ebsd_mode_options[0])
            self.slice_min = 0
            self.slice_max = self.ebsd_data[self.ebsd_mode_options[0]].shape[0] - 1
            self.slice_num.set(self.slice_min)
            # self.ebsd_cStack = np.zeros(self.ebsd_data[self.ebsd_mode_options[0]].shape)
            # Configure UI
            self.slice_picker["state"] = "readonly"
            self.slice_picker["values"] = list(np.arange(self.slice_min, self.slice_max + 1))
            self.ebsd_picker["state"] = "readonly"
            self.ebsd_picker["values"] = self.ebsd_mode_options
            # Finish
            self.folder = os.path.dirname(ebsd_path)
            self._update_viewers()
            self.menu.entryconfig("Apply", state="normal")
            self.clear_ebsd_points["state"] = "normal"
            self.clear_bse_points["state"] = "normal"
            self.ebsd_resize_dropdown["state"] = "readonly"
            self.bse_resize_dropdown["state"] = "readonly"
            self.clahe_b["state"] = "normal"
            self.ex_ctr_pt_ims["state"] = "normal"
            self.view_pts["state"] = "normal"
            self.show_points.set(1)
    
    ### Coords stuff
    def add_coords(self, pos, event):
        """Responds to a click on an image. Redraws the images after the click. Also saves the click location in a file."""
        i = self.slice_num.get()
        scale = int(self.resize_vars[pos].get()) / 100
        if pos == "ebsd":
            x = int(self.ebsd.canvasx(event.x))
            y = int(self.ebsd.canvasy(event.y))
        else:
            x = int(self.bse.canvasx(event.x))
            y = int(self.bse.canvasy(event.y))
        self.points[pos][i].append([int(np.around(x / scale)), int(np.around(y / scale))])
        path = self.points_path[pos]
        with open(path, "a", encoding="utf8") as output:
            output.write(f"{i} {event.x} {event.y}\n")
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
        
    def clear_points(self, mode):
        if mode == "ebsd":
            self.ebsd.delete("all")
            self.points["ebsd"][self.slice_num.get()] = []
        elif mode == "bse":
            self.bse.delete("all")
            self.points["bse"][self.slice_num.get()] = []
        self._update_viewers()
    
    def remove_coords(self, pos, event):
        """Remove the point closes to the clicked location, the point should be removed from both images"""
        if pos == 'bse': v = self.bse
        elif pos == 'ebsd': v = self.ebsd
        closest = v.find_closest(v.canvasx(event.x), v.canvasy(event.y))[0]
        tag = v.itemcget(closest, "tags")
        tag = tag.replace("current", "").replace("text", "").replace("bbox", "").strip()
        if tag == "":
            return
        self.points[pos][self.slice_num.get()].pop(int(tag))
        path = self.points_path[pos]
        with open(path, "w", encoding="utf8") as output:
            for z in self.points[pos].keys():
                for i in range(len(self.points[pos][z])):
                    if i == int(tag):
                        continue
                    x, y = self.points[pos][z][i]
                    output.write(f"{int(z)} {int(x)} {int(y)}\n")
        self._update_viewers()

    def _show_points(self):
        """Either turns on or turns off control point viewing"""
        if self.show_points.get() == 1:
            j = self.slice_num.get()
            pc = {"ebsd": "#FEBC11", "bse": "#EF5645"}
            viewers = {"ebsd": self.ebsd, "bse": self.bse}
            for mode in ["ebsd", "bse"]:
                scale = int(self.resize_vars[mode].get()) / 100
                pts = np.around(np.array(self.points[mode][j]) * scale).astype(int)
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

    ### Apply stuff for visualizing
    def apply(self, algo="TPS"):
        """Applies the correction algorithm and calls the interactive view"""
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
        ebsd_im = self.ebsd_data[self.ebsd_mode.get()][i, :, :]
        im1 = align.apply(ebsd_im, out="array")
        im0 = self.bse_imgs[i, :, :]
        print("Creating interactive view")
        IV.Interactive2D(im0, im1, "2D TPS Correction".format(i))
        # self._interactive_view(algo, im1)
        plt.close("all")

    def apply_3D(self, algo="LR"):
        """Applies the correction algorithm and calls the interactive view"""
        self.config(cursor="watch")
        points = self.points
        ebsd_stack = np.sqrt(np.sum(self.ebsd_data[self.ebsd_mode.get()][...], axis=3))
        referencePoints = np.array(self.points["bse"][self.slice_num.get()])
        distortedPoints = np.array(self.points["ebsd"][self.slice_num.get()])
        print("Aligning the full ESBSD stack in mode {}".format(self.ebsd_mode.get()))
        align = core.Alignment(referencePoints, distortedPoints, algorithm=algo)
        ebsd_cStack = align.TPS_apply_3D(points, ebsd_stack, self.bse_imgs)
        print("Creating interactive view")
        self.config(cursor="tcross")
        IV.Interactive3D(self.bse_imgs, ebsd_cStack, "3D TPS Correction")
        # self._interactive_view(algo, self.ebsd_cStack, True)
        plt.close("all")

    ### Apply stuff for exporting 

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

    ### Other functions for buttons on the UI
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

    ### Viewer stuff
    def _resize(self, pos, scale):
        """Resizes the image in the viewer"""
        print(f"Resizing {pos} to {scale}%")
        if pos == "ebsd":
            self.ebsd.delete("all")
        else:
            self.bse.delete("all")
        self._update_viewers()

    def _update_viewers(self, *args):
        i = self.slice_num.get()
        key = self.ebsd_mode.get()
        print(f"Updating viewers for slice {i} and mode {key}")
        bse_im = self.bse_imgs[i]
        ebsd_im = self.ebsd_data[key][i]
        if self.clahe_active: bse_im = exposure.equalize_adapthist(bse_im, clip_limit=0.03)
        # Check if there are 3 dimensions in which the last one is not needed
        if len(ebsd_im.shape) == 3 and ebsd_im.shape[-1] == 1: ebsd_im = ebsd_im[:, :, 0]
        if len(bse_im.shape) == 3 and bse_im.shape[-1] == 1:   bse_im = bse_im[:, :, 0]
        # Check if there are 4 dimensions in which we just take the sum of the squares               
        if len(ebsd_im.shape) == 4: ebsd_im = np.sum(ebsd_im ** 2, axis=-1)
        # Resize the images
        scale_ebsd = int(self.resize_vars["ebsd"].get()) / 100
        scale_bse = int(self.resize_vars["bse"].get()) / 100
        if scale_ebsd != 1: ebsd_im = transform.resize(ebsd_im, (int(ebsd_im.shape[0] * scale_ebsd), int(ebsd_im.shape[1] * scale_ebsd)), anti_aliasing=True)
        if scale_bse != 1: bse_im = transform.resize(bse_im, (int(bse_im.shape[0] * scale_bse), int(bse_im.shape[1] * scale_bse)), anti_aliasing=True)
        # Ensure dtype is uint8, accounting for RGB or greyscale
        if ebsd_im.ndim == 3 and ebsd_im.dtype != np.uint8:   self.ebsd_im = np.around(255 * (ebsd_im - ebsd_im.min(axis=(0, 1))) / (ebsd_im.max(axis=(0, 1)) - ebsd_im.min(axis=(0, 1))), 0).astype(np.uint8)
        elif ebsd_im.ndim == 2 and ebsd_im.dtype != np.uint8: self.ebsd_im = np.around(255 * (ebsd_im - ebsd_im.min()) / (ebsd_im.max() - ebsd_im.min()), 0).astype(np.uint8)
        else: self.ebsd_im = ebsd_im
        if bse_im.ndim == 3 and bse_im.dtype != np.uint8:     self.bse_im = np.around(255 * (bse_im - bse_im.min(axis=(0, 1))) / (bse_im.max(axis=(0, 1)) - bse_im.min(axis=(0, 1))), 0).astype(np.uint8)
        elif bse_im.ndim == 2 and bse_im.dtype != np.uint8:   self.bse_im = np.around(255 * (bse_im - bse_im.min()) / (bse_im.max() - bse_im.min()), 0).astype(np.uint8)
        else: self.bse_im = bse_im
        # Update the images and draw points
        self.bse.delete("all")
        self.ebsd.delete("all")
        self._update_imgs()
        self._show_points()

    def _update_imgs(self):
        """Updates the images in the viewers"""
        self.ebsd.delete("all")
        self.bse.delete("all")
        # BSE
        # self.bse["width"] = self.bse_im.shape[1]
        # self.bse["height"] = self.bse_im.shape[0]
        self.bse_im_ppm = self._photo_image(self.bse_im, channels=1)
        self.bse.create_image(0, 0, anchor="nw", image=self.bse_im_ppm)
        self.bse.config(scrollregion=self.bse.bbox("all"))
        # EBSD
        # self.ebsd["width"] = self.ebsd_im.shape[1]
        # self.ebsd["height"] = self.ebsd_im.shape[0]
        channels = 3 if self.ebsd_im.ndim == 3 else 1
        self.ebsd_im_ppm = self._photo_image(self.ebsd_im, channels=channels)
        self.ebsd.create_image(0, 0, anchor="nw", image=self.ebsd_im_ppm)
        self.ebsd.config(scrollregion=self.ebsd.bbox("all"))

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
