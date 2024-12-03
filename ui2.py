"""
Author: James Lamb

UI for running distortion correction
"""

# Python packages
import os
from pathlib import Path
# from threading import Thread
from multiprocessing.pool import ThreadPool
import shutil
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from copy import deepcopy

# 3rd party packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, transform
import torch
from torchvision.transforms.functional import resize as RESIZE
from torchvision.transforms import InterpolationMode
from kornia.enhance import equalize_clahe

# Local files
import warping
import core
import Inputs
import InteractiveView as IV


def CLAHE(im, clip_limit=20.0, kernel_size=(8, 8)):
    tensor = torch.tensor(im).unsqueeze(0).unsqueeze(0).float()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = equalize_clahe(tensor, clip_limit, kernel_size)
    tensor = torch.round(255 * (tensor - tensor.min()) / (tensor.max() - tensor.min()),
                         decimals=0)
    return np.squeeze(tensor.detach().numpy().astype(np.uint8)).reshape(im.shape)


class App(tk.Tk):
    def __init__(self, screenName=None, baseName=None):
        super().__init__(screenName, baseName)
        self._style_call("dark")
        #
        # Starting stuff
        self.update_idletasks()
        self.withdraw()
        self.folder = os.getcwd()
        self.deiconify()
        self.title("Distortion Correction")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        # self.geometry(f"{int(screen_width*2/3)}x{int(screen_height*2/3)}")
        self.resizable(False, False)
        #
        # Frames
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
        filemenu.add_command(label="Open 2D", command=lambda: self.select_data_popup("2D"))
        filemenu.add_command(label="Export 2D", command=lambda: self.apply_correction_to_tif("TPS"))
        filemenu.add_command(label="Quick start", command=self.easy_start)
        self.menu.add_cascade(label="File", menu=filemenu)
        applymenu = tk.Menu(self.menu, tearoff=0)
        applymenu.add_command(label="TPS", command=lambda: self.apply("TPS"))
        self.menu.add_cascade(label="View", menu=applymenu, state="disabled")
        self.config(menu=self.menu)
        #
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
        #
        # setup viewer_left
        l = ttk.Label(self.viewer_left, text="EBSD/Distorted image", anchor=tk.CENTER)
        l.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.ebsd = tk.Canvas(self.viewer_left, highlightbackground=self.bg, bg=self.bg, bd=1, highlightthickness=0.2, cursor='tcross', width=int(screen_width*.45), height=int(screen_height*.7))
        self.ebsd.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.ebsd_hscroll = ttk.Scrollbar(self.viewer_left, orient=tk.HORIZONTAL, command=self.ebsd.xview, cursor="sb_h_double_arrow")
        self.ebsd_hscroll.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.ebsd_vscroll = ttk.Scrollbar(self.viewer_left, orient=tk.VERTICAL, command=self.ebsd.yview, cursor="sb_v_double_arrow")
        self.ebsd_vscroll.grid(row=1, column=1, sticky="ns", padx=5, pady=5)
        self.ebsd.config(xscrollcommand=self.ebsd_hscroll.set, yscrollcommand=self.ebsd_vscroll.set)
        #
        # setup viewer right
        l = ttk.Label(self.viewer_right, text="BSE/Control image", anchor=tk.CENTER)
        l.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.bse = tk.Canvas(self.viewer_right, highlightbackground=self.bg, bg=self.bg, bd=1, highlightthickness=0.2, cursor='tcross', width=int(screen_width*.45), height=int(screen_height*.7))
        self.bse.grid(row=1, column=0, pady=5, padx=5, sticky="nsew")
        self.bse_hscroll = ttk.Scrollbar(self.viewer_right, orient=tk.HORIZONTAL, command=self.bse.xview, cursor="sb_h_double_arrow")
        self.bse_hscroll.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.bse_vscroll = ttk.Scrollbar(self.viewer_right, orient=tk.VERTICAL, command=self.bse.yview, cursor="sb_v_double_arrow")
        self.bse_vscroll.grid(row=1, column=1, sticky="ns", padx=5, pady=5)
        self.bse.config(xscrollcommand=self.bse_hscroll.set, yscrollcommand=self.bse_vscroll.set)
        #
        # Bindings on viewers
        if os.name == 'posix':
            self.ebsd.bind("<Button 2>", lambda arg: self.remove_coords("ebsd", arg))
            self.ebsd.bind('<MouseWheel>', lambda event: self.ebsd.yview_scroll(int(-1*(event.delta)), "units"))
            self.ebsd.bind('<Shift-MouseWheel>', lambda event: self.ebsd.xview_scroll(int(-1*(event.delta)), "units"))
            self.bse.bind("<Button 2>", lambda arg: self.remove_coords("bse", arg))
            self.bse.bind('<MouseWheel>', lambda event: self.bse.yview_scroll(int(-1*(event.delta)), "units"))
            self.bse.bind('<Shift-MouseWheel>', lambda event: self.bse.xview_scroll(int(-1*(event.delta)), "units"))
        else:
            self.ebsd.bind("<Button 3>", lambda arg: self.remove_coords("ebsd", arg))
            self.ebsd.bind("<MouseWheel>", lambda event: self.ebsd.yview_scroll(int(-1*(event.delta/120)), "units"))
            self.ebsd.bind("<Shift-MouseWheel>", lambda event: self.ebsd.xview_scroll(int(-1*(event.delta/120)), "units"))
            self.bse.bind("<Button 3>", lambda arg: self.remove_coords("bse", arg))
            self.bse.bind("<MouseWheel>", lambda event: self.bse.yview_scroll(int(-1*(event.delta/120)), "units"))
            self.bse.bind("<Shift-MouseWheel>", lambda event: self.bse.xview_scroll(int(-1*(event.delta/120)), "units"))
        self.ebsd.bind("<ButtonPress-1>", lambda arg: self.add_coords("ebsd", arg))
        self.bse.bind("<ButtonPress-1>", lambda arg: self.add_coords("bse", arg))
        self.bind('<Control-equal>', lambda event: self.change_zoom_event(+1), "units")
        self.bind('<Control-minus>', lambda event: self.change_zoom_event(-1), "units")
        # self.ebsd.bind('<Control-equal>', lambda event: self.change_zoom_event(+1, "ebsd"), "units")
        # self.ebsd.bind('<Control-minus>', lambda event: self.change_zoom_event(-1, "ebsd"), "units")
        # self.bse.bind('<Control-equal>', lambda event: self.change_zoom_event(+1, "bse"), "units")
        # self.bse.bind('<Control-minus>', lambda event: self.change_zoom_event(-1, "bse"), "units")
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
        self.resize_options = [1, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 600]
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

        ### Additional things added
        self.points = {"ebsd": {}, "bse": {}}
        self.points_path = {"ebsd": "", "bse": ""}
    
    def _thread_function(self, function, *args):
        output = function(*args)
        self.event_generate("<<Foo>>", when="tail")
        return output

    def _run_in_background(self, text, function, *args):
        self.w = ProgressWindow(text)
        self.bind("<<Foo>>", lambda arg: self.w.destroy())
        thread_obj = ThreadPool(processes=1)
        output = thread_obj.apply_async(self._thread_function, (function, *args))
        self.wait_window(self.w)
        return output.get()

    def change_zoom_event(self, change):
        print("Changing zoom !", change)
        index = self.resize_options.index(int(self.resize_var_bse.get()))
        index = max(0, min(len(self.resize_options) - 1, index + change))
        self.resize_var_bse.set(self.resize_options[index])
        index = self.resize_options.index(int(self.resize_var_ebsd.get()))
        index = max(0, min(len(self.resize_options) - 1, index + change))
        self.resize_var_ebsd.set(self.resize_options[index])

    ### IO ###
    def easy_start(self):
        ebsd_path = "./test_data/EBSD.ang"
        ebsd_points_path = "./test_data/distorted_pts.txt"
        bse_path = "./test_data/BSE.tif"
        bse_points_path = "./test_data/control_pts.txt"
        # ebsd_points_path = "D:/Research/scripts/Alignment/CoNi67/distorted_pts.txt"
        # bse_points_path = "D:/Research/scripts/Alignment/CoNi67/control_pts.txt"
        # ebsd_path = "D:/Research/scripts/Alignment/CoNi67/CoNi67_aligned.dream3d"
        # bse_path = "D:/Research/scripts/Alignment/CoNi67/se_images_aligned/*.tif"
        ebsd_res = 2.5
        bse_res = 1.0
        rescale = True
        r180 = False
        flip = False
        crop = False
        self.ang_path = ebsd_path
        self.ebsd_res = ebsd_res
        self.bse_res = bse_res
        e_d, b_d, e_pts, b_pts = self._run_in_background("Importing data...", Inputs.read_data, ebsd_path, bse_path, ebsd_points_path, bse_points_path)
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

        if rescale:
            b_d = Inputs.rescale_control(b_d, bse_res, ebsd_res)
        # Flip or rotate 180 if desired
        if flip:
            b_d = np.flip(b_d, axis=1).copy(order='C')
        elif r180:
            b_d = np.rot90(b_d, 2, axes=(1,2)).copy(order='C')
        # Crop if desired
        if crop:
            ### TODO: Add multiple control images
            self.w = Inputs.CropControl(self, b_d[0, :, :])
            self.wait_window(self.w.w)
            if self.w.clean_exit:
                s, e = self.w.start, self.w.end
                ### TODO: Add multiple control images
                b_d = b_d[:, s[0]:e[0], s[1]:e[1]]
        self._handle_input(ebsd_path, bse_path, e_d, b_d, ebsd_points_path, bse_points_path, e_pts, b_pts, ebsd_res, bse_res)

    def select_data_popup(self, mode):
        self.w = Inputs.DataInput(self, mode)
        self.wait_window(self.w.w)
        if self.w.clean_exit:
            if self.w.ang:
                self.ang_path = self.w.ebsd_path
            ebsd_path, bse_path = self.w.ebsd_path, self.w.bse_path
            ebsd_points_path, bse_points_path = self.w.ebsd_points_path, self.w.bse_points_path
            ebsd_res, bse_res = self.w.ebsd_res, self.w.bse_res
            e_d, b_d, e_pts, b_pts = Inputs.read_data(ebsd_path, bse_path, ebsd_points_path, bse_points_path)
            self.w = Inputs.DataSummary(self, e_d, b_d, e_pts, b_pts, ebsd_res, bse_res)
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
                # Rescale if desired
                if self.w.rescale:
                    b_d = Inputs.rescale_control(b_d, bse_res, ebsd_res)
                # Flip or rotate 180 if desired
                if self.w.flip:
                    b_d = np.flip(b_d, axis=1).copy(order='C')
                elif self.w.r180:
                    b_d = np.rot90(b_d, 2, axes=(1,2)).copy(order='C')
                # Crop if desired
                if self.w.crop:
                    ### TODO: Add multiple control images
                    self.w = Inputs.CropControl(self, b_d[0, :, :])
                    self.wait_window(self.w.w)
                    if self.w.clean_exit:
                        s, e = self.w.start, self.w.end
                        ### TODO: Add multiple control images
                        b_d = b_d[:, s[0]:e[0], s[1]:e[1]]
                self._handle_input(ebsd_path, bse_path, e_d, b_d, ebsd_points_path, bse_points_path, e_pts, b_pts, ebsd_res, bse_res)
            else:
                return
        
    def _handle_input(self, ebsd_path, bse_path, e_d, b_d, ebsd_points_path, bse_points_path, e_pts, b_pts, ebsd_res, bse_res):
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
        self.menu.entryconfig("View", state="normal")
        self.clear_ebsd_points["state"] = "normal"
        self.clear_bse_points["state"] = "normal"
        self.ebsd_resize_dropdown["state"] = "readonly"
        self.bse_resize_dropdown["state"] = "readonly"
        self.clahe_b["state"] = "normal"
        self.view_pts["state"] = "normal"
        self.show_points.set(1)

    ### Coords stuff ###
    def add_coords(self, pos, event):
        """Responds to a click on an image. Redraws the images after the click. Also saves the click location in a file."""
        i = self.slice_num.get()
        if pos == "ebsd":
            scale = int(self.resize_vars["ebsd"].get()) / 100
            x = int(np.around(self.ebsd.canvasx(event.x), 0))
            y = int(np.around(self.ebsd.canvasy(event.y), 0))
            x = int(np.around(x / scale, 0))
            y = int(np.around(y / scale, 0))
        else:
            scale = int(self.resize_vars["bse"].get()) / 100
            x = int(np.around(self.bse.canvasx(event.x), 0))
            y = int(np.around(self.bse.canvasy(event.y), 0))
            x = int(np.around(x / scale, 0))
            y = int(np.around(y / scale, 0))
        if i not in self.points[pos].keys():
            self.points[pos][i] = []
        try:
            self.points[pos][i] = np.append(self.points[pos][i], [[x, y]], axis=0)
        except ValueError:
            self.points[pos][i] = np.array([[x, y]])
        self.write_coords()
        # path = self.points_path[pos]
        # with open(path, "a", encoding="utf8") as output:
        #     output.write(f"{i} {x} {y}\n")
        self._show_points()

    def write_coords(self):
        for mode in ["ebsd", "bse"]:
            path = self.points_path[mode]
            data = []
            pts = self.points[mode]
            for key in pts.keys():
                pts_temp = np.array(pts[key])
                if pts_temp.size == 0:
                    continue
                # Handle if there is only one point and give a z value for each point
                if pts_temp.ndim == 1:
                    s = np.insert(pts_temp, 0, key).reshape(1, 3)
                else:
                    s = np.hstack((np.ones((pts_temp.shape[0], 1)) * key, pts_temp))
                data.append(s)
            # Handle if there is one slice or if there are multiple slices with points
            if len(data) == 0:
                continue
            elif len(data) > 1:
                data = np.vstack(data)
            else:
                data = np.array(data[0])
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
        self.points[pos][self.slice_num.get()] = np.delete(self.points[pos][self.slice_num.get()], int(tag), axis=0)
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
                try:
                    pts = np.around(np.array(self.points[mode][j]) * scale).astype(int)
                except KeyError:
                    continue
                if pts.ndim == 1:
                    if pts.size == 0:
                        continue
                    pts = pts.reshape((1, 2))
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

    ### Apply stuff for visualizing ###
    def _get_cropping_slice(self, centroid, target_shape, current_shape):
        """Returns a slice object that can be used to crop an image"""
        # print("Target shape: {}".format(target_shape))
        # print("Current shape: {}".format(current_shape))
        rstart = centroid[0] - target_shape[0] // 2
        rend = rstart + target_shape[0]
        # print("Row raw:", rstart, rend)
        if rstart < 0:
            r_slice = slice(0, target_shape[0])
        elif rend > current_shape[0]:
            r_slice = slice(current_shape[0] - target_shape[0], current_shape[0])
        else:
            r_slice = slice(rstart, rend)
        # print("Row slice:", r_slice)

        cstart = centroid[1] - target_shape[1] // 2
        cend = cstart + target_shape[1]
        # print("Col raw:", cstart, cend)
        if cstart < 0:
            c_slice = slice(0, target_shape[1])
        elif cend > current_shape[1]:
            c_slice = slice(current_shape[1] - target_shape[1], current_shape[1])
        else:
            c_slice = slice(cstart, cend)
        # print("Col slice:", c_slice)
        return r_slice, c_slice

    def _check_sizes(self, im1, im2, ndims=2):
        """im1: ebsd (distorted), im2: ebsd (corrected)"""
        if ndims == 2:
            if im1.shape[0] > im2.shape[0]:
                im2_temp = np.zeros((im1.shape[0], im2.shape[1]))
                im2_temp[:im2.shape[0], :] = im2
                im2 = im2_temp
            if im1.shape[1] > im2.shape[1]:
                im2_temp = np.zeros((im2.shape[0], im1.shape[1]))
                im2_temp[:, :im2.shape[1]] = im2
                im2 = im2_temp
            return im2
        else:
            if im1.shape[1] > im2.shape[1]:
                im2_temp = np.zeros((im2.shape[0], im1.shape[1], im2.shape[2]))
                im2_temp[:, :im2.shape[1], :] = im2
                im2 = im2_temp
            if im1.shape[2] > im2.shape[2]:
                im2_temp = np.zeros((im2.shape[0], im2.shape[1], im1.shape[2]))
                im2_temp[:, :, :im2.shape[2]] = im2
                im2 = im2_temp
            return im2
    
    def apply(self, algo="TPS"):
        """Applies the correction algorithm and calls the interactive view"""
        result = tk.messagebox.askyesnocancel("Crop?", "Would you like to crop the corrected output to match the distorted grid?")
        if result is None:
            return
        
        im0, im1 = self._run_in_background("Applying correction...", self._apply, algo, result)
        if (im0.shape[0] > 2000) or (im0.shape[1] > 2000):
            result = tk.messagebox.askyesnocancel("Resize?", "The images are large, would you like to resize for the preview? (Recommended)")
            if result is None:
                pass
            if result:
                scale = int(max(im0.shape) / 2000)
                im0 = im0[::scale, ::scale]
                im1 = im1[::scale, ::scale]
        # View
        print("Creating interactive view")
        IV.Interactive2D(im0, im1, "2D TPS Correction")
        plt.close("all")

    def _apply(self, algo, crop_to_distorted_grid):
        # Get the bse and ebsd images
        im0 = self.bse_imgs[int(self.slice_num.get())]
        i = self.slice_num.get()
        ebsd_im = self.ebsd_data[self.ebsd_mode.get()][i, :, :]

        # Process
        if self.clahe_active:
            im0 = CLAHE(im0)
        if im0.shape[0] < ebsd_im.shape[0]:
            im0 = np.pad(im0, ((0, ebsd_im.shape[0] - im0.shape[0]), (0, 0)), mode="constant", constant_values=0)
        if im0.shape[1] < ebsd_im.shape[1]:
            im0 = np.pad(im0, ((0, 0), (0, ebsd_im.shape[1] - im0.shape[1])), mode="constant", constant_values=0)

        # Get the points and performa alignment
        referencePoints = np.array(self.points["bse"][i])
        distortedPoints = np.array(self.points["ebsd"][i])
        im1 = warping.transform_image(im0, distortedPoints, referencePoints, mode="tps", size=im0.shape)

        # Crop to the center of the corrected image if desired
        if crop_to_distorted_grid:
            # Do this by correcting an empty image (all ones) and finding the centroid of the corrected image
            dummy = np.ones(ebsd_im.shape)
            dummy = warping.transform_image(dummy, distortedPoints, referencePoints, mode="tps", size=im0.shape)
            rc, cc = np.array(np.where(dummy)).reshape(2, -1).T.mean(axis=0).astype(int)
            rslc, cslc = self._get_cropping_slice((rc, cc), ebsd_im.shape, im1.shape)
            im0 = im0[rslc, cslc]
            im1 = im1[rslc, cslc]
            print("Centroid:", rc, cc)
            print("Aligned/Control data cropped from {} to {} (to match distorted grid size)".format(im1.shape, im1[rslc, cslc].shape))
            print("Details", rslc, cslc, im1.shape, im0.shape, ebsd_im.shape)

        return im0, im1

    def apply_correction_to_tif(self, algo):
        # Get the control points
        i = self.slice_num.get()
        referencePoints = np.array(self.points["bse"][i])
        distortedPoints = np.array(self.points["ebsd"][i])
        align = core.Alignment(referencePoints, distortedPoints, algorithm=algo)
        # Get BSE
        bse_im = self.bse_imgs[int(self.slice_num.get())]
        if self.clahe_active:
            # bse_im = exposure.equalize_adapthist(self.bse_imgs[int(self.slice_num.get())], clip_limit=0.03)
            bse_im = CLAHE(bse_im)
        # Create the output filename
        SAVE_PATH_EBSD = filedialog.asksaveasfilename(initialdir=os.path.basename(self.ebsd_path), title="Save corrected (from distorted) image", filetypes=[("TIF", "*.tif"), ("TIFF", "*.tiff"), ("ANG", "*.ang"), ("All files", "*.*")], defaultextension=".tiff")
        extension = os.path.splitext(SAVE_PATH_EBSD)[1]
        if SAVE_PATH_EBSD != "":
            if "." not in SAVE_PATH_EBSD:
                raise ValueError("No extension provided!")
            result = tk.messagebox.askyesnocancel("Crop?", "Would you like to crop the output to match the distorted grid (usually yes)?")
            if result is None:
                return
            if extension != ".ang":
                # Get the images
                ebsd_im = np.squeeze(self.ebsd_data[self.ebsd_mode.get()][int(self.slice_num.get())])
                print("EBSD image grabbed")
                # Align the image
                align.get_solution(size=bse_im.shape)
                print("Alignment solution found")
                if len(ebsd_im.shape) == 3:
                    aligned = []
                    for i in range(ebsd_im.shape[-1]):
                        aligned.append(align.apply(ebsd_im[:, :, i], out="array"))
                    aligned = np.moveaxis(np.array(aligned), 0, -1)
                else:
                    aligned = align.apply(ebsd_im, out="array")
                print("EBSD image aligned")
                # Correct dtype
                if aligned.dtype != ebsd_im.dtype:
                    aligned = core.handle_dtype(aligned, ebsd_im.dtype)
                    print("EBSD image dtype corrected")
                # Correct shape
                aligned = self._check_sizes(ebsd_im, aligned)
                bse_im = self._check_sizes(ebsd_im, bse_im)
                print("EBSD image shape corrected")
                if result:
                    # Do this by correcting an empty image (all ones) and finding the centroid of the corrected image
                    dummy = np.ones(ebsd_im.shape)
                    rc, cc = self._get_corrected_centroid(dummy, align)
                    # Now crop the corrected image
                    rslc, cslc = self._get_cropping_slice((rc, cc), ebsd_im.shape, aligned.shape)
                    aligned = aligned[rslc, cslc]
                    bse_im = bse_im[rslc, cslc]
                    print("EBSD image cropped")
                # Save the image
                io.imsave(SAVE_PATH_EBSD, aligned)
                print("EBSD image saved")
            else:
                data = deepcopy(self.ebsd_data)
                del data["EulerAngles"]
                columns = list(data.keys())
                data_stack = np.zeros((len(columns), *data["x"][0].shape))
                print("EBSD entries:", columns)
                print("EBSD grid size:", data_stack.shape)
                for i, key in enumerate(columns):
                    # Get the images
                    ebsd_im = np.squeeze(data[key][int(self.slice_num.get())])
                    # Align the image
                    align.get_solution(size=bse_im.shape)
                    if len(ebsd_im.shape) == 3:
                        aligned = []
                        for i in range(ebsd_im.shape[-1]):
                            aligned.append(align.apply(ebsd_im[:, :, i], out="array"))
                        aligned = np.moveaxis(np.array(aligned), 0, -1)
                    else:
                        aligned = align.apply(ebsd_im, out="array")
                    if key.lower() in ["phi1", "phi", "phi2"]:
                        aligned[aligned == 0] = 4*float(np.pi)
                    elif key.lower() == "ci":
                        aligned[aligned == 0] = -1
                    elif key.lower() == "fit":
                        aligned[aligned == 0] = 180
                    # Correct dtype
                    aligned = core.handle_dtype(np.around(aligned, 5), ebsd_im.dtype)
                    # Correct shape
                    aligned = self._check_sizes(ebsd_im, aligned)
                    bse_im = self._check_sizes(ebsd_im, bse_im)
                    # Do this by correcting an empty image (all ones) and finding the centroid of the corrected image
                    dummy = np.ones(ebsd_im.shape)
                    rc, cc = self._get_corrected_centroid(dummy, align)
                    # Now crop the corrected image
                    rslc, cslc = self._get_cropping_slice((rc, cc), ebsd_im.shape, aligned.shape)
                    aligned = aligned[rslc, cslc]
                    data_stack[i] = aligned
                d = data_stack.reshape(data_stack.shape[0], -1).T
                x_index = columns.index("x")
                y_index = columns.index("y")
                y, x = np.indices(data["x"][0].shape)
                x = x.ravel() * self.ebsd_res
                y = y.ravel() * self.ebsd_res
                d[:, x_index] = x
                d[:, y_index] = y
                # Get the header
                with open(self.ang_path, "r") as f:
                    header = []
                    for line in f.readlines():
                        if line.startswith("#"):
                            header.append(line)
                        else:
                            break
                header = "".join(header)
                # Save the data
                with open(SAVE_PATH_EBSD, "w") as f:
                    f.write(header)
                    for i in range(d.shape[0]):
                        fmts = ["%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.1f", "%.3f", "%.0f", "%.0f", "%.3f", "%.6f", "%.6f", "%.6f"]
                        space = [3, 4, 4, 7, 7, 7, 3, 3, 7, 4, 7, 7, 7]
                        line = [" "*(space[j]-len(str(int(d[i,j])))) + fmts[j] % (d[i,j]+0.0) for j in range(d.shape[1])]
                        line = "".join(line)
                        f.write(f" {line}\n")
            # Save the control image (if desired)
            SAVE_PATH_BSE = filedialog.asksaveasfilename(initialdir=os.path.basename(self.ebsd_path), title="Save control image", filetypes=[("TIF", "*.tif"), ("TIFF", "*.tiff"), ("All files", "*.*")], defaultextension=extension)
            if SAVE_PATH_BSE != "":
                if "." not in SAVE_PATH_BSE:
                    SAVE_PATH_BSE += extension
                io.imsave(SAVE_PATH_BSE, bse_im)
            print("Correction complete!")

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
        # print(f"Resizing {pos} to {scale}%")
        if pos == "ebsd":
            self.ebsd.delete("all")
        else:
            self.bse.delete("all")
        self._update_viewers()

    def _update_viewers(self, *args):
        i = self.slice_num.get()
        key = self.ebsd_mode.get()
        # print(f"Updating viewers for slice {i} and mode {key}")
        bse_im = self.bse_imgs[i]
        ebsd_im = self.ebsd_data[key][i]
        if self.clahe_active:
            # bse_im = exposure.equalize_adapthist(bse_im, clip_limit=0.03)
            bse_im = CLAHE(bse_im)
        # Check if there are 3 dimensions in which the last one is not needed
        if len(ebsd_im.shape) == 3 and ebsd_im.shape[-1] == 1: ebsd_im = ebsd_im[:, :, 0]
        if len(bse_im.shape) == 3 and bse_im.shape[-1] == 1:   bse_im = bse_im[:, :, 0]
        # Check if there are 4 dimensions in which we just take the sum of the squares               
        if len(ebsd_im.shape) == 4: ebsd_im = np.sum(ebsd_im ** 2, axis=-1)
        # Resize the images
        scale_ebsd = int(self.resize_vars["ebsd"].get()) / 100
        scale_bse = int(self.resize_vars["bse"].get()) / 100
        if scale_ebsd != 1:
            _t = torch.tensor(ebsd_im).unsqueeze(0).unsqueeze(0).float()
            _out = RESIZE(_t, (int(ebsd_im.shape[0] * scale_ebsd), int(ebsd_im.shape[1] * scale_ebsd)), InterpolationMode.NEAREST)
            ebsd_im = _out.detach().squeeze().numpy()
        if scale_bse != 1:
            _t = torch.tensor(bse_im).unsqueeze(0).unsqueeze(0).float()
            _out = RESIZE(_t, (int(bse_im.shape[0] * scale_bse), int(bse_im.shape[1] * scale_bse)), InterpolationMode.NEAREST)
            bse_im = _out.detach().squeeze().numpy()
        # if scale_ebsd != 1: ebsd_im = transform.resize(ebsd_im, (int(ebsd_im.shape[0] * scale_ebsd), int(ebsd_im.shape[1] * scale_ebsd)), anti_aliasing=False)
        # if scale_bse != 1: bse_im = transform.resize(bse_im, (int(bse_im.shape[0] * scale_bse), int(bse_im.shape[1] * scale_bse)), anti_aliasing=False)
        # Ensure dtype is uint8, accounting for RGB or greyscale
        if ebsd_im.ndim == 3 and ebsd_im.dtype != np.uint8:
            self.ebsd_im = np.around(255 * (ebsd_im - ebsd_im.min(axis=(0, 1))) / (ebsd_im.max(axis=(0, 1)) - ebsd_im.min(axis=(0, 1))), 0).astype(np.uint8)
        elif ebsd_im.ndim == 2 and ebsd_im.dtype != np.uint8:
            self.ebsd_im = np.around(255 * (ebsd_im - ebsd_im.min()) / (ebsd_im.max() - ebsd_im.min()), 0).astype(np.uint8)
        else:
            self.ebsd_im = ebsd_im
        if bse_im.ndim == 3 and bse_im.dtype != np.uint8:
            self.bse_im = np.around(255 * (bse_im - bse_im.min(axis=(0, 1))) / (bse_im.max(axis=(0, 1)) - bse_im.min(axis=(0, 1))), 0).astype(np.uint8)
        elif bse_im.ndim == 2 and bse_im.dtype != np.uint8:
            self.bse_im = np.around(255 * (bse_im - bse_im.min()) / (bse_im.max() - bse_im.min()), 0).astype(np.uint8)
        else:
            self.bse_im = bse_im
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


class ProgressWindow(tk.Toplevel):
    """ displays progress """
    def __init__(self, text):
        super().__init__()
        self.f = ttk.Frame(self)
        self.f.grid(row=0, column=0, sticky="nsew")
        self.note = text
        self.p_label = ttk.Label(self.f, text=self.note, font=("", 12, "bold"), anchor=tk.CENTER)
        self.p_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)
        self.progress = ttk.Progressbar(self.f, orient="horizontal", length=300, mode="indeterminate")
        self.progress.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.progress.start()


if __name__ == "__main__":
    app = App()
    app.mainloop()

