"""
Author: James Lamb

UI for running distortion correction
"""

# Python packages
import os

# from threading import Thread
from multiprocessing.pool import ThreadPool
import shutil
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# 3rd party packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import torch
from torchvision.transforms.functional import resize as RESIZE
from torchvision.transforms import InterpolationMode
from kornia.enhance import equalize_clahe

# Local files
import warping
import Inputs
import InteractiveView as IV


# TODO: Exporting 3D data has not been tested
# TODO: Handle errors when reading in files
# TODO: Fix toplevel behavior of IO windows
# TODO: Add saving the state so that the user can come back to it later more easily (basically save the filepaths to everything)
# TODO: Change the way the points are saved so that the filename is more clear


def CLAHE(im, clip_limit=20.0, kernel_size=(8, 8)):
    tensor = torch.tensor(im).unsqueeze(0).unsqueeze(0).float()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = equalize_clahe(tensor, clip_limit, kernel_size)
    tensor = torch.round(
        255 * (tensor - tensor.min()) / (tensor.max() - tensor.min()), decimals=0
    )
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
        filemenu.add_command(
            label="Open 2D", command=lambda: self.select_data_popup("2D")
        )
        filemenu.add_command(
            label="Export 2D", command=self.apply_and_export_2d, state="disabled"
        )
        filemenu.add_command(
            label="Open 3D", command=lambda: self.select_data_popup("3D")
        )
        filemenu.add_command(
            label="Export 3D", command=self.apply_and_export_3d, state="disabled"
        )
        filemenu.add_command(
            label="Export transform", command=self.export_transform, state="disabled"
        )
        filemenu.add_command(label="Quick start", command=self.easy_start)
        self.menu.add_cascade(label="File", menu=filemenu)
        applymenu2d = tk.Menu(self.menu, tearoff=0)
        labels = [
            "TPS",
            "Affine only TPS",
        ]
        keywords = [
            "tps",
            "tps affine",
        ]
        for label, keyword in zip(labels, keywords):
            applymenu2d.add_command(
                label=label,
                command=lambda keyword=keyword: self.apply_2d(keyword),
            )
        self.menu.add_cascade(
            label="View transform (current image)", menu=applymenu2d, state="disabled"
        )
        applymenu3d = tk.Menu(self.menu, tearoff=0)
        for label, keyword in zip(labels, keywords):
            applymenu3d.add_command(
                label=label,
                command=lambda keyword=keyword: self.apply_3d(keyword),
            )
        self.menu.add_cascade(
            label="Apply transform (all images)", menu=applymenu3d, state="disabled"
        )
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
        self.ebsd = tk.Canvas(
            self.viewer_left,
            highlightbackground=self.bg,
            bg=self.bg,
            bd=1,
            highlightthickness=0.2,
            cursor="tcross",
            width=int(screen_width * 0.45),
            height=int(screen_height * 0.7),
        )
        self.ebsd.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.ebsd_hscroll = ttk.Scrollbar(
            self.viewer_left,
            orient=tk.HORIZONTAL,
            command=self.ebsd.xview,
            cursor="sb_h_double_arrow",
        )
        self.ebsd_hscroll.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.ebsd_vscroll = ttk.Scrollbar(
            self.viewer_left,
            orient=tk.VERTICAL,
            command=self.ebsd.yview,
            cursor="sb_v_double_arrow",
        )
        self.ebsd_vscroll.grid(row=1, column=1, sticky="ns", padx=5, pady=5)
        self.ebsd.config(
            xscrollcommand=self.ebsd_hscroll.set, yscrollcommand=self.ebsd_vscroll.set
        )
        #
        # setup viewer right
        l = ttk.Label(self.viewer_right, text="BSE/Control image", anchor=tk.CENTER)
        l.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.bse = tk.Canvas(
            self.viewer_right,
            highlightbackground=self.bg,
            bg=self.bg,
            bd=1,
            highlightthickness=0.2,
            cursor="tcross",
            width=int(screen_width * 0.45),
            height=int(screen_height * 0.7),
        )
        self.bse.grid(row=1, column=0, pady=5, padx=5, sticky="nsew")
        self.bse_hscroll = ttk.Scrollbar(
            self.viewer_right,
            orient=tk.HORIZONTAL,
            command=self.bse.xview,
            cursor="sb_h_double_arrow",
        )
        self.bse_hscroll.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.bse_vscroll = ttk.Scrollbar(
            self.viewer_right,
            orient=tk.VERTICAL,
            command=self.bse.yview,
            cursor="sb_v_double_arrow",
        )
        self.bse_vscroll.grid(row=1, column=1, sticky="ns", padx=5, pady=5)
        self.bse.config(
            xscrollcommand=self.bse_hscroll.set, yscrollcommand=self.bse_vscroll.set
        )
        #
        # Bindings on viewers
        if os.name == "posix":
            remove_ = "<Button 2>"
            scalar_ = 1
        else:
            remove_ = "<Button 3>"
            scalar_ = 120
        self.ebsd.bind(remove_, lambda arg: self.remove_coords("ebsd", arg))
        self.ebsd.bind(
            "<MouseWheel>",
            lambda event: self.ebsd.yview_scroll(
                int(-1 * (event.delta / scalar_)), "units"
            ),
        )
        self.ebsd.bind(
            "<Shift-MouseWheel>",
            lambda event: self.ebsd.xview_scroll(
                int(-1 * (event.delta / scalar_)), "units"
            ),
        )
        self.bse.bind(remove_, lambda arg: self.remove_coords("bse", arg))
        self.bse.bind(
            "<MouseWheel>",
            lambda event: self.bse.yview_scroll(
                int(-1 * (event.delta / scalar_)), "units"
            ),
        )
        self.bse.bind(
            "<Shift-MouseWheel>",
            lambda event: self.bse.xview_scroll(
                int(-1 * (event.delta / scalar_)), "units"
            ),
        )
        self.ebsd.bind("<ButtonPress-1>", lambda arg: self.add_coords("ebsd", arg))
        self.bse.bind("<ButtonPress-1>", lambda arg: self.add_coords("bse", arg))
        self.bind("<Control-equal>", lambda event: self.change_zoom_event(+1), "units")
        self.bind("<Control-minus>", lambda event: self.change_zoom_event(-1), "units")
        # self.ebsd.bind('<Control-equal>', lambda event: self.change_zoom_event(+1, "ebsd"), "units")
        # self.ebsd.bind('<Control-minus>', lambda event: self.change_zoom_event(-1, "ebsd"), "units")
        # self.bse.bind('<Control-equal>', lambda event: self.change_zoom_event(+1, "bse"), "units")
        # self.bse.bind('<Control-minus>', lambda event: self.change_zoom_event(-1, "bse"), "units")
        #
        # setup bottom left
        self.clear_ebsd_points = ttk.Button(
            self.bot_left,
            text="Clear points",
            command=lambda: self.clear_points("ebsd"),
            state="disabled",
        )
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
        self.clahe_ebsd_b = ttk.Button(
            self.bot_left,
            text="Apply CLAHE",
            command=lambda *args: self.clahe("ebsd"),
            state="disabled",
        )
        self.clahe_ebsd_b.grid(row=0, column=5, sticky="ew", padx=5, pady=5)
        sep = ttk.Separator(self.bot_left, orient=tk.VERTICAL)
        sep.grid(row=0, column=6, sticky="ns")

        # setup bottom right
        self.clear_bse_points = ttk.Button(
            self.bot_right,
            text="Clear points",
            command=lambda: self.clear_points("bse"),
            state="disabled",
        )
        self.clear_bse_points.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        sep = ttk.Separator(self.bot_right, orient=tk.VERTICAL)
        sep.grid(row=0, column=1, sticky="ns")
        bse_mode_label = ttk.Label(self.bot_right, text="EBSD mode:")
        bse_mode_label.grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.bse_mode = tk.StringVar()
        self.bse_mode_options = ["Intensity"]
        self.bse_mode.set(self.bse_mode_options[0])
        self.bse_picker = ttk.Combobox(
            self.bot_right,
            textvariable=self.bse_mode,
            values=self.bse_mode_options,
            height=10,
            width=20,
        )
        self.bse_picker["state"] = "disabled"
        self.bse_picker.bind("<<ComboboxSelected>>", self._update_viewers)
        self.bse_picker.grid(row=0, column=3, sticky="ew", padx=5, pady=5)
        sep = ttk.Separator(self.bot_right, orient=tk.VERTICAL)
        sep.grid(row=0, column=4, sticky="ns")
        self.clahe_bse_b = ttk.Button(
            self.bot_right,
            text="Apply CLAHE",
            command=lambda *args: self.clahe("bse"),
            state="disabled",
        )
        self.clahe_bse_b.grid(row=0, column=5, sticky="ew", padx=5, pady=5)
        sep = ttk.Separator(self.bot_right, orient=tk.VERTICAL)
        sep.grid(row=0, column=6, sticky="ns")

        # Setup resizing
        self.resize_options = [
            1,
            5,
            10,
            25,
            50,
            75,
            100,
            125,
            150,
            175,
            200,
            300,
            400,
            600,
        ]
        self.resize_var_ebsd = tk.StringVar()
        self.resize_var_bse = tk.StringVar()
        self.resize_var_ebsd.set(self.resize_options[3])
        self.resize_var_bse.set(self.resize_options[3])
        self.resize_var_ebsd.trace_add(
            "write", lambda *args: self._resize("ebsd", self.resize_vars["ebsd"].get())
        )
        self.resize_var_bse.trace_add(
            "write", lambda *args: self._resize("bse", self.resize_vars["bse"].get())
        )
        ebsd_resize_label = ttk.Label(self.bot_left, text="Zoom:")
        bse_resize_label = ttk.Label(self.bot_right, text="Zoom:")
        ebsd_resize_label.grid(row=0, column=7, sticky="e", padx=5, pady=5)
        bse_resize_label.grid(row=0, column=7, sticky="e", padx=5, pady=5)
        self.ebsd_resize_dropdown = ttk.Combobox(
            self.bot_left,
            textvariable=self.resize_var_ebsd,
            values=self.resize_options,
            state="readonly",
            width=5,
        )
        self.bse_resize_dropdown = ttk.Combobox(
            self.bot_right,
            textvariable=self.resize_var_bse,
            values=self.resize_options,
            state="readonly",
            width=5,
        )
        self.ebsd_resize_dropdown["state"] = "disabled"
        self.bse_resize_dropdown["state"] = "disabled"
        self.ebsd_resize_dropdown.grid(row=0, column=8, sticky="ew")
        self.bse_resize_dropdown.grid(row=0, column=8, sticky="ew")
        self.resize_vars = {"ebsd": self.resize_var_ebsd, "bse": self.resize_var_bse}

        ### Additional things added
        self.clahe_active_bse = False
        self.clahe_active_ebsd = False
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
        # ebsd_path = "./test_data/EBSD.ang"
        # ebsd_points_path = "./test_data/distorted_pts.txt"
        # bse_path = "./test_data/BSE.tif"
        # bse_points_path = "./test_data/control_pts.txt"
        # ebsd_res = 2.5
        # bse_res = 1.0
        # rescale = True
        # r180 = False
        # flip = False
        # crop = False
        ebsd_points_path = (
            "/Users/jameslamb/Documents/research/data/CoNi90-thin/distorted_pts.txt"
        )
        bse_points_path = (
            "/Users/jameslamb/Documents/research/data/CoNi90-thin/control_pts.txt"
        )
        ebsd_path = (
            "/Users/jameslamb/Documents/research/data/CoNi90-thin/CoNi90-thin.dream3d"
        )
        bse_path = "/Users/jameslamb/Documents/research/data/CoNi90-thin/CoNi90-thin_BSE_small.dream3d"
        ebsd_res = 1.0
        bse_res = 1.0
        rescale = False
        r180 = False
        flip = True
        crop = False
        self.ang_path = ebsd_path
        self.ebsd_res = ebsd_res
        self.bse_res = bse_res
        e_d, b_d, e_pts, b_pts = self._run_in_background(
            "Importing data...",
            Inputs.read_data,
            ebsd_path,
            bse_path,
            ebsd_points_path,
            bse_points_path,
        )
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
        if flip or r180:
            b_d = Inputs.flip_rotate_control(b_d, flip, r180)
        # Crop if desired
        if crop:
            ### TODO: Add multiple control images
            self.w = Inputs.CropControl(self, b_d[0, :, :])
            self.wait_window(self.w.w)
            if self.w.clean_exit:
                s, e = self.w.start, self.w.end
                ### TODO: Add multiple control images
                b_d = b_d[:, s[0] : e[0], s[1] : e[1]]
        self._handle_input(
            ebsd_path,
            bse_path,
            e_d,
            b_d,
            ebsd_points_path,
            bse_points_path,
            e_pts,
            b_pts,
            ebsd_res,
            bse_res,
        )

    def select_data_popup(self, mode):
        self.w = Inputs.DataInput(self, mode)
        self.wait_window(self.w.w)
        self.update()
        if self.w.clean_exit:
            if self.w.ang:
                self.ang_path = self.w.ebsd_path
            ebsd_path, bse_path = self.w.ebsd_path, self.w.bse_path
            ebsd_points_path, bse_points_path = (
                self.w.ebsd_points_path,
                self.w.bse_points_path,
            )
            ebsd_res, bse_res = self.w.ebsd_res, self.w.bse_res
            e_d, b_d, e_pts, b_pts = Inputs.read_data(
                ebsd_path, bse_path, ebsd_points_path, bse_points_path
            )
            self.w = Inputs.DataSummary(self, e_d, b_d, e_pts, b_pts, ebsd_res, bse_res)
            self.wait_window(self.w.w)
            self.update()
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
                if self.w.flip or self.w.r180:
                    b_d = Inputs.flip_rotate_control(b_d, self.w.flip, self.w.r180)
                # Crop if desired
                if self.w.crop:
                    im = b_d[list(b_d.keys())[0]][0]
                    self.w = Inputs.CropControl(self, im)
                    self.wait_window(self.w.w)
                    self.update()
                    if self.w.clean_exit:
                        s, e = self.w.start, self.w.end
                        for key in b_d.keys():
                            b_d[key] = b_d[key][:, s[0] : e[0], s[1] : e[1]]
                self._handle_input(
                    ebsd_path,
                    bse_path,
                    e_d,
                    b_d,
                    ebsd_points_path,
                    bse_points_path,
                    e_pts,
                    b_pts,
                    ebsd_res,
                    bse_res,
                )
            else:
                return

    def _handle_input(
        self,
        ebsd_path,
        bse_path,
        e_d,
        b_d,
        ebsd_points_path,
        bse_points_path,
        e_pts,
        b_pts,
        ebsd_res,
        bse_res,
    ):
        # Set the data
        self.points_path = {"ebsd": ebsd_points_path, "bse": bse_points_path}
        self.ebsd_path, self.bse_path = ebsd_path, bse_path
        self.ebsd_data, self.bse_data = e_d, b_d
        self.ebsd_res, self.bse_res = ebsd_res, bse_res
        self.points["ebsd"], self.points["bse"] = e_pts, b_pts
        # Set the UI stuff
        self.ebsd_mode_options = list(self.ebsd_data.keys())
        self.ebsd_mode.set(self.ebsd_mode_options[0])
        self.bse_mode_options = list(self.bse_data.keys())
        self.bse_mode.set(self.bse_mode_options[0])
        self.slice_min = 0
        self.slice_max = self.ebsd_data[self.ebsd_mode_options[0]].shape[0] - 1
        self.slice_num.set(self.slice_min)
        # Update slice picker values
        self.slice_picker["values"] = list(
            np.arange(self.slice_min, self.slice_max + 1)
        )
        # Turn on GUI elements that are always on
        self.menu.entryconfig("View transform (current image)", state="normal")
        self.menu.nametowidget(self.menu.entrycget("File", "menu")).entryconfig(
            "Export transform", state="normal"
        )
        self.menu.nametowidget(self.menu.entrycget("File", "menu")).entryconfig(
            "Export 2D", state="normal"
        )
        # self.menu.entryconfig("Export transform", state="normal")
        # self.menu.entryconfig("Export 2D", state="normal")
        self.ebsd_picker["state"] = "readonly"
        self.ebsd_picker["values"] = self.ebsd_mode_options
        self.bse_picker["state"] = "readonly"
        self.bse_picker["values"] = self.bse_mode_options
        # Turn on dimensionality specific elements
        if self.slice_max == self.slice_min:  # 2D data
            self.slice_picker["state"] = "disabled"
        else:  # 3D data
            self.slice_picker["state"] = "readonly"
            self.menu.entryconfig("Apply transform (all images)", state="normal")
            self.menu.nametowidget(self.menu.entrycget("File", "menu")).entryconfig(
                "Export 3D", state="normal"
            )
        # Finish
        self.folder = os.path.dirname(ebsd_path)
        self._update_viewers()
        self.clear_ebsd_points["state"] = "normal"
        self.clear_bse_points["state"] = "normal"
        self.ebsd_resize_dropdown["state"] = "readonly"
        self.bse_resize_dropdown["state"] = "readonly"
        self.clahe_bse_b["state"] = "normal"
        self.clahe_ebsd_b["state"] = "normal"
        self.view_pts["state"] = "normal"
        self.show_points.set(1)

    def export_transform(self):
        save_path = filedialog.asksaveasfilename(
            initialdir=self.folder,
            title="Save the transform",
            filetypes=(
                ("Binary NumPy Array", ("*.npy")),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ),
        )
        if save_path == "":
            return
        delimiter = "," if save_path.endswith(".csv") else " "

        # Get the mode
        mode = ArbitraryMessageBox(
            self,
            title="Mode",
            msg="Select the mode to use for the correction.",
            options=[
                "Thin-Plate Spline",
                "Thin-Plate Spline (Affine only)",
            ],
            orientation="vertical",
        )
        if mode is None:
            return
        elif mode == False:
            return
        else:
            mode = [
                "tps",
                "tps affine",
            ][mode.choice]
        print("Mode choice:", mode)

        # Get the shape of the BSE image and the points
        dst_img_shape = self.bse_data[self.bse_mode.get()][self.slice_num.get()].shape
        src_points = np.array(self.points["ebsd"][self.slice_num.get()])
        dst_points = np.array(self.points["bse"][self.slice_num.get()])
        if src_points.size == 0 or dst_points.size == 0:
            messagebox.showerror(
                "Error",
                "No points selected for the transformation. Please select points.",
            )
            return

        # Get the transform for the EBSD data
        if "tps" in mode.lower():
            tform = warping.get_transform(
                src_points, dst_points, mode=mode, size=dst_img_shape
            )
        else:
            tform = warping.get_transform(src_points, dst_points, mode=mode)
        params = tform.params

        # Save the transform
        print("Saving transform to:", save_path)
        if save_path.endswith(".npy"):
            np.save(save_path, params)
        else:
            params = params.reshape(2, -1).T
            np.savetxt(
                save_path,
                params,
                delimiter=delimiter,
                header=f"Transform mode: {mode}, Shape: {params.shape}",
                comments="#",
            )

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
        if self.points_path["ebsd"] == "" or self.points_path["bse"] == "":
            return
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
        if pos == "bse":
            v = self.bse
        elif pos == "ebsd":
            v = self.ebsd
        closest = v.find_closest(v.canvasx(event.x), v.canvasy(event.y))[0]
        tag = v.itemcget(closest, "tags")
        tag = tag.replace("current", "").replace("text", "").replace("bbox", "").strip()
        if tag == "":
            return
        self.points[pos][self.slice_num.get()] = np.delete(
            self.points[pos][self.slice_num.get()], int(tag), axis=0
        )
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
                    o_item = viewers[mode].create_oval(
                        p[0] - 1,
                        p[1] - 1,
                        p[0] + 1,
                        p[1] + 1,
                        width=2,
                        outline=pc[mode],
                        tags=str(i),
                    )
                    t_item = viewers[mode].create_text(
                        p[0] + 3,
                        p[1] + 3,
                        anchor=tk.NW,
                        text=i,
                        fill=pc[mode],
                        font=("", 10, "bold"),
                        tags=str(i) + "text",
                    )
                    bbox = viewers[mode].bbox(t_item)
                    r_item = viewers[mode].create_rectangle(
                        bbox, fill="#FFFFFF", outline=pc[mode], tags=str(i) + "bbox"
                    )
                    viewers[mode].tag_raise(t_item, r_item)
        else:
            self.ebsd.delete("all")
            self.bse.delete("all")
            self._update_imgs()

    ### Apply stuff for visualizing ###
    def _get_cropping_slice(self, centroid, target_shape, current_shape):
        """Returns a slice object that can be used to crop an image"""
        rstart = centroid[0] - target_shape[0] // 2
        rend = rstart + target_shape[0]
        if rstart < 0:
            r_slice = slice(0, target_shape[0])
        elif rend > current_shape[0]:
            r_slice = slice(current_shape[0] - target_shape[0], current_shape[0])
        else:
            r_slice = slice(rstart, rend)

        cstart = centroid[1] - target_shape[1] // 2
        cend = cstart + target_shape[1]
        if cstart < 0:
            c_slice = slice(0, target_shape[1])
        elif cend > current_shape[1]:
            c_slice = slice(current_shape[1] - target_shape[1], current_shape[1])
        else:
            c_slice = slice(cstart, cend)
        return r_slice, c_slice

    def _get_ebsd_path(self, volume=False):
        # Set filetypes depending on image or volume data
        filetypes = [
            ("DREAM.3D File", "*.dream3d"),
            ("EBSD Data", "*.ang *.h5"),
            ("Image Files", "*.tif *.tiff *.png *.jpg *.jpeg"),
            ("All files", "*.*"),
        ]
        if volume:
            filetypes.pop(1)
            defaultextension = ".dream3d"
            title = "Save corrected (from distorted) volume"
        else:
            filetypes.pop(0)
            defaultextension = ".ang"
            title = "Save corrected (from distorted) image"
        # Ask for the EBSD path
        SAVE_PATH_EBSD = filedialog.asksaveasfilename(
            initialdir=os.path.basename(self.ebsd_path),
            title=title,
            filetypes=filetypes,
            defaultextension=defaultextension,
        )
        # Check if the user canceled the dialog
        if SAVE_PATH_EBSD == "":
            return None
        elif "." not in SAVE_PATH_EBSD:
            raise None
        # Check if the extension is valid
        extension = os.path.splitext(SAVE_PATH_EBSD)[1]
        if extension.lower() not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
            ".h5",
            ".ang",
            ".dream3d",
        ]:
            return None
        print("Saving EBSD to:", SAVE_PATH_EBSD)
        return SAVE_PATH_EBSD

    def _get_mode_choice(self):
        # Get the mode choice
        mode = ArbitraryMessageBox(
            self,
            title="Mode",
            msg="Select the mode to use for the correction.",
            options=[
                "Thin-Plate Spline",
                "Thin-Plate Spline (Affine only)",
            ],
            orientation="vertical",
        )
        if mode == False:
            return None
        else:
            mode = [
                "tps",
                "tps affine",
            ][mode.choice]
        print("Mode choice:", mode)
        return mode

    def _get_crop_choice(self):
        # Ask if the user wants to crop the corrected image to match the distorted grid
        crop_choice = ArbitraryMessageBox(
            self,
            title="Crop",
            msg="Select the grid to crop the transformed output to.",
            options=["Source (EBSD)", "Destination (BSE)", "None"],
        )
        if crop_choice == False:
            return None
        else:
            crop_choice = ["src", "dst", "none"][crop_choice.choice]
        print("Crop choice:", crop_choice)
        return crop_choice

    def _correct_dtype(self, data, dtype):
        print("Correcting dtype", dtype, data.dtype)
        # Fix integer types, leave float types as is
        if data.dtype == dtype:
            return data
        else:
            try:
                # Succeeds if it tis an integer dtype
                info = np.iinfo(dtype)
                # Convert to 0-1 range and then scale to the max value of the dtype
                data = (2**info.bits - 1) * data / data.max()
                # Convert to the correct dtype
                data += info.min
                return np.around(data, 0).astype(dtype)
            except ValueError:
                # Fails if it is a float dtype
                return data.astype(dtype)

    def _get_dream3d_structure(self, path, data_keys):
        h5 = h5py.File(path, "r")
        # Get the outer key
        keys_0 = list(h5.keys())
        keys_0_lower = [k.lower() for k in keys_0]
        key = [
            keys_0[i]
            for i in range(len(keys_0))
            if keys_0_lower[i] not in ["datacontainerbundles", "montages", "pipeline"]
        ][0]
        # Always only one key
        key = key + f"/{list(h5[key].keys())[0]}"
        # Match
        keys = [key + f"/{k}" for k in h5[key].keys()]
        for k in keys:
            if len(h5[k].keys()) == len(data_keys):
                key = k
                break
        h5.close()
        return key

    def apply_2d(self, mode="tps"):
        """Applies the correction algorithm and calls the interactive view"""
        crop_choice = self._get_crop_choice()

        # im0, im1 = self._run_in_background("Applying correction...", self._apply, crop_choice, mode)
        im0, im1 = self._apply_2d(crop_choice, mode)
        if im0 is None or im1 is None:
            messagebox.showerror(
                "Error",
                "No points selected for the transformation. Please select points.",
            )
            return
        if (im0.shape[0] > 2000) or (im0.shape[1] > 2000):
            result = tk.messagebox.askyesnocancel(
                "Resize?",
                "The images are large, would you like to resize for the preview? (Recommended)",
            )
            if result is None:
                pass
            if result:
                scale = int(max(im0.shape) / 2000)
                im0 = im0[::scale, ::scale]
                im1 = im1[::scale, ::scale]
        # Handle multiple channels
        im0 = np.squeeze(im0).astype(np.float32)
        im1 = np.squeeze(im1).astype(np.float32)
        if im1.ndim > im0.ndim:
            im0 = np.repeat(im0[:, :, np.newaxis], im1.shape[-1], axis=-1)
        # Correct intensity ranges, normalize the data to 0-1 range
        im0 = (im0 - im0.min()) / (im0.max() - im0.min()) * im1.max()
        # View
        print("Creating interactive view")
        IV.Interactive2D(im0, im1, "2D TPS Correction")
        plt.close("all")

    def apply_3d(self, mode="tps"):
        """Applies the correction algorithm and calls the interactive view"""
        crop_choice = self._get_crop_choice()

        # im0, im1 = self._run_in_background(
        #     "Applying correction...", self._apply_3d, crop_choice, mode
        # )
        im0, im1 = self._apply_3d(
            crop_choice, mode
        )  # im0-dst/control, im1-src/distorted
        if im0 is None or im1 is None:
            messagebox.showerror(
                "Error",
                "No points selected for the transformation. Please select points.",
            )
            return
        # Handle multiple channels
        im0 = np.squeeze(im0).astype(np.float32)
        im1 = np.squeeze(im1).astype(np.float32)
        if im1.ndim > im0.ndim:
            im0 = np.repeat(im0[:, :, :, np.newaxis], im1.shape[-1], axis=-1)
        # Correct intensity ranges, normalize the data to 0-1 range
        im0 = (im0 - im0.min()) / (im0.max() - im0.min())
        im1 = (im1 - im1.min()) / (im1.max() - im1.min())

        # View
        print("Creating interactive view")
        print("im0 shape:", im0.shape)
        print("im1 shape:", im1.shape)
        IV.Interactive3D(im0, im1, "3D TPS Correction")
        plt.close("all")

    def apply_and_export_2d(self):
        # Get save location and make sure it is valid
        SAVE_PATH_EBSD = filedialog.asksaveasfilename(
            initialdir=os.path.basename(self.ebsd_path),
            title="Save corrected (from distorted) image",
            filetypes=[
                ("EBSD Data", "*.ang *.h5"),
                ("Image Files", "*.tif *.tiff *.png *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
            defaultextension=".ang",
        )
        if SAVE_PATH_EBSD == "":
            return
        elif "." not in SAVE_PATH_EBSD:
            raise ValueError("No extension provided!")
        extension = os.path.splitext(SAVE_PATH_EBSD)[1]
        if extension.lower() not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
            ".h5",
            ".ang",
        ]:
            raise ValueError("Invalid extension provided!")

        # Get the BSE path based on the EBSD path
        if extension.lower() in [".h5", ".ang"]:
            SAVE_PATH_BSE = os.path.splitext(SAVE_PATH_EBSD)[0] + "_dst.tif"
        else:
            SAVE_PATH_BSE = os.path.splitext(SAVE_PATH_EBSD)[0] + "_dst" + extension

        # Get the mode choice
        mode = ArbitraryMessageBox(
            self,
            title="Mode",
            msg="Select the mode to use for the correction.",
            options=[
                "Thin-Plate Spline",
                "Thin-Plate Spline (Affine only)",
            ],
            orientation="vertical",
        )
        if mode == False:
            return
        else:
            mode = [
                "tps",
                "tps affine",
            ][mode.choice]
        print("Mode choice:", mode)

        # Ask if the user wants to crop the corrected image to match the distorted grid
        crop_choice = ArbitraryMessageBox(
            self,
            title="Crop",
            msg="Select the grid to crop the transformed output to.",
            options=["Source (EBSD)", "Destination (BSE)", "None"],
        )
        if crop_choice == False:
            return
        else:
            crop_choice = ["src", "dst", "none"][crop_choice.choice]
        print("Crop choice:", crop_choice)

        # Get the modality keys
        ebsd_keys = list(self.ebsd_data.keys())
        ebsd_keys.pop([k.lower() for k in ebsd_keys].index("eulerangles"))
        bse_keys = list(self.bse_data.keys())

        # Apply the correction
        # dst_img, src_imgs = self._run_in_background("Applying correction to all EBSD modes...", self._apply_multiple, crop_to_distorted_grid)
        dst_imgs, src_imgs = self._apply_multiple_2d(crop_choice, mode=mode)

        # Handle dtype of images
        # for i, key in enumerate(ebsd_keys):
        #     src_imgs[i] = core.handle_dtype(src_imgs[i], self.ebsd_data[key].dtype)

        # Get the ang header
        _, _, col_names, res, header_string, _ = Inputs.read_ang_header(self.ang_path)

        # Create the grid for the ang file
        x, y = np.meshgrid(
            np.arange(src_imgs[0].shape[1]), np.arange(src_imgs[0].shape[0])
        )
        x.flatten().astype(np.float32) * res
        y.flatten().astype(np.float32) * res
        src_imgs[col_names.index("x")] = x
        src_imgs[col_names.index("y")] = y

        # Modify the header according to the new grid
        nrows_index = header_string.index("# NROWS: ")
        end_index = header_string[nrows_index:].index("\n") + nrows_index
        header_string = header_string.replace(
            header_string[nrows_index:end_index], f"# NROWS: {src_imgs[0].shape[0]}"
        )
        ncolsE_index = header_string.index("# NCOLS_EVEN: ")
        end_index = header_string[ncolsE_index:].index("\n") + ncolsE_index
        header_string = header_string.replace(
            header_string[ncolsE_index:end_index],
            f"# NCOLS_EVEN: {src_imgs[0].shape[1]}",
        )
        ncolsO_index = header_string.index("# NCOLS_ODD: ")
        end_index = header_string[ncolsO_index:].index("\n") + ncolsO_index
        header_string = header_string.replace(
            header_string[ncolsO_index:end_index],
            f"# NCOLS_ODD: {src_imgs[0].shape[1]}",
        )

        # Save the data based on the extension
        if extension.lower() == ".ang":
            data_out = []
            for i, key in enumerate(ebsd_keys):
                if key in col_names:
                    data_out.append(src_imgs[i].reshape(-1, 1).astype(np.float32))
            data_out = np.hstack(data_out)
            np.savetxt(
                SAVE_PATH_EBSD,
                data_out,
                header=header_string,
                fmt="%.5f",
                delimiter=" ",
                comments="",
            )
            for i, key in enumerate(bse_keys):
                SAVE_PATH_BSE = os.path.splitext(SAVE_PATH_BSE)[0] + f"_{key}.tif"
                io.imsave(SAVE_PATH_BSE, dst_imgs[i])

        elif extension.lower() == ".h5":
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
                h5.create_dataset(
                    name=f"{key}", data=dst_imgs[i], dtype=dst_imgs[i].dtype
                )
                h5[f"{key}"].attrs.create(name="name", data=f"{key}")
                h5[f"{key}"].attrs.create(name="dtype", data=str(dst_imgs[i].dtype))
                h5[f"{key}"].attrs.create(name="shape", data=dst_imgs[i].shape)
            h5.close()
        else:
            for i, key in enumerate(ebsd_keys):
                SAVE_PATH_EBSD = (
                    os.path.splitext(SAVE_PATH_EBSD)[0] + f"_{key}" + extension
                )
                io.imsave(SAVE_PATH_EBSD, src_imgs[i])
            for i, key in enumerate(bse_keys):
                SAVE_PATH_BSE = (
                    os.path.splitext(SAVE_PATH_BSE)[0] + f"_{key}." + extension
                )
                io.imsave(SAVE_PATH_BSE, dst_imgs[i])

        print("Correction complete!")

    def apply_and_export_3d(self):
        # Get save location and make sure it is valid
        SAVE_PATH_EBSD = self._get_ebsd_path(volume=True)
        if SAVE_PATH_EBSD is None:
            print("Path not provided or invalid!")
            return
        elif not SAVE_PATH_EBSD.endswith(".dream3d"):
            raise ValueError("The save path must end with .dream3d for DREAM.3D files.")

        # Get the mode choice
        mode = self._get_mode_choice()
        if mode is None:
            print("Mode not selected!")
            return
        crop_choice = "src"

        # Copy old dream3d file and parse the structure
        shutil.copyfile(self.ebsd_path, SAVE_PATH_EBSD)
        cell_data_key = self._get_dream3d_structure(
            SAVE_PATH_EBSD, self.ebsd_data.keys()
        )
        print("Cell data key:", cell_data_key)

        # Apply corrections
        ref_written = False
        rslc, cslc = None, None
        params = None
        with h5py.File(SAVE_PATH_EBSD, "r+") as h5:
            for key in self.ebsd_data.keys():
                print("Saving key:", key)
                # Update current mode and correct the stack
                dtype = h5[cell_data_key][key].dtype
                shape = h5[cell_data_key][key].shape
                self.ebsd_mode.set(key)
                reference_stack, corrected_stack, rslc, cslc, params = self._apply_3d(
                    crop_choice,
                    mode,
                    rslc=rslc,
                    cslc=cslc,
                    params=params,
                    return_params=True,
                )

                # Store the corrected stack
                h5[cell_data_key][key][...] = self._correct_dtype(
                    corrected_stack, dtype
                )

                # Check if the current key is a valid copy candidate for the reference stack
                if (
                    dtype == np.uint8
                    and shape == reference_stack.shape
                    and not ref_written
                ):
                    h5.copy(
                        f"{cell_data_key}/{key}",
                        f"{cell_data_key}/BSE",
                    )
                    h5[cell_data_key + "/BSE"][...] = reference_stack
                    h5.close()
                    ref_written = True

        print("Saved DREAM.3D file to:", SAVE_PATH_EBSD)

    def _apply_2d(self, crop_choice, mode="tps"):
        print("Transforming image using mode:", mode)
        # Get the bse image and process
        dst_img = self.bse_data[self.bse_mode.get()][self.slice_num.get()]
        src_img = self.ebsd_data[self.ebsd_mode.get()][self.slice_num.get()]
        print("SRC shape:", src_img.shape)

        # Get the points and perform alignment
        src_points = np.array(self.points["ebsd"][self.slice_num.get()])
        dst_points = np.array(self.points["bse"][self.slice_num.get()])
        if src_points.size == 0 or dst_points.size == 0:
            print("No points found, returning")
            return None, None
        print("Gathered data for correction")

        # Process
        if self.clahe_active_bse:
            dst_img = CLAHE(dst_img)
        if self.clahe_active_ebsd:
            src_img = CLAHE(src_img)

        # Stack a dummy channel
        dummy = np.ones(src_img[..., :1].shape, dtype=src_img.dtype)
        print("Dummy shape:", dummy.shape)
        src_img = np.concatenate((src_img, dummy), axis=-1)
        print("Source image shape after stacking dummy channel:", src_img.shape)

        # Warp all the EBSD modes
        if "tps" in mode.lower():
            warped_src = warping.transform_image(
                src_img,
                src_points,
                dst_points,
                output_shape=dst_img.shape[:2],
                mode=mode,
                size=dst_img.shape[:2],
            )
        else:
            warped_src = warping.transform_image(
                src_img,
                src_points,
                dst_points,
                output_shape=dst_img.shape[:2],
                mode=mode,
            )

        IV.Interactive2D(warped_src[..., 1], warped_src[..., 1], "Warped Source Image")

        # Crop to the center of the corrected image if desired
        if crop_choice == "src":
            print("\tCropping to match distorted grid")
            # Do this by correcting an empty image (all ones) and finding the centroid of the corrected image
            dummy = warped_src[..., -1]
            rc, cc = np.array(np.where(dummy)).reshape(2, -1).T.mean(axis=0).astype(int)
            print("\tCentroid of corrected image:", rc, cc)
            rslc, cslc = self._get_cropping_slice(
                (rc, cc), src_img.shape, dst_img.shape[:2]
            )
            print("\tCropping slices:", rslc, cslc)
            dst_img = dst_img[rslc, cslc]
            warped_src = warped_src[rslc, cslc]
        elif crop_choice == "dst":
            print("\tCropping to match destination grid")
            warped_src = warped_src[: dst_img.shape[0], : dst_img.shape[1]]
        else:
            print("\tNot cropping, leaving destination and warped source as is.")
        warped_src = warped_src[..., :-1]  # Remove the dummy channel

        return dst_img, warped_src

    def _apply_multiple_2d(self, crop_choice, mode="tps"):
        # Get the bse image and process
        src_img_shape = self.ebsd_data[self.ebsd_mode.get()][self.slice_num.get()].shape
        dst_img_shape = self.bse_data[self.bse_mode.get()][self.slice_num.get()].shape
        dst_imgs = [
            self.bse_data[key][self.slice_num.get()] for key in self.bse_data.keys()
        ]
        print("SRC shape:", src_img_shape)
        print("DST shape:", dst_img_shape)

        # Get the points and perform alignment
        src_points = np.array(self.points["ebsd"][self.slice_num.get()])
        dst_points = np.array(self.points["bse"][self.slice_num.get()])
        print("Gathered data for correction")

        # Process
        # if dst_img_shape[0] < src_img_shape[0]:
        #     dst_imgs = [np.pad(img, ((0, src_img_shape[0] - img.shape[0]), (0, 0)), mode="constant", constant_values=0) for img in dst_imgs]
        # if dst_img_shape[1] < src_img_shape[1]:
        #     dst_imgs = [np.pad(img, ((0, 0), (0, src_img_shape[1] - img.shape[1])), mode="constant", constant_values=0) for img in dst_imgs]

        # Get the transform for the EBSD data
        if "tps" in mode.lower():
            tform = warping.get_transform(
                src_points, dst_points, mode=mode, size=dst_img_shape
            )
        else:
            tform = warping.get_transform(src_points, dst_points, mode=mode)

        # Warp all the EBSD modes
        warped_srcs = []
        for i, ebsd_mode in enumerate(self.ebsd_data.keys()):
            print(
                f"Processing source (EBSD) mode {ebsd_mode} ({i + 1}/{len(self.ebsd_data.keys())})"
            )
            if "eulerangles" in ebsd_mode.lower():
                continue
            src_img = self.ebsd_data[ebsd_mode][self.slice_num.get()]
            warped_src_img = transform.warp(
                src_img,
                tform,
                mode="constant",
                cval=0,
                order=0,
                output_shape=dst_img_shape,
            )
            warped_srcs.append(warped_src_img)

        # Crop to the center of the corrected image if desired
        if crop_choice == "src":
            print("Cropping to match source (EBSD) grid")
            # Do this by correcting an empty image (all ones) and finding the centroid of the corrected image
            dummy = np.ones(src_img_shape)
            dummy = warping.transform_image(
                dummy,
                dst_points,
                src_points,
                output_shape=dst_img_shape,
                mode="tps",
                size=dst_img_shape,
            )
            rc, cc = np.array(np.where(dummy)).reshape(2, -1).T.mean(axis=0).astype(int)
            rslc, cslc = self._get_cropping_slice(
                (rc, cc), src_img_shape, dst_img_shape
            )
            dst_imgs = [img[rslc, cslc] for img in dst_imgs]
            warped_srcs = [img[rslc, cslc] for img in warped_srcs]
        elif crop_choice == "dst":
            print("Cropping to match destination grid")
            warped_srcs = [
                img[: dst_img_shape[0], : dst_img_shape[1]] for img in warped_srcs
            ]
        else:
            print("Not cropping, leaving destination and warped source as is.")

        print("Finished processing")
        return dst_imgs, warped_srcs

    def _apply_3d(
        self,
        crop_choice,
        mode="tps",
        rslc=None,
        cslc=None,
        params=None,
        return_params=False,
    ):
        # Get shapes
        src_img_shape = self.ebsd_data[self.ebsd_mode.get()][
            self.slice_num.get()
        ].shape[:2]
        dst_img_shape = self.bse_data[self.bse_mode.get()][self.slice_num.get()].shape[
            :2
        ]

        # Get images
        image_stack = self.ebsd_data[self.ebsd_mode.get()]
        image_stack = np.concatenate(
            (image_stack, np.ones_like(image_stack[..., :1])), axis=-1
        )
        reference_stack = np.squeeze(self.bse_data[self.bse_mode.get()])

        # Get points
        src_points = []
        dst_points = []
        for k in self.points["ebsd"].keys():
            src = np.array(self.points["ebsd"][k]).reshape(-1, 2)
            dst = np.array(self.points["bse"][k]).reshape(-1, 2)
            src = np.hstack((np.ones((src.shape[0], 1)) * k, src))
            dst = np.hstack((np.ones((dst.shape[0], 1)) * k, dst))
            src_points.extend(src)
            dst_points.extend(dst)
        src_points = np.array(src_points).astype(int)
        dst_points = np.array(dst_points).astype(int)

        # Transform the image stack
        if "tps" in mode.lower():
            corrected_stack = warping.transform_image_stack(
                image_stack,
                src_points,
                dst_points,
                dst_img_shape,
                mode,
                order=0,
                size=dst_img_shape,
                params=params,
                return_params=return_params,
            )
            if return_params:
                corrected_stack, params = corrected_stack
        else:
            corrected_stack = warping.transform_image_stack(
                image_stack, src_points, dst_points, dst_img_shape, mode, order=0
            )

        # Crop to the center of the corrected image if desired
        if crop_choice == "src":
            print("\tCropping to match distorted grid")
            if rslc is None or cslc is None:
                # Do this by correcting an empty image (all ones) and finding the centroid of the corrected image
                dummy = corrected_stack[:, :, :, -1]
                zc, yc, xc = (
                    np.array(np.where(dummy)).reshape(3, -1).T.mean(axis=0).astype(int)
                )
                rslc, cslc = self._get_cropping_slice(
                    (yc, xc), src_img_shape, dst_img_shape
                )
            print("\tCropping slices:", rslc, cslc)
            reference_stack = reference_stack[:, rslc, cslc]
            corrected_stack = corrected_stack[:, rslc, cslc]
        elif crop_choice == "dst":
            print("\tCropping to match destination grid")
            corrected_stack = corrected_stack[:, : dst_img_shape[0], : dst_img_shape[1]]
        else:
            print("\tNot cropping, leaving destination and warped source as is.")
        # Remove the last channel (the one with the ones)
        corrected_stack = corrected_stack[:, :, :, :-1]

        if return_params:
            # Return the slices for the reference and corrected stacks
            return reference_stack, corrected_stack, rslc, cslc, params

        return reference_stack, corrected_stack

    def _apply_multiple_3d(self, crop_choice, mode="tps"):
        # Get shapes
        src_img_shape = self.ebsd_data[self.ebsd_mode.get()][
            self.slice_num.get()
        ].shape[:2]
        dst_img_shape = self.bse_data[self.bse_mode.get()][self.slice_num.get()].shape[
            :2
        ]

        # Get images
        keys = list(self.ebsd_data.keys())
        image_stack = self.ebsd_data[keys[0]]
        for k in keys[1:]:
            image_stack = np.concatenate((image_stack, self.ebsd_data[k]), axis=-1)
        # Add a stack of ones for cropping the data afterwards
        image_stack = np.concatenate(
            (image_stack, np.ones(image_stack.shape[:-1] + (1,))), axis=-1
        )
        reference_stack = self.bse_data[self.bse_mode.get()]
        print("SRC shape:", image_stack.shape)
        print("DST shape:", reference_stack.shape)

        # Get points
        src_points = []
        dst_points = []
        for k in self.points["ebsd"].keys():
            src = np.array(self.points["ebsd"][k]).reshape(-1, 2)
            dst = np.array(self.points["bse"][k]).reshape(-1, 2)
            src = np.hstack((np.ones((src.shape[0], 1)) * k, src))
            dst = np.hstack((np.ones((dst.shape[0], 1)) * k, dst))
            src_points.extend(src)
            dst_points.extend(dst)
        src_points = np.array(src_points).astype(int)
        dst_points = np.array(dst_points).astype(int)

        # Transform the image stack
        if "tps" in mode.lower():
            corrected_stack = warping.transform_image_stack(
                image_stack,
                src_points,
                dst_points,
                dst_img_shape,
                mode,
                order=0,
                size=dst_img_shape,
            )
        else:
            corrected_stack = warping.transform_image_stack(
                image_stack, src_points, dst_points, dst_img_shape, mode, order=0
            )

        # Crop to the center of the corrected image if desired
        if crop_choice == "src":
            print("Cropping to match distorted grid")
            dummy = corrected_stack[:, :, :, -1]
            zc, yc, xc = (
                np.array(np.where(dummy)).reshape(3, -1).T.mean(axis=0).astype(int)
            )
            rslc, cslc = self._get_cropping_slice(
                (yc, xc), src_img_shape, dst_img_shape
            )
            reference_stack = reference_stack[:, rslc, cslc]
            corrected_stack = corrected_stack[:, rslc, cslc]
        elif crop_choice == "dst":
            print("Cropping to match destination grid")
            corrected_stack = corrected_stack[:, : dst_img_shape[0], : dst_img_shape[1]]
        else:
            print("Not cropping, leaving destination and warped source as is.")

        # Remove the last channel (the one with the ones)
        corrected_stack = corrected_stack[:, :, :, :-1]

        print("Reference shape:", reference_stack.shape)
        print("Corrected shape:", corrected_stack.shape)

        # Unpack the image stack
        output = {}
        i = 0
        for key in self.ebsd_data.keys():
            n_channels = self.ebsd_data[key].shape[-1]
            output[key] = corrected_stack[:, :, :, i : i + n_channels]
            i += n_channels

        output["reference"] = reference_stack

        return output

    def clahe(self, viewer):
        if viewer == "ebsd":
            if self.clahe_active_ebsd:
                self.clahe_active_ebsd = False
                self.clahe_ebsd_b["text"] = "Apply CLAHE to EBSD"
                self._update_viewers()
            else:
                self.clahe_active_ebsd = True
                self.clahe_ebsd_b["text"] = "Remove CLAHE from EBSD"
                self._update_viewers()
        elif viewer == "bse":
            if self.clahe_active_bse:
                self.clahe_active_bse = False
                self.clahe_bse_b["text"] = "Apply CLAHE to BSE"
                self._update_viewers()
            else:
                self.clahe_active_bse = True
                self.clahe_bse_b["text"] = "Remove CLAHE from BSE"
                self._update_viewers()

    ### Viewer stuff
    def _resize(self, pos, scale):
        """Resizes the image in the viewer"""
        if pos == "ebsd":
            self.ebsd.delete("all")
        else:
            self.bse.delete("all")
        self._update_viewers()

    def _update_viewers(self, *args):
        i = self.slice_num.get()
        key = self.ebsd_mode.get()
        ebsd_im = np.squeeze(self.ebsd_data[key][i])
        key = self.bse_mode.get()
        bse_im = np.squeeze(self.bse_data[key][i])
        if self.clahe_active_bse:
            bse_im = CLAHE(bse_im)
        if self.clahe_active_ebsd:
            if ebsd_im.ndim != 3 or (ebsd_im.ndim == 3 and ebsd_im.shape[-1] == 1):
                ebsd_im = CLAHE(ebsd_im)
        # Convert to grayscale if channels is more than 3
        if len(ebsd_im.shape) == 3 and ebsd_im.shape[-1] > 3:
            ebsd_im = np.sqrt(np.sum(ebsd_im**2, axis=-1))
        # Resize the images
        scale_ebsd = int(self.resize_vars["ebsd"].get()) / 100
        scale_bse = int(self.resize_vars["bse"].get()) / 100
        if scale_ebsd != 1:
            if ebsd_im.ndim == 3:
                _t = torch.tensor(np.transpose(ebsd_im, (2, 0, 1))).unsqueeze(0).float()
            else:
                _t = torch.tensor(ebsd_im).unsqueeze(0).unsqueeze(0).float()
            ebsd_im = (
                RESIZE(
                    _t,
                    (
                        int(ebsd_im.shape[0] * scale_ebsd),
                        int(ebsd_im.shape[1] * scale_ebsd),
                    ),
                    InterpolationMode.NEAREST,
                )
                .detach()
                .squeeze()
                .numpy()
            )
            if ebsd_im.ndim == 3:
                ebsd_im = np.transpose(ebsd_im, (1, 2, 0))
        if scale_bse != 1:
            if bse_im.ndim == 3:
                _t = torch.tensor(np.transpose(bse_im, (2, 0, 1))).unsqueeze(0).float()
            else:
                _t = torch.tensor(bse_im).unsqueeze(0).unsqueeze(0).float()
            bse_im = (
                RESIZE(
                    _t,
                    (
                        int(bse_im.shape[0] * scale_bse),
                        int(bse_im.shape[1] * scale_bse),
                    ),
                    InterpolationMode.NEAREST,
                )
                .detach()
                .squeeze()
                .numpy()
            )
            if bse_im.ndim == 3:
                bse_im = np.transpose(bse_im, (1, 2, 0))
        # Ensure dtype is uint8, accounting for RGB or greyscale
        with np.errstate(divide="ignore", invalid="ignore"):
            if ebsd_im.ndim == 3 and ebsd_im.dtype != np.uint8:
                self.ebsd_im = np.around(
                    255
                    * (ebsd_im - ebsd_im.min(axis=(0, 1)))
                    / (ebsd_im.max(axis=(0, 1)) - ebsd_im.min(axis=(0, 1))),
                    0,
                ).astype(np.uint8)
            elif ebsd_im.ndim == 2 and ebsd_im.dtype != np.uint8:
                self.ebsd_im = np.around(
                    255 * (ebsd_im - ebsd_im.min()) / (ebsd_im.max() - ebsd_im.min()), 0
                ).astype(np.uint8)
            else:
                self.ebsd_im = ebsd_im
            if bse_im.ndim == 3 and bse_im.dtype != np.uint8:
                self.bse_im = np.around(
                    255
                    * (bse_im - bse_im.min(axis=(0, 1)))
                    / (bse_im.max(axis=(0, 1)) - bse_im.min(axis=(0, 1))),
                    0,
                ).astype(np.uint8)
            elif bse_im.ndim == 2 and bse_im.dtype != np.uint8:
                self.bse_im = np.around(
                    255 * (bse_im - bse_im.min()) / (bse_im.max() - bse_im.min()), 0
                ).astype(np.uint8)
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
        self.bse_im_ppm = self._photo_image(self.bse_im, channels=1)
        self.bse.create_image(0, 0, anchor="nw", image=self.bse_im_ppm)
        self.bse.config(scrollregion=self.bse.bbox("all"))
        # EBSD
        channels = 3 if self.ebsd_im.ndim == 3 else 1
        self.ebsd_im_ppm = self._photo_image(self.ebsd_im, channels=channels)
        self.ebsd.create_image(0, 0, anchor="nw", image=self.ebsd_im_ppm)
        self.ebsd.config(scrollregion=self.ebsd.bbox("all"))

    def _photo_image(self, image: np.ndarray, channels: int = 1):
        """Creates a PhotoImage object that plays nicely with a tkinter canvas for viewing purposes."""
        ### TODO: There is an error here with rgb data, not sure why "truncated PPM data", probably need to scale values
        if channels == 1:
            height, width = image.shape
            data = (
                f"P5 {width} {height} 255 ".encode() + image.astype(np.uint8).tobytes()
            )
        else:
            height, width = image.shape[:2]
            ppm_header = f"P6 {width} {height} 255 ".encode()
            data = ppm_header + image.tobytes()
        return tk.PhotoImage(width=width, height=height, data=data, format="PPM")

    def _style_call(self, style="dark"):
        if style == "dark":
            self.bg = "#333333"
            self.fg = "#ffffff"
            self.hl = "#007fff"
            self.tk.call("source", r"./theme/dark.tcl")
            s = ttk.Style(self)
            s.theme_use("azure-dark")
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
        elif style == "light":
            self.bg = "#ffffff"
            self.fg = "#000000"
            self.hl = "#007fff"
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


class ProgressWindow(tk.Toplevel):
    """displays progress"""

    def __init__(self, text):
        super().__init__()
        self.f = ttk.Frame(self)
        self.f.grid(row=0, column=0, sticky="nsew")
        self.note = text
        self.p_label = ttk.Label(
            self.f, text=self.note, font=("", 12, "bold"), anchor=tk.CENTER
        )
        self.p_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)
        self.progress = ttk.Progressbar(
            self.f, orient="horizontal", length=300, mode="indeterminate"
        )
        self.progress.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.progress.start()


class ArbitraryMessageBox(object):
    # Adapted from
    # https://stackoverflow.com/questions/29619418/how-to-create-a-custom-messagebox-using-tkinter-in-python-with-changing-message
    def __init__(
        self, master, title="Mess", msg="", options=["OK"], orientation="horizontal"
    ):
        # Required Data of Init Function
        self.master = master
        self.title = title
        self.msg = msg
        self.options = options
        n_options = len(self.options)
        self.choice = ""

        # Just the colors for my messagebox
        self.bgcolor = self.master.bg
        self.bgcolor2 = self.master.hl
        self.textcolor = self.master.fg

        # Creating Dialogue for messagebox
        self.root = tk.Toplevel(self.master)
        self.root.title(self.title)
        # self.root.geometry("300x120+100+100")
        self.root.rowconfigure(0, weight=2)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(list(range(n_options)), weight=1)

        # Setting Background color of Dialogue
        self.root.config(bg=self.bgcolor)

        # Creating Label For message
        self.msg = tk.Label(self.root, text=msg)
        if orientation == "horizontal":
            self.msg.grid(row=0, column=0, columnspan=n_options, padx=10, pady=10)
        elif orientation == "vertical":
            self.msg.grid(row=0, column=0, padx=10, pady=10)

        # Creating Buttons
        if orientation == "horizontal":
            rows = [1 for _ in range(n_options)]
            columns = list(range(n_options))
        elif orientation == "vertical":
            rows = list(range(1, n_options + 1))
            columns = [0 for _ in range(n_options)]
        for i in range(n_options):
            B = tk.Button(
                self.root, text=self.options[i], command=lambda i=i: self.click(i)
            )
            B.grid(row=rows[i], column=columns[i], padx=10, pady=10, sticky="nsew")

        # Making MessageBox Visible
        self.root.resizable(0, 0)
        self.root.protocol("WM_DELETE_WINDOW", self.closed)
        self.root.wait_window()

    # Function on Closeing MessageBox
    def closed(self):
        self.root.destroy()
        self.choice = False

    # Function on pressing B1
    def click(self, i):
        self.root.destroy()
        self.choice = i


if __name__ == "__main__":
    app = App()
    app.mainloop()
