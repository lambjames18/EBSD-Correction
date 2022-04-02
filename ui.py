"""
Author: James Lamb

UI for running distortion correction
"""

# Python packages
import os
import tkinter as tk
from tkinter import filedialog
from threading import Thread

# 3rd party packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
import imageio

# Local files
import core


class App(tk.Tk):
    def __init__(self, screenName=None, baseName=None):
        super().__init__(screenName, baseName)
        # handle main folder
        self.update_idletasks()
        self.withdraw()
        self.folder = os.getcwd()
        # self.select_folder_popup()
        self._easy_start()
        self.deiconify()
        # frames
        # frame_w = 1920
        # frame_h = 1080
        # self.geometry(f"{frame_w}x{frame_h}")
        # self.resizable(False, False)
        self.top = tk.Frame(self)
        self.viewer = tk.Frame(self)
        self.bot = tk.Frame(self)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=3)
        self.rowconfigure(2, weight=1)
        self.top.grid(row=0, column=0, sticky="nsew")
        self.viewer.grid(row=1, column=0, sticky="nsew")
        self.bot.grid(row=2, column=0, sticky="nsew")
        self.viewer.rowconfigure(0, weight=1)
        self.viewer.columnconfigure(0, weight=1)
        self.viewer.columnconfigure(1, weight=1)
        # setup top
        self.show_points = tk.IntVar()
        self.show_points.set(1)
        view_pts = tk.Checkbutton(
            self.top,
            text="Show points",
            variable=self.show_points,
            onvalue=1,
            offvalue=0,
            command=self._show_points,
            fg="black",
        )
        view_pts.grid(row=0, column=0)
        self.slice_picker = tk.Spinbox(
            self.top,
            textvariable=self.slice_num,
            from_=self.slice_min,
            to=self.slice_max,
            command=self._update_viewers,
        )
        self.slice_picker.grid(row=0, column=1)
        self.ebsd_picker = tk.OptionMenu(
            self.top, self.ebsd_mode, *self.ebsd_mode_options, command=self._update_viewers
        )
        self.ebsd_picker.grid(row=0, column=2)
        # setup viewer
        self.bse = tk.Canvas(self.viewer)
        self.bse.grid(row=0, column=0)
        self.bse.bind("<Button 1>", lambda arg: self.coords("bse", arg))
        self.ebsd = tk.Canvas(self.viewer)
        self.ebsd.grid(row=0, column=1)
        self.ebsd.bind("<Button 1>", lambda arg: self.coords("ebsd", arg))
        # handle points
        self.all_points = {}
        self.current_points = {"ebsd": [], "bse": []}
        self._read_points()
        # Update viewers
        self._update_viewers()
        # setup bot
        tps = tk.Button(
            self.bot, text="View TPS Correction", command=lambda: self.apply("TPS"), fg="black"
        )
        tps.grid(row=0, column=0)
        lr = tk.Button(
            self.bot, text="View LR Correction", command=lambda: self.apply("LR"), fg="black"
        )
        lr.grid(row=0, column=1)
        tps_stack = tk.Button(
            self.bot,
            text="Apply TPS correction to stack",
            fg="black",
            command=lambda: self.apply_3D("TPS"),
        )
        tps_stack.grid(row=0, column=2)
        lr_stack = tk.Button(
            self.bot,
            text="Apply LR correction to stack",
            fg="black",
            command=lambda: self.apply_3D("LR"),
        )
        lr_stack.grid(row=0, column=3)
        ex_ctr_pt_ims = tk.Button(
            self.bot, text="Export control point images", fg="black", command=self.export_CP_imgs
        )
        ex_ctr_pt_ims.grid(row=0, column=4)

    def select_folder_popup(self):
        self.w = tk.Toplevel(self)
        frame_w = 1920 // 6
        frame_h = 1080 // 5
        self.w.geometry(f"{frame_w}x{frame_h}")
        self.resizable(False, False)
        for i in range(5):
            self.w.rowconfigure(i, weight=1)
        self.w.columnconfigure(0, weight=1)
        des = tk.Label(self.w, text="Select relevant folders/files", fg="black")
        des.grid(row=0, column=0, sticky="ew")
        folder = tk.Button(
            self.w,
            text="Select folder for control points",
            command=self._get_FOLDER_dir,
            fg="black",
        )
        folder.grid(row=1, column=0, sticky="nsew")
        self.w.d3d = tk.Button(
            self.w, text="Select Dream3d file", command=self._get_EBSD_dir, fg="black"
        )
        self.w.d3d.grid(row=2, column=0, sticky="nsew")
        self.w.d3d["state"] = "disabled"
        self.w.bse = tk.Button(
            self.w, text="Select BSE folder", command=self._get_BSE_dir, fg="black"
        )
        self.w.bse.grid(row=3, column=0, sticky="nsew")
        self.w.bse["state"] = "disabled"
        self.w.close = tk.Button(
            self.w,
            text="Close (reading in data will take a few moments)",
            command=self._startup_items,
            fg="black",
        )
        self.w.close["state"] = "disabled"
        self.w.close.grid(row=4, column=0, sticky="nsew")
        self.wait_window(self.w)

    def coords(self, pos, event):
        """Responds to a click on an image. Redraws the images after the click. Also saves the click location in a file."""
        i = self.slice_num.get()
        self.current_points[pos].append([event.x, event.y])
        self.all_points[i] = self.current_points
        path = os.path.join(self.folder, f"ctr_pts_{i}_{pos}.txt")
        with open(path, "a", encoding="utf8") as output:
            output.write(f"{event.x} {event.y}\n")
        self._show_points()

    def apply(self, algo="TPS"):
        """Applies the correction algorithm and calls the interactive view"""
        referencePoints = np.array(self.current_points["bse"])
        distortedPoints = np.array(self.current_points["ebsd"])
        align = core.Alignment(referencePoints, distortedPoints, algorithm=algo)
        if algo == "TPS":
            save_name = os.path.join(self.folder, "TPS_mapping.npy")
            align.get_solution(l=self.bse_im.shape, solutionFile=save_name)
        elif algo == "LR":
            save_name = os.path.join(self.folder, "LR_mapping.npy")
            align.get_solution(degree=3, solutionFile=save_name)
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

    def export_CP_imgs(self):
        i = self.slice_num.get()
        pts = np.array(self.current_points["ebsd"])
        self._save_CP_img(f"{i}_ebsd", self.ebsd_im, pts, "gray", "#c2344e")
        pts = np.array(self.current_points["bse"])
        self._save_CP_img(f"{i}_bse", self.bse_im, pts, "gray", "#34c295")
        print("Control point images exported successfully.")

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

    def _update_viewers(self, *args):
        i = self.slice_num.get()
        key = self.ebsd_mode.get()
        bse_im = self.bse_imgs[i]
        ebsd_im = self.ebsd_data[key][i]
        if ebsd_im.shape[-1] == 1:
            ebsd_im = ebsd_im[:, :, 0]
        else:
            ebsd_im = np.sqrt(np.sum(ebsd_im ** 2, axis=2))

        if ebsd_im.dtype == np.uint8:
            self.ebsd_im = ebsd_im
        else:
            self.ebsd_im = np.around(255 * ebsd_im / ebsd_im.max(), 0).astype(np.uint8)

        self.bse_im = np.around(255 * bse_im / bse_im.max(), 0).astype(np.uint8)
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
            pc = "#c2344e"
            for i, p in enumerate(self.current_points["ebsd"]):
                self.ebsd.create_oval(p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1, width=2, outline=pc)
                self.ebsd.create_text(
                    p[0] + 3, p[1] + 3, anchor=tk.NW, text=i, fill=pc, font=("", 10, "bold")
                )
            pc = "#34c295"
            for i, p in enumerate(self.current_points["bse"]):
                self.bse.create_oval(p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1, width=2, outline=pc)
                self.bse.create_text(
                    p[0] + 3, p[1] + 3, anchor=tk.NW, text=i, fill=pc, font=("", 10, "bold")
                )
        else:
            self.ebsd.delete()
            self.bse.delete()
            self._update_imgs()

    def _update_imgs(self):
        """Updates the images in the viewers"""
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
            bse_pts = list(np.loadtxt(bse_files[i]))
            ebsd_pts = list(np.loadtxt(ebsd_files[i]))
            self.all_points[key] = {"ebsd": ebsd_pts, "bse": bse_pts}
        if self.slice_num.get() in self.all_points.keys():
            self.current_points = self.all_points[self.slice_num.get()]
        else:
            self.current_points = {"ebsd": [], "bse": []}
        self._show_points()

    def _startup_items(self):
        print("Running startup")
        self._open_BSE_stack(self.BSE_DIR)
        self._read_h5(self.EBSD_DIR)
        self.ebsd_mode_options = list(self.ebsd_data.keys())
        self.ebsd_mode = tk.StringVar()
        self.ebsd_mode.set(self.ebsd_mode_options[0])
        self.slice_min = 0
        self.slice_max = self.ebsd_data[self.ebsd_mode_options[0]].shape[0] - 1
        self.slice_num = tk.IntVar()
        self.slice_num.set(self.slice_min)
        self.w.destroy()
        print("Startup complete")

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
        self.bse_imgs = np.array(
            [io.imread(os.path.join(imgs_path, p), as_gray=True) for p in paths]
        )

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
            ax=axrow, label="Y pos", valmin=0, valmax=max_r, valinit=max_r, orientation="vertical"
        )
        col_slider = Slider(
            ax=axcol, label="X pos", valmin=0, valmax=max_c, valinit=max_c, orientation="horizontal"
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
                orientation="vertical",
            )
            slice_slider.on_changed(change_image)
        plt.show()

    def _easy_start(self):
        self.BSE_DIR = "D:/Research/CoNi_16/Data/3D/BSE/small/"
        self.EBSD_DIR = "D:/Research/CoNi_16/Data/3D/CoNi16_aligned.dream3d"
        self.folder = "D:/Research/scripts/Alignment/CoNi16_3D/"
        self._open_BSE_stack(self.BSE_DIR)
        self._read_h5(self.EBSD_DIR)
        self.ebsd_mode_options = list(self.ebsd_data.keys())
        self.ebsd_mode = tk.StringVar()
        self.ebsd_mode.set(self.ebsd_mode_options[0])
        self.slice_min = 0
        self.slice_max = self.ebsd_data[self.ebsd_mode_options[0]].shape[0] - 1
        self.slice_num = tk.IntVar()
        self.slice_num.set(self.slice_min)


if __name__ == "__main__":
    app = App()
    app.mainloop()
