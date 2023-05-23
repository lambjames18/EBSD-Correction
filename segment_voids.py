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

    def _update_viewers(self, *args):
        i = self.slice_num.get()
        key = self.ebsd_mode.get()
        bse_im = self.bse_imgs[i]

        # Check the dtype of the BSE image, if they are a float, convert to uint8
        if bse_im.dtype == np.uint8:
            self.bse_im = bse_im
        else:
            bse_im = bse_im - bse_im.min()
            self.bse_im = np.around(255 * bse_im / bse_im.max(), 0).astype(np.uint8)

        # Update the images and draw points
        self._update_imgs()

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
        if channels == 1:
            height, width = image.shape
            data = f"P5 {width} {height} 255 ".encode() + image.astype(np.uint8).tobytes()
        else:
            height, width = image.shape[:2]
            ppm_header = f"P6 {width} {height} 255 ".encode()
            data = ppm_header + image.tobytes()
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
        # Correct for cropped EBSD data
        self._bse_mask = (slice(None), slice(None))
        size_diff = np.array(self.bse_imgs.shape) - np.array(self.ebsd_cStack.shape)
        if size_diff[1] > 0:
            print(f"{size_diff[1]=}")
            start = size_diff[1] // 2
            end = -(size_diff[1] - start)
            self._bse_mask = (slice(start, end), self._bse_mask[1])
        if size_diff[2] > 0:
            print(f"{size_diff[2]=}")
            start = size_diff[2] // 2
            end = -(size_diff[2] - start)
            self._bse_mask = (self._bse_mask[0], slice(start, end))
        # Generate the figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        im0 = self.bse_imgs[self.slice_num.get()][self._bse_mask]
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
            im0 = self.bse_imgs[val][self._bse_mask]
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
        # self.BSE_DIR = "D:/Research/CoNi_16/Data/3D/BSE/small/"
        self.BSE_DIR = "D:/Research/Ta/Data/3D/AMSpall/BSE/small/"
        # self.EBSD_DIR = "D:/Research/CoNi_16/Data/3D/CoNi16_aligned.dream3d"
        self.EBSD_DIR = "D:/Research/Ta/Data/3D/AMSpall/TaAMS_Stripped.dream3d"
        # self.folder = "D:/Research/scripts/Alignment/CoNi16_3D/"
        self.folder = "D:/Research/scripts/Alignment/TaAMSpalled/"
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
