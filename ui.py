"""
Author: James Lamb

UI for running distortion correction
"""

# Python packages
import os
import tkinter as tk
from tkinter import filedialog

# 3rd party packages
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
        self.select_folder_popup()
        self.deiconify()
        # frames
        #frame_w = 1920
        #frame_h = 1080
        #self.geometry(f"{frame_w}x{frame_h}")
        #self.resizable(False, False)
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
        view_pts = tk.Checkbutton(self.top,
                             text="Show points",
                             variable=self.show_points,
                             onvalue=1,
                             offvalue=0,
                             command=self._show_points,
                             fg='black')
        view_pts.grid(row=0, column=0)
        # setup viewer
        self.bse = tk.Canvas(self.viewer)
        self.bse.grid(row=0, column=0)
        self.bse.bind("<Button 1>", lambda arg: self.coords("bse", arg))
        self.ebsd = tk.Canvas(self.viewer)
        self.ebsd.grid(row=0, column=1)
        self.ebsd.bind("<Button 1>", lambda arg: self.coords("ebsd", arg))
        self._update_imgs()
        # setup bot
        other = tk.Button(self.bot, text="Apply Correction", command=self.apply, fg='black')
        other.grid(row=0, column=0)
        # handle points
        self._read_points()
        

    def select_folder_popup(self):
        self.w = tk.Toplevel(self)
        frame_w = 1920 // 6
        frame_h = 1080 // 5
        self.w.geometry(f"{frame_w}x{frame_h}")
        self.resizable(False, False)
        des = tk.Label(self.w, text="Select location to save alignment\nimages/points/solutions", fg='black')
        des.pack(fill="both", expand=1)
        but = tk.Button(self.w, text="Select directory", command=self._get_dir, fg='black')
        but.pack(fill="both", expand=1)
        close = tk.Button(self.w, text="Close (select CWD)", command=self.w.destroy, fg='black')
        close.pack(fill="both", expand=1)
        self.wait_window(self.w)
        # EBSD
        self.ebsd_path = os.path.join(self.folder, "ebsd.tif")
        ebsd_im = io.imread(self.ebsd_path)
        if ebsd_im.dtype != np.uint8:
            self.ebsd_im = np.around(255*ebsd_im/ebsd_im.max(), 0).astype(np.uint8)
        else:
            self.ebsd_im = ebsd_im
        # BSE
        self.bse_path = os.path.join(self.folder, "bse.tif")
        bse_im = io.imread(self.bse_path)
        if bse_im.dtype != np.uint8:
            self.bse_im = np.around(255*bse_im/bse_im.max(), 0).astype(np.uint8)
        else:
            self.bse_im = bse_im

    def coords(self, pos, event):
        self.points[pos].append([event.x, event.y])
        path = os.path.join(self.folder, f"ctr_pts_{pos}.txt")
        with open(path, "a", encoding="utf8") as output:
            output.write(f"{event.x} {event.y}\n")
        self._show_points()
    
    def apply(self):
        referencePoints = os.path.join(self.folder, "Ctr_pts_bse.txt")
        distortedPoints = os.path.join(self.folder, "Ctr_pts_ebsd.txt")
        align = core.Alignment(referencePoints, distortedPoints, algorithm="TPS")
        align.get_solution(self.bse_path)
        align.apply(self.ebsd_im)
        self._interactive_view()
        plt.close("all")

    def _show_points(self):
        if self.show_points.get() == 1:
            for i, p in enumerate(self.points['ebsd']):
                self.ebsd.create_oval(p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1, activeoutline="#476042")
                self.ebsd.create_text(p[0] + 3, p[1] + 3, anchor=tk.NW, text=i)
            for i, p in enumerate(self.points['bse']):
                self.bse.create_oval(p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1, activeoutline="#476042")
                self.bse.create_text(p[0] + 3, p[1] + 3, anchor=tk.NW, text=i)
        else:
            print("Removing points")
            self.ebsd.delete()
            self.bse.delete()
            self._update_imgs()

    def _update_imgs(self):
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
        height, width = image.shape
        data = f"P5 {width} {height} 255 ".encode() + image.astype(np.uint8).tobytes()
        return tk.PhotoImage(width=width, height=height, data=data, format="PPM")

    def _read_points(self):
        bse_path = os.path.join(self.folder, "ctr_pts_bse.txt")
        ebsd_path = os.path.join(self.folder, "ctr_pts_ebsd.txt")
        try:
            bse_pts = list(np.loadtxt(bse_path))
            ebsd_pts = list(np.loadtxt(ebsd_path))
            self.points = {"ebsd": ebsd_pts, "bse":bse_pts}
            self._show_points()
        except FileNotFoundError:
            self.points = {"ebsd": [], "bse": []}
    
    def _get_dir(self):
        self.folder = filedialog.askdirectory(initialdir=self.folder, title="Select folder")
        self.w.destroy()
        
    def _interactive_view(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.set_title("TPS Output")
        im0 = self.bse_im
        im1 = io.imread(os.path.join(self.folder, "TPS_out.tif"))
        max_r = im0.shape[0]
        max_c = im0.shape[1]
        alphas = np.ones(im0.shape)
        # Show images
        ax.imshow(im1, cmap="gray")
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

        # Enable update functions
        row_slider.on_changed(update_row)
        col_slider.on_changed(update_col)
        plt.show()
        


if __name__ == "__main__":
    app = App()
    app.mainloop()
