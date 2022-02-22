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
from skimage import io
import imageio

# Local files
import core


class App(tk.Tk):
    def __init__(
        self,
        screenName=None,
        baseName=None,
    ) -> None:
        super().__init__(screenName, baseName)
        self.points = {}
        # handle main folder
        self.update_idletasks()
        self.withdraw()
        self.folder = os.getcwd()
        self.select_folder_popup()
        self.deiconify()
        # frames
        self.control = tk.Frame(self)
        self.viewer = tk.Frame(self)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)
        self.control.grid(row=0, column=0, sticky="nsew")
        self.viewer.grid(row=0, column=1, sticky="nsew")
        # setup control
        ebsd_b = tk.Button(
            self.control, text="Select Distorted", command=lambda: self.get_file("ebsd")
        )
        ebsd_b.grid(row=0, column=0, columnspan=2, sticky="ew")
        bse_b = tk.Button(self.control, text="Select Control", command=lambda: self.get_file("bse"))
        bse_b.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.view_var = tk.StringVar()
        self.view_options = ["None"]
        view_label = tk.Label(self.control, text="Viewing: ")
        view_label.grid(row=2, column=0, sticky="ew")
        self.view_picker = tk.OptionMenu(self.control, self.view_var, ())
        self.view_picker.grid(row=2, column=1, sticky="ew")
        self.view_var.trace("w", self.update_view_selection)
        self.show_points = False
        points = tk.Button(self.control, text="Show points", command=self.points_option)
        points.grid(row=3, column=0, columnspan=2, sticky="ew")
        # points_clear = tk.Button(self.control, text="Clear points", command=self.clear_points)
        # points_clear.grid(row=4, column=0, columnspan=2, sticky="ew")
        # setup viewer
        self.c = tk.Canvas(self.viewer)
        self.c.grid(row=1, column=0)
        self.c.bind("<Button 1>", self.coords)

    def select_folder_popup(self):
        self.w = tk.Toplevel(self)
        frame_w = 1920 // 6
        frame_h = 1080 // 5
        self.w.geometry(f"{frame_w}x{frame_h}")
        self.resizable(False, False)
        des = tk.Label(self.w, text="Select location to save alignment images/points/solutions")
        des.pack(fill="both", expand=1)
        but = tk.Button(self.w, text="Select directory", command=self._get_dir)
        but.pack(fill="both", expand=1)
        close = tk.Button(self.w, text="Close (select CWD)", command=self.w.destroy)
        close.pack(fill="both", expand=1)
        self.wait_window(self.w)

    def update_view_selection(self, *args):
        pick = self.view_var.get()
        if pick == "EBSD":
            self.im = self.ebsd_im
        elif pick == "BSE":
            self.im = self.bse_im
        self._update_im()

    def get_file(self, pick):
        if pick == "ebsd":
            self.ebsd_path = filedialog.askopenfilename(
                initialdir=self.folder, title="Select Distorted"
            )
            self.view_options.append("EBSD")
            self.ebsd_im = io.imread(self.ebsd_path)
            self.points["EBSD"] = []
        elif pick == "bse":
            self.bse_path = filedialog.askopenfilename(
                initialdir=self.folder, title="Select Control"
            )
            self.view_options.append("BSE")
            self.bse_im = io.imread(self.bse_path)
            self.points["BSE"] = []
        self._update_option_menu()

    def coords(self, event):
        self.points[self.view_var.get()].append([event.x, event.y])
        if self.show_points == True:
            self._show_points()

    def points_option(self):
        if self.show_points == False:
            self.show_points = True
            self._gen_points_im()
        else:
            self.show_points = False
            self._clear_points()

    def _clear_points(self):
        self.c.delete()
        self.update_view_selection()

    def _show_points(self):
        pick = self.view_var.get()
        for point in self.points[pick]:
            self.c.create_oval(
                point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1, activeoutline="#476042"
            )

    def _update_option_menu(self):
        menu = self.view_picker["menu"]
        menu.delete(0, "end")
        for string in self.view_options:
            menu.add_command(label=string, command=lambda value=string: self.view_var.set(value))

    def _update_im(self):
        self.c["width"] = self.im.shape[1]
        self.c["height"] = self.im.shape[0]
        self.im_ppm = self._photo_image(self.im)
        self.c.create_image(0, 0, anchor="nw", image=self.im_ppm)

    def _photo_image(self, image: np.ndarray):
        height, width = image.shape
        data = f"P5 {width} {height} 255 ".encode() + image.astype(np.uint8).tobytes()
        return tk.PhotoImage(width=width, height=height, data=data, format="PPM")

    def _get_dir(self):
        self.folder = filedialog.askdirectory(initialdir=self.folder, title="Select folder")
        self.w.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
