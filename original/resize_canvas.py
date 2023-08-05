# Author: James Lamb

# This script is a simple tkinter UI that has an image viewer (on a canvas) and a dropdown menu to resize the image to a given percentage.
# The image is resized using the Pillow library.
# It also prints the coordinates of the mouse when the mouse is clicked on the canvas.
# Use ttk for a more modern look.
# Use scrollbars to allow scrolling through the image when it is larger than the canvas.

import tkinter as tk
from tkinter import ttk
# from PIL import Image, ImageTk
import numpy as np
from skimage import io, transform

class App(ttk.Frame):
    def __init__(self, master, *args, **kwargs) -> None:
        super().__init__(master, *args, **kwargs)
        self.master = master
        self.master.title("Resize Image")
        self.master.bind("<Escape>", lambda e: self.master.destroy())

        self.resize_options = [25, 50, 75, 100, 125, 150, 175, 200]
        self.resize_var = tk.StringVar(self.master)
        self.resize_var.set(self.resize_options[3])
        self.resize_var.trace("w", lambda *args: self._resize(self.resize_var.get()))
        self.resize_dropdown = ttk.Combobox(self.master, textvariable=self.resize_var, values=self.resize_options, state="readonly")
        # self.resize_dropdown = tk.OptionMenu(self.master, self.resize_var, *self.resize_options, command=self._resize)
        self.resize_dropdown.grid(row=0, column=0, sticky=tk.W)

        self.mouse_coords_raw = tk.StringVar(self.master)
        self.mouse_coords_raw.set("Mouse coordinates (raw): ")
        self.mouse_coords_raw_label = ttk.Label(self.master, textvariable=self.mouse_coords_raw)
        self.mouse_coords_raw_label.grid(row=1, column=0, sticky=tk.W)
        self.mouse_coords_scaled = tk.StringVar(self.master)
        self.mouse_coords_scaled.set("Mouse coordinates (scaled): ")
        self.mouse_coords_scaled_label = ttk.Label(self.master, textvariable=self.mouse_coords_scaled)
        self.mouse_coords_scaled_label.grid(row=2, column=0, sticky=tk.W)

        self.im_array = io.imread("test_data/BSE_rescaled.tif", as_gray=True)
        self.im_array_rescaled = io.imread("test_data/BSE_rescaled.tif", as_gray=True)
        self.im_ppm = self._photo_image(self.im_array)

        self.canvas = tk.Canvas(self.master, width=self.im_array.shape[1], height=self.im_array.shape[0])
        self.canvas.grid(row=3, column=0, columnspan=3, sticky=tk.NSEW)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.create_image(0, 0, anchor="nw", image=self.im_ppm)
        self.canvas.image = self.im_ppm
        self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))

        self.hbar = ttk.Scrollbar(self.master, orient=tk.HORIZONTAL, command=self.canvas.xview, cursor="sb_h_double_arrow")
        self.hbar.grid(row=4, column=0, columnspan=3, sticky=tk.EW)
        self.vbar = ttk.Scrollbar(self.master, orient=tk.VERTICAL, command=self.canvas.yview, cursor="sb_v_double_arrow")
        self.vbar.grid(row=3, column=3, sticky=tk.NS)
        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

    def _on_click(self, event):
        scale = int(self.resize_var.get()) / 100
        x = int(self.canvas.canvasx(event.x))
        y = int(self.canvas.canvasy(event.y))
        x_s = int(np.around(x / scale, 0))
        y_s = int(np.around(y / scale, 0))
        print(x, y, x_s, y_s)
        self.mouse_coords_raw.set(f"Mouse coordinates (raw): ({x}, {y})")
        self.mouse_coords_scaled.set(f"Mouse coordinates (scaled): ({x_s}, {y_s})")

    def _resize(self, value: str):
        """Resizes the image to the given percentage."""
        self.im_array_rescaled = transform.resize(self.im_array, (self.im_array.shape[0] * int(value) // 100, self.im_array.shape[1] * int(value) // 100), anti_aliasing=True)
        self.im_array_rescaled = np.around(255 * (self.im_array_rescaled - self.im_array_rescaled.min()) / (self.im_array_rescaled.max() - self.im_array_rescaled.min())).astype(np.uint8)
        self.im_ppm = self._photo_image(self.im_array_rescaled)
        self.canvas.create_image(0, 0, anchor="nw", image=self.im_ppm)
        self.canvas.image = self.im_ppm
        self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))

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

if __name__ == "__main__":
    root = tk.Tk()
    app = App(master=root)
    app.mainloop()
