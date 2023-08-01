import os
import tkinter as tk, ttk, filedialog

import numpy as np
import skimage.transform as tf

class PrepareData(object):
    """This is a class to handle resolution differences between distorted and reference images.
    Need to make a toplevel on the parent that asks for an EBSD resolution and a reference HFW (the HFW is used to calculate the resolution of the reference image).
    Then upon clicking ok, downscale (tf.rescale) the reference image to the EBSD resolution."""
    def __init__(self, parent, d_image, r_image) -> None:
        self.parent = parent
        self.d_image = d_image
        self.r_image = r_image
        self.d_res = None
        self.r_res = None
        self.w = tk.Toplevel(self.parent)
        self.clean_exit = False
        self.w = tk.Toplevel(parent)
        self.w.title(f"Prepare Data")
        # Create layout
        self.w.rowconfigure((0, 1), weight=10)
        self.w.columnconfigure((0, 1), weight=1)
        self.left = ttk.LabelFrame(self.w, text="Distorted", relief="groove", borderwidth=5, padding=5, labelanchor="n")
        self.left.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.right = ttk.LabelFrame(self.w, text="Control", relief="groove", borderwidth=5, padding=5, labelanchor="n")
        self.right.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.bot = ttk.Frame(self.w)
        self.bot.grid(row=1, column=0, columnspan=2, sticky="nse")
        # Left side
        self.d_res_label = ttk.Label(self.left, text="Distorted Resolution (µm/px):")
        self.d_res_label.grid(row=0, column=0, sticky="ns", pady=2)
        self.d_res_entry = ttk.Entry(self.left)
        self.d_res_entry.grid(row=1, column=0, sticky="ns", pady=2)
        self.d_res_entry.insert(0, "")
        # Right side
        self.r_res_label = ttk.Label(self.right, text="Reference Resolution (µm/px):")
        self.r_res_label.grid(row=0, column=0, sticky="ns", pady=2)
        self.r_res_entry = ttk.Entry(self.right)
        self.r_res_entry.grid(row=1, column=0, sticky="ns", pady=2)
        self.r_res_entry.insert(0, "")
        self.r_hfw_label = ttk.Label(self.right, text="Reference HFW (µm):")
        self.r_hfw_label.grid(row=2, column=0, sticky="ns", pady=2)
        self.r_hfw_entry = ttk.Entry(self.right)
        self.r_hfw_entry.grid(row=3, column=0, sticky="ns", pady=2)
        self.r_hfw_entry.insert(0, "")
        # Bottom
        self.ok_button = ttk.Button(self.bot, text="OK", command=self.ok)
        self.ok_button.grid(row=0, column=0, sticky="ns", pady=2)
        self.cancel_button = ttk.Button(self.bot, text="Cancel", command=self.cancel)
        self.cancel_button.grid(row=0, column=1, sticky="ns", pady=2)

    def ok(self):
        if self.d_res_entry.get() == "":
             self.cancel()
             return
        if self.r_res_entry.get() == "":
            if self.r_hfw_entry.get() == "":
                self.cancel()
                return
            bse_hfw = float(self.r_hfw_entry.get())
            bse_res = bse_hfw / bse.shape[2]
downscale = bse_res / ebsd_res
print("Current BSE resolution:", bse_res, "Target EBSD resolution:", ebsd_res)
print("BSE needs to be downscaled by a factor of", downscale, "to match EBSD resolution.")
bse = np.array([transform.rescale(bse[i], downscale, anti_aliasing=True) for i in range(bse.shape[0])])
se = np.array([transform.rescale(se[i], downscale, anti_aliasing=True) for i in range(se.shape[0])])
        pass

    def cancel(self):
        self.w.destroy()
