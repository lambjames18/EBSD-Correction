import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import h5py
import numpy as np


class Data3D(object):
    def __init__(self, parent):
        self.parent = parent
        self.directory = os.getcwd()
        self.clean_exit = False
        self.w = tk.Toplevel(parent)
        self.w.title("Open 3D Data")
        self.w.rowconfigure(0, weight=10)
        self.w.rowconfigure(1, weight=1)
        self.w.columnconfigure(0, weight=1)
        self.master = ttk.Frame(self.w)
        self.master.grid(row=0, column=0, sticky="nsew")
        self.bot = ttk.Frame(self.w)
        self.bot.grid(row=1, column=0, sticky="nsew")
        self.bot.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)
        for i in range(5):
            self.master.rowconfigure(i, weight=1)
        self.master.columnconfigure(0, weight=8)
        self.master.columnconfigure(1, weight=10)
        self.master.columnconfigure(2, weight=1)
        # Vertical structure: EBSD, BSE, EBSD Points, BSE Points, Save/Cancel Buttons
        # EBSD: "Dream3D File" label, Entry for file path (3 columns), "Browse" button
        self.ebsd = ttk.Label(self.master, text="Dream3D File")
        self.ebsd.grid(row=1, column=0, sticky="nse")
        self.ebsd_entry = ttk.Entry(self.master, width=60)
        self.ebsd_entry.grid(row=1, column=1, sticky="nsew")
        self.ebsd_browse = ttk.Button(self.master, text="...", command=self.ebsd_browse)
        self.ebsd_browse.grid(row=1, column=2, sticky="ns", padx=4, pady=2)
        # BSE: "BSE File" label, Entry for file path (3 columns), "Browse" button
        self.bse = ttk.Label(self.master, text="BSE File")
        self.bse.grid(row=2, column=0, sticky="nse")
        self.bse_entry = ttk.Entry(self.master, width=60)
        self.bse_entry.grid(row=2, column=1, sticky="nsew")
        self.bse_browse = ttk.Button(self.master, text="...", command=self.bse_browse)
        self.bse_browse.grid(row=2, column=2, sticky="ns", padx=4, pady=2)
        # Save/Cancel Buttons: "Save" button, "Cancel" button
        self.save = ttk.Button(self.bot, text="Open", command=self.open)
        self.save.grid(row=0, column=4, sticky="nsew", padx=2, pady=2)
        self.cancel = ttk.Button(self.bot, text="Cancel", command=self.cancel)
        self.cancel.grid(row=0, column=5, sticky="nsew", padx=2, pady=2)

    def ebsd_browse(self):
        # Open file dialog to select a .dream3d file
        path = filedialog.askopenfilename(initialdir=self.directory, title="Select a .dream3d file", filetypes=(("dream3d files", "*.dream3d"), ("all files", "*.*")))
        # If a file is selected, update the entry box
        if path:
            self.ebsd_entry.delete(0, tk.END)
            self.ebsd_entry.insert(0, path)
            self.directory = os.path.dirname(path)

    def bse_browse(self):
        # Open file dialog to select a folder containing BSE images (.png, .tif, .tiff)
        path = filedialog.askdirectory(initialdir=self.directory, title="Select a folder containing BSE images", filetypes=(("png files", "*.png"), ("tif files", "*.tif"), ("tiff files", "*.tiff"), ("all files", "*.*")))
        # If a folder is selected, update the entry box
        if path:
            self.bse_entry.delete(0, tk.END)
            self.bse_entry.insert(0, path)
            self.directory = os.path.dirname(path)

    def ebsd_points_browse(self):
        # Open file dialog to select a .txt file containing EBSD control points
        path = filedialog.askopenfilename(initialdir=self.directory, title="Select a .txt file containing EBSD control points", filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
        # If a file is selected, update the entry box
        if path:
            self.ebsd_points_entry.delete(0, tk.END)
            self.ebsd_points_entry.insert(0, path)
            self.directory = os.path.dirname(path)

    def bse_points_browse(self):
        # Open file dialog to select a .txt file containing BSE control points
        path = filedialog.askopenfilename(initialdir=self.directory, title="Select a .txt file containing BSE control points", filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
        # If a file is selected, update the entry box
        if path:
            self.bse_points_entry.delete(0, tk.END)
            self.bse_points_entry.insert(0, path)
            self.directory = os.path.dirname(path)

    def open(self):
        # Read in the selected data, clear the toplevel, and display the summary
        self.ebsd_path = self.ebsd_entry.get()
        self.bse_path = self.bse_entry.get()
        self.ebsd_points_path = self.ebsd_points_entry.get()
        self.bse_points_path = self.bse_points_entry.get()
        self.clean_exit = True
        self.cancel()

    def cancel(self):
        # Clear the toplevel and destroy it
        self.w.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    Data3D(root)
    root.mainloop()