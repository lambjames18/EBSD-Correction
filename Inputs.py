import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from skimage import io
import h5py
import numpy as np
import skimage.transform as tf
from PIL import Image, ImageTk

### TODO: Add BSE modalities
### TODO: Test reading in control points
### TODO: Test providing a control point path that doesn't exist so that it is created

class DataInput(object):
    def __init__(self, parent, mode="3D"):
        self.parent = parent
        self.mode = mode
        self.ebsd_path = ""
        self.bse_path = ""
        self.ebsd_points_path = ""
        self.bse_points_path = ""
        self.directory = os.getcwd()
        self.clean_exit = False
        self.w = tk.Toplevel(parent)
        self.w.attributes("-topmost", True)
        self.w.title(f"Open {self.mode} Data")
        self.w.rowconfigure(0, weight=10)
        self.w.rowconfigure(1, weight=1)
        self.w.columnconfigure(0, weight=1)
        self.master = ttk.Frame(self.w)
        self.master.grid(row=0, column=0, sticky="nsew", padx=2, pady=5)
        self.bot = ttk.Frame(self.w)
        self.bot.grid(row=1, column=0, sticky="nsew", padx=2, pady=5)
        self.bot.columnconfigure((0, 1, 2, 3), weight=1)
        for i in range(5):
            self.master.rowconfigure(i, weight=1)
        self.master.columnconfigure(0, weight=8)
        self.master.columnconfigure(1, weight=10)
        self.master.columnconfigure(2, weight=1)
        # Vertical structure: EBSD, BSE, EBSD Points, BSE Points, Save/Cancel Buttons
        # EBSD: "Dream3D File" label, Entry for file path (3 columns), "Browse" button
        if self.mode == "3D":
            self.ebsd = ttk.Label(self.master, text="Dream3D File")
        elif self.mode == "2D":
            self.ebsd = ttk.Label(self.master, text="Distorted File")
        self.ebsd.grid(row=1, column=0, sticky="nse", pady=2)
        self.ebsd_entry = ttk.Entry(self.master, width=60)
        self.ebsd_entry.grid(row=1, column=1, sticky="nsew", pady=2)
        self.ebsd_browse = ttk.Button(self.master, text="...", command=self.ebsd_browse)
        self.ebsd_browse.grid(row=1, column=2, sticky="ns", padx=4, pady=2)
        # BSE: "BSE File" label, Entry for file path (3 columns), "Browse" button
        if self.mode == "3D":
            self.bse = ttk.Label(self.master, text="1st Control Image")
        elif self.mode == "2D":
            self.bse = ttk.Label(self.master, text="Control File(s)")
        self.bse.grid(row=2, column=0, sticky="nse", pady=2)
        self.bse_entry = ttk.Entry(self.master, width=60)
        self.bse_entry.grid(row=2, column=1, sticky="nsew", pady=2)
        self.bse_browse = ttk.Button(self.master, text="...", command=self.bse_browse)
        self.bse_browse.grid(row=2, column=2, sticky="ns", padx=4, pady=2)
        # EBSD Points: "EBSD control points" label, Entry for file path (3 columns), "Browse" button
        self.ebsd_points = ttk.Label(self.master, text="Distorted points")
        self.ebsd_points.grid(row=3, column=0, sticky="nse", pady=2)
        self.ebsd_points_entry = ttk.Entry(self.master, width=60)
        self.ebsd_points_entry.grid(row=3, column=1, sticky="nsew", pady=2)
        self.ebsd_points_browse = ttk.Button(self.master, text="...", command=self.ebsd_points_browse)
        self.ebsd_points_browse.grid(row=3, column=2, sticky="ns", padx=4, pady=2)
        # BSE Points: "BSE control points" label, Entry for file path (3 columns), "Browse" button
        self.bse_points = ttk.Label(self.master, text="Control points")
        self.bse_points.grid(row=4, column=0, sticky="nse", pady=2)
        self.bse_points_entry = ttk.Entry(self.master, width=60)
        self.bse_points_entry.grid(row=4, column=1, sticky="nsew", pady=2)
        self.bse_points_browse = ttk.Button(self.master, text="...", command=self.bse_points_browse)
        self.bse_points_browse.grid(row=4, column=2, sticky="ns", padx=4, pady=2)
        # Save/Cancel Buttons: "Save" button, "Cancel" button
        self.ebsd_res_l = ttk.Label(self.bot, text="Distorted Resolution (µm/px): ")
        self.bse_res_l = ttk.Label(self.bot, text="Control Resolution (µm/px): ")
        self.ebsd_res_l.grid(row=0, column=0, sticky="nse", pady=2)
        self.bse_res_l.grid(row=1, column=0, sticky="nse", pady=2)
        self.ebsd_res = ttk.Entry(self.bot)
        self.bse_res = ttk.Entry(self.bot)
        self.ebsd_res.grid(row=0, column=1, sticky="nsw", pady=2)
        self.bse_res.grid(row=1, column=1, sticky="nsw", pady=2)
        self.save = ttk.Button(self.bot, text="Open", command=self.open)
        self.save.grid(row=0, column=2, rowspan=2, padx=2, pady=2)
        self.cancel_b = ttk.Button(self.bot, text="Cancel", command=self.cancel)
        self.cancel_b.grid(row=0, column=3, rowspan=2, padx=2, pady=2)
        # Add info to the bottom
        self.separator = ttk.Separator(self.bot, orient="horizontal")
        self.separator.grid(row=2, column=0, columnspan=4, sticky="nsew", padx=2, pady=2)
        if self.mode == "3D":
            self.info1 = ttk.Label(self.bot, text="Note: The distorted data must be a .dream3d file.")
            self.info2 = ttk.Label(self.bot, text="Note: The control data must be a single image. It needs to be the first of all control images (png, tif, tiff).")
            self.info3 = ttk.Label(self.bot, text="Note: The points files must be a text file. If not provided, they will be created in the directory of the distorted data.")
            self.info4 = ttk.Label(self.bot, text="Note: If a points file is passed that does not exist, a new file will be created with the given name.")
            self.info5 = ttk.Label(self.bot, text="Note: If a points file is passed that does exist, the point data will be read in.")
            self.info6 = ttk.Label(self.bot, text="Note: The control/distorted resolution is only used for rescaling the control image(s). It can be left blank if no resizing is desired.")
        elif self.mode == "2D":
            self.info1 = ttk.Label(self.bot, text="Note: The distorted data can be an ang, h5, or image file (tif, tiff, png, jpg).")
            self.info2 = ttk.Label(self.bot, text="Note: The control data must be an image file (tif, tiff, png, jpg).")
            self.info3 = ttk.Label(self.bot, text="Note: The points files must be a text file. If not provided, they will be created in the directory of the distorted data.")
            self.info4 = ttk.Label(self.bot, text="Note: If a points file is passed that does not exist, a new file will be created with the given name.")
            self.info5 = ttk.Label(self.bot, text="Note: If a points file is passed that does exist, the point data will be read in.")
            self.info6 = ttk.Label(self.bot, text="Note: The control/distorted resolution is only used for rescaling the control image(s). It can be left blank if no resizing is desired.")
        self.info1.grid(row=3, column=0, columnspan=4, sticky="nsew", padx=2, pady=2)
        self.info2.grid(row=4, column=0, columnspan=4, sticky="nsew", padx=2, pady=2)
        self.info3.grid(row=5, column=0, columnspan=4, sticky="nsew", padx=2, pady=2)
        self.info4.grid(row=6, column=0, columnspan=4, sticky="nsew", padx=2, pady=2)
        self.info5.grid(row=7, column=0, columnspan=4, sticky="nsew", padx=2, pady=2)
        self.info6.grid(row=8, column=0, columnspan=4, sticky="nsew", padx=2, pady=2)

    def ebsd_browse(self):
        # Open file dialog to select a .dream3d file
        self.w.attributes("-topmost", False)
        if self.mode == "3D":
            path = filedialog.askopenfilename(initialdir=self.directory, title="Select a .dream3d file", filetypes=(("dream3d files", "*.dream3d"), ("all files", "*.*")))
        else:
            path = filedialog.askopenfilename(initialdir=self.directory, title="Select a distorted (EBSD) file", filetypes=(("ang files", "*.ang"), ("h5 files", "*.h5"), ("tif files", "*.tif"), ("tiff files", "*.tiff"), ("all files", "*.*")))
        # If a file is selected, update the entry box
        if path:
            self.ebsd_entry.delete(0, tk.END)
            self.ebsd_entry.insert(0, path)
            self.directory = os.path.dirname(path)
        self.w.attributes("-topmost", True)

    def bse_browse(self):
        # Open file dialog to select a folder containing BSE images (.png, .tif, .tiff)
        self.w.attributes("-topmost", False)
        if self.mode == "3D":
            path = filedialog.askopenfilename(initialdir=self.directory, title="Select the first control image", filetypes=(("tif files", "*.tif"), ("tiff files", "*.tiff"), ("png files", "*.png"), ("all files", "*.*")))
            if path == "": return
            path = os.path.dirname(path) + "/*" + os.path.splitext(path)[1]
            self.directory = os.path.dirname(path)
            self.bse_entry.delete(0, tk.END)
            self.bse_entry.insert(0, path)
        else:
            path = filedialog.askopenfilenames(initialdir=self.directory, title="Select a control image(s)", filetypes=(("tif files", "*.tif"), ("tiff files", "*.tiff"), ("png files", "*.png"), ("all files", "*.*")))
            if path == "": return
            self.directory = os.path.dirname(path[0])
            path = ";".join(path)
            self.bse_entry.delete(0, tk.END)
            self.bse_entry.insert(0, path)
        self.w.attributes("-topmost", True)

    def ebsd_points_browse(self):
        # Open file dialog to select a .txt file containing EBSD control points
        self.w.attributes("-topmost", False)
        path = filedialog.askopenfilename(initialdir=self.directory, title="Select a .txt file containing distorted points", filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
        # If a file is selected, update the entry box
        if path:
            self.ebsd_points_entry.delete(0, tk.END)
            self.ebsd_points_entry.insert(0, path)
            self.directory = os.path.dirname(path)
        self.w.attributes("-topmost", True)

    def bse_points_browse(self):
        # Open file dialog to select a .txt file containing BSE control points
        self.w.attributes("-topmost", False)
        path = filedialog.askopenfilename(initialdir=self.directory, title="Select a .txt file containing control points", filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
        # If a file is selected, update the entry box
        if path:
            self.bse_points_entry.delete(0, tk.END)
            self.bse_points_entry.insert(0, path)
            self.directory = os.path.dirname(path)
        self.w.attributes("-topmost", True)

    def open(self):
        # Read in the selected data, clear the toplevel, and display the summary
        self.ebsd_path = self.ebsd_entry.get()
        self.bse_path = self.bse_entry.get()
        self.ebsd_points_path = self.ebsd_points_entry.get()
        self.bse_points_path = self.bse_points_entry.get()
        if self.ebsd_res.get() == "":
            self.ebsd_res = 1
        else:
            self.ebsd_res = float(self.ebsd_res.get())
        if self.bse_res.get() == "":
            self.bse_res = 1
        else:
            self.bse_res = float(self.bse_res.get())
        if self.ebsd_path != "" and self.bse_path != "":
            if self.ebsd_points_path == "":
                print("No distorted points passed, creating a file.")
            if self.bse_points_path == "":
                print("No control points passed, creating a file.")
            self.clean_exit = True
        else:
            print("Distorted and control data must be passed.")
        self.cancel()

    def cancel(self):
        # Clear the toplevel and destroy it
        self.w.destroy()


class DataSummary(object):
    """Displays a summary of the data that was read in
    Needs to show the number of BSE images, EBSD images, and control points
    Should show the EBSD data modalities (phase, CI, IQ, etc.)
    Should show the BSE data modalities (BSE, SE)
    Should show the dataset dimensions
    Should show the slices that have points
    Shoud be two panels, one for EBSD and one for BSE
    Should use a dropdown box to view the modalities and the slices that have points"""
    def __init__(self, parent, ebsd_data, bse_data, ebsd_points, bse_points, ebsd_res, bse_res) -> None:
        self.parent = parent
        self.clean_exit = False
        self.rescale = False
        self.parse_data(ebsd_data, bse_data, ebsd_points, bse_points)
        self.w = tk.Toplevel(parent)
        self.w.attributes("-topmost", True)
        self.w.title("Data Summary")
        self.w.rowconfigure(0, weight=10)
        self.w.rowconfigure(1, weight=1)
        self.w.columnconfigure((0, 1), weight=1)
        # Create frames
        self.left = ttk.LabelFrame(self.w, text="Distorted Data", relief="groove", borderwidth=5, padding=5, labelanchor="n")
        self.left.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.right = ttk.LabelFrame(self.w, text="Control Data", relief="groove", borderwidth=5, padding=5, labelanchor="n")
        self.right.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.bot = ttk.Frame(self.w)
        self.bot.grid(row=1, column=0, columnspan=2, sticky="nse")
        # Setup frames
        self.left.rowconfigure((0, 1, 2, 3), weight=1)
        self.left.columnconfigure((0, 1), weight=1)
        self.right.rowconfigure((0, 1, 2, 3), weight=1)
        self.right.columnconfigure(0, weight=1)
        self.bot.columnconfigure((0, 1, 2, 3, 4), weight=1)
        # EBSD labels
        self.ebsd_dims_l = ttk.Label(self.left, text="Dimensions: ")
        self.ebsd_dims_l.grid(row=0, column=0, sticky="nsew")
        self.ebsd_modalities_l = ttk.Label(self.left, text="Modalities: ")
        self.ebsd_modalities_l.grid(row=1, column=0, sticky="nsew")
        self.ebsd_points_l = ttk.Label(self.left, text="Control Points: ")
        self.ebsd_points_l.grid(row=2, column=0, sticky="nsew")
        self.ebsd_res_l = ttk.Label(self.left, text="Resolution (µm/px): ")
        self.ebsd_res_l.grid(row=3, column=0, sticky="nsew")
        # EBSD data
        self.ebsd_dims = ttk.Label(self.left, text=f"{self.ebsd_dims[0]} x {self.ebsd_dims[1]} x {self.ebsd_dims[2]}")
        self.ebsd_dims.grid(row=0, column=1, sticky="nsew")
        self.ebsd_modalities = ttk.Combobox(self.left, values=self.ebsd_modalities, state="readonly", width=10)
        self.ebsd_modalities.grid(row=1, column=1, sticky="nsew")
        self.ebsd_modalities.current(0)
        if self.ebsd_points is None:
            self.ebsd_points = ttk.Label(self.left, text="None")
        else:
            self.ebsd_points = ttk.Combobox(self.left, values=self.ebsd_points, state="readonly", width=5)
            self.ebsd_points.current(0)
        self.ebsd_points.grid(row=2, column=1, sticky="nsew")
        self.ebsd_res = ttk.Label(self.left, text=f"{ebsd_res}")
        self.ebsd_res.grid(row=3, column=1, sticky="nsew")
        # BSE
        self.bse_dims_l = ttk.Label(self.right, text="Dimensions: ")
        self.bse_dims_l.grid(row=0, column=0, sticky="nsew")
        self.bse_modalities_l = ttk.Label(self.right, text="Modalities: ")
        self.bse_modalities_l.grid(row=1, column=0, sticky="nsew")
        self.bse_points_l = ttk.Label(self.right, text="Control Points: ")
        self.bse_points_l.grid(row=2, column=0, sticky="nsew")
        self.bse_res_l = ttk.Label(self.right, text="Resolution (µm/px): ")
        self.bse_res_l.grid(row=3, column=0, sticky="nsew")
        # BSE data
        self.bse_dims = ttk.Label(self.right, text=f"{self.bse_dims[0]} x {self.bse_dims[1]} x {self.bse_dims[2]}")
        self.bse_dims.grid(row=0, column=1, sticky="nsew")
        self.bse_modalities = ttk.Label(self.right, text="Intensity")
        ### TODO: Add BSE modalities
        # self.bse_modalities = ttk.Combobox(self.right, values=self.bse_modalities, state="readonly", width=20)
        self.bse_modalities.grid(row=1, column=1, sticky="nsew")
        if self.bse_points is None:
            self.bse_points = ttk.Label(self.right, text="None")
        else:
            self.bse_points = ttk.Combobox(self.right, values=self.bse_points, state="readonly", width=5)
            self.bse_points.current(0)
        self.bse_points.grid(row=2, column=1, sticky="nsew")
        self.bse_res = ttk.Label(self.right, text=f"{bse_res}")
        self.bse_res.grid(row=3, column=1, sticky="nsew")
        # Bottom
        self.crop = tk.BooleanVar()
        self.crop_check = ttk.Checkbutton(self.bot, text="Crop Control Images?", variable=self.crop)
        self.crop_check.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.flip = tk.BooleanVar()
        self.flip_check = ttk.Checkbutton(self.bot, text="Flip (Up/Down) Control Images?", variable=self.flip)
        self.flip_check.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.r180 = tk.BooleanVar()
        self.r180_check = ttk.Checkbutton(self.bot, text="Rotate Control Images 180º?", variable=self.r180)
        self.r180_check.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)
        self.rescale = tk.BooleanVar()
        self.rescale_check = ttk.Checkbutton(self.bot, text="Rescale Control Images?", variable=self.rescale)
        self.rescale_check.grid(row=0, column=3, sticky="nsew", padx=2, pady=2)
        self.save = ttk.Button(self.bot, text="Continue", command=self.exit)
        self.save.grid(row=0, column=4, sticky="nsew", padx=2, pady=2)
        # Put in some information at the bottom
        self.separator = ttk.Separator(self.bot, orient="horizontal")
        self.separator.grid(row=1, column=0, columnspan=5, sticky="nsew", padx=2, pady=2)
        self.info1 = ttk.Label(self.bot, text="Note: Selecting the crop option will open a new window to select a region of the control image(s) (i.e. remove empty space around sample).")
        self.info2 = ttk.Label(self.bot, text="Note: Rescale Control will attempt to match the pixel resolutions of the distorted and control images by downsampling the control image (you should do this).")
        self.info3 = ttk.Label(self.bot, text="Note: Flip or rotate 180º is only applied to the control images if selected (often needed for TriBeam datasets).")
        self.info1.grid(row=2, column=0, columnspan=5, sticky="nsew", padx=2, pady=2)
        self.info2.grid(row=3, column=0, columnspan=5, sticky="nsew", padx=2, pady=2)
        self.info3.grid(row=4, column=0, columnspan=5, sticky="nsew", padx=2, pady=2)

    def parse_data(self, ebsd_data, bse_data, ebsd_points, bse_points):
        self.ebsd_modalities = list(ebsd_data.keys())
        ### TODO: Add BSE modalities
        # self.bse_modalities = list(bse_data.keys())
        self.ebsd_dims = ebsd_data[self.ebsd_modalities[0]].shape
        ### TODO: Add BSE modalities
        # self.bse_dims = bse_data[self.bse_modalities[0]].shape
        self.bse_dims = bse_data.shape
        if ebsd_points is None:
            self.ebsd_points = None
        else:
            self.ebsd_points = list(ebsd_points.keys())
        if bse_points is None:
            self.bse_points = None
        else:
            self.bse_points = list(bse_points.keys())
    
    def exit(self):
        self.clean_exit = True
        self.rescale = self.rescale.get()
        self.crop = self.crop.get()
        self.flip = self.flip.get()
        self.r180 = self.r180.get()
        self.w.destroy()
    
    def cancel(self):
        self.w.destroy()


class CropControl(object):
    """Displays the control image and prompts the user to select a ROI for cropping"""
    def __init__(self, parent, im) -> None:
        self.parent = parent
        self.im = im
        self.shape = np.around(np.array(im.shape) / 2).astype(int)
        self.scale_factor = np.array(self.im.shape) / self.shape
        self.TK_im = Image.frombytes("L", (self.im.shape[1], self.im.shape[0]), self.im)
        self.TK_im = self.TK_im.resize((self.shape[1], self.shape[0]))
        self.TK_im = ImageTk.PhotoImage(self.TK_im)
        self.clean_exit = False
        self.start = [0, 0]
        self.end = list(self.shape)
        self.w = tk.Toplevel(parent)
        self.w.attributes("-topmost", True)
        self.w.title("Cropping the control image(s): Click and drag to select a region.")
        # Configure the UI
        self.w.rowconfigure(0, weight=10)
        self.w.rowconfigure(1, weight=1)
        self.w.columnconfigure(0, weight=1)
        self.w.columnconfigure(1, weight=10)
        self.details = ttk.Frame(self.w)
        self.details.grid(row=0, column=0, sticky="nsew", padx=2, pady=5)
        self.image = ttk.Frame(self.w)
        self.image.grid(row=0, column=1, sticky="nsew", padx=2, pady=5)
        self.bot = ttk.Frame(self.w)
        self.bot.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=2, pady=5)
        # Configure and make details (row start, row end, column start, column end)
        self.details.rowconfigure((0, 1, 2, 3), weight=1)
        self.details.columnconfigure((0, 1), weight=1)
        self.srl = ttk.Label(self.details, text="Row start: ")
        self.srl.grid(row=0, column=0, sticky="e")
        self.sre = ttk.Entry(self.details)
        self.sre.grid(row=0, column=1, sticky="w")
        self.sre.bind("<Return>", self.read_entries)
        self.erl = ttk.Label(self.details, text="Row end: ")
        self.erl.grid(row=1, column=0, sticky="e")
        self.ere = ttk.Entry(self.details)
        self.ere.grid(row=1, column=1, sticky="w")
        self.ere.bind("<Return>", self.read_entries)
        self.scl = ttk.Label(self.details, text="Column start: ")
        self.scl.grid(row=2, column=0, sticky="e")
        self.sce = ttk.Entry(self.details)
        self.sce.grid(row=2, column=1, sticky="w")
        self.sce.bind("<Return>", self.read_entries)
        self.ecl = ttk.Label(self.details, text="Column end: ")
        self.ecl.grid(row=3, column=0, sticky="e")
        self.ece = ttk.Entry(self.details)
        self.ece.grid(row=3, column=1, sticky="w")
        self.ece.bind("<Return>", self.read_entries)
        # Configure and make image
        self.image.rowconfigure(0, weight=1)
        self.image.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.image, width=self.shape[1], height=self.shape[0])
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.create_image((0, 0), image=self.TK_im, anchor="nw")
        # Configure and make bottom
        self.bot.columnconfigure((0, 1, 2), weight=1)
        self.save = ttk.Button(self.bot, text="Save", command=self.save)
        self.save.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.cancel = ttk.Button(self.bot, text="Cancel", command=self.cancel)
        self.cancel.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.reset_b = ttk.Button(self.bot, text="Reset", command=self.reset)
        self.reset_b.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)
        # Update the details
        self.update_details()
    
    def click(self, event):
        self.start = [int(event.y), int(event.x)]
        self.end = [int(event.y), int(event.x)]
        self.update_details()
    
    def drag(self, event):
        self.end = [int(event.y), int(event.x)]
        self.update_details()
    
    def release(self, event):
        self.end = [int(event.y), int(event.x)]
        self.update_details()
    
    def reset(self):
        self.start = [0, 0]
        self.end = list(self.im.shape)
        self.update_details()
    
    def read_entries(self, *args):
        self.start[0] = int(np.around(float(self.sre.get()) / self.scale_factor[0]))
        self.end[0] = int(np.around(float(self.ere.get()) / self.scale_factor[0]))
        self.start[1] = int(np.around(float(self.sce.get()) / self.scale_factor[0]))
        self.end[1] = int(np.around(float(self.ece.get()) / self.scale_factor[0]))
        self.update_details()
    
    def update_details(self):
        self.start[0] = min(max(0, self.start[0]), self.shape[0])
        self.end[0] = min(max(0, self.end[0]), self.shape[0])
        self.start[1] = min(max(0, self.start[1]), self.shape[1])
        self.end[1] = min(max(0, self.end[1]), self.shape[1])
        self.canvas.delete("crop")
        self.canvas.create_rectangle(self.start[1], self.start[0], self.end[1], self.end[0], outline="red", width=2, tag="crop")
        self.sre.delete(0, tk.END)
        self.sre.insert(0, int(np.around(self.start[0] * self.scale_factor[0])))
        self.ere.delete(0, tk.END)
        self.ere.insert(0, int(np.around(self.end[0] * self.scale_factor[0])))
        self.sce.delete(0, tk.END)
        self.sce.insert(0, int(np.around(self.start[1] * self.scale_factor[0])))
        self.ece.delete(0, tk.END)
        self.ece.insert(0, int(np.around(self.end[1] * self.scale_factor[0])))
    
    def save(self):
        self.clean_exit = True
        self.start = np.around(np.array(self.start) * self.scale_factor).astype(int)
        self.end = np.around(np.array(self.end) * self.scale_factor).astype(int)
        self.w.destroy()

    def cancel(self):
        self.w.destroy()


##########
# Functions for reading data
##########

def read_ang(path):
    """Reads an ang file into a numpy array"""
    num_header_lines = 0
    col_names = None
    with open(path, "r") as f:
        for line in f:
            if line[0] == "#":
                num_header_lines += 1
                if "NCOLS_ODD" in line:
                    ncols = int(line.split(": ")[1].strip())
                elif "NROWS" in line:
                    nrows = int(line.split(": ")[1].strip())
                elif "COLUMN_HEADERS" in line:
                    col_names = line.split(": ")[1].strip().split(", ")
                elif "XSTEP" in line:
                    res = float(line.split(": ")[1].strip())
            else:
                break
    if col_names is None:
        col_names = ["phi1", "PHI", "phi2", "x", "y", "IQ", "CI", "Phase index"]
    raw_data = np.genfromtxt(path, skip_header=num_header_lines)
    n_entries = raw_data.shape[-1]
    if raw_data.shape[0] == ncols * nrows:
        data = raw_data.reshape((nrows, ncols, n_entries))
    elif raw_data.shape != ncols * nrows:
        raise ValueError(f"The number of data points ({raw_data.size}) does not match the expected grid ({nrows} rows, {ncols} cols, {ncols * nrows} total points). ")
        
    out = {col_names[i]: data[:, :, i] for i in range(n_entries)}
    out["EulerAngles"] = np.array([out["phi1"], out["PHI"], out["phi2"]]).T.astype(float)
    out["Phase"] = out["Phase index"].astype(np.int32)
    out["XPos"] = out["x"].astype(float)
    out["YPos"] = out["y"].astype(float)
    out["IQ"] = out["IQ"].astype(float)
    out["CI"] = out["CI"].astype(float)
    for key in ["phi1", "PHI", "phi2", "Phase index", "PRIAS Bottom Strip", "PRIAS Top Strip", "PRIAS Center Square", "SEM", "Fit", "x", "y"]:
        try:
            del out[key]
        except KeyError:
            pass
    for key in out.keys():
        if key != "EulerAngles":
            out[key] = np.fliplr(np.rot90(out[key], k=3))
        if len(out[key].shape) == 2:
            out[key] = out[key].T
        else:
            out[key] = out[key].transpose((1, 0, 2))
        out[key] = out[key].reshape((1,) + out[key].shape)
    return out, res

def read_h5(path):
    h5 = h5py.File(path, "r")
    if "1" in h5.keys():
        entry = "1"
    elif "Scan 1" in h5.keys():
        entry = "Scan 1"
    else:
        raise ValueError("Could not find EBSD data in the h5 file.")
    ebsd_data = h5[entry + "/EBSD/Data"]
    nrows = h5[entry + "/EBSD/Header/nRows"][0]
    ncols = h5[entry + "/EBSD/Header/nColumns"][0]
    res = h5[entry + "/EBSD/Header/Step X"][0]
    keys = list(ebsd_data.keys())
    ebsd_data = {key.upper().replace(" ", "-"): ebsd_data[key][...].reshape(1, nrows, ncols, -1) for key in keys}
    h5.close()
    ebsd_data["EULERANGLES"] = np.stack((ebsd_data["PHI1"], ebsd_data["PHI"], ebsd_data["PHI2"]), axis=-1).astype(float)
    return ebsd_data, res

def read_dream3d(path):
    h5 = h5py.File(path, "r")
    if "DataStructure" in h5.keys():
        ebsd_data = h5["DataStructure/DataContainer/CellData"]
        res = np.asarray(h5["DataStructure/DataContainer"].attrs.get("_SPACING"))[1]
    elif "DataContainers" in h5.keys():
        ebsd_data = h5["DataContainers/ImageDataContainer/CellData"]
        res = h5["DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"][...][1]
    else:
        raise ValueError("Could not find EBSD data in the dream3d file. The top level group should be 'DataStructure' or 'DataContainers'.")
    ebsd_keys = list(ebsd_data.keys())
    ebsd_data = {key: ebsd_data[key][...] for key in ebsd_keys}
    h5.close()
    return ebsd_data, res

def read_many_images(path, ext):
    ### TODO: Add multiple control images
    paths = sorted(
        [p for p in os.listdir(path) if os.path.splitext(p)[1] == ext],
        key=lambda x: int(x.replace(ext, "")),
    )
    imgs = []
    for i in range(len(paths)):
        p = os.path.join(path, paths[i])
        im = io.imread(p, as_gray=True).astype(np.float32)
        im = np.around(255 * (im - im.min()) / (im.max() - im.min()), 0).astype(np.uint8)
        imgs.append(im)
    imgs = np.array(imgs, dtype=np.uint8)
    imgs = imgs.reshape(imgs.shape)
    return imgs

def read_points(path):
    points = np.loadtxt(path, delimiter=" ", dtype=int)
    if points.ndim == 1:
        points = points.reshape((1, -1))
    z, y, x = points[:, 0], points[:, 1], points[:, 2]
    unique_slices = np.unique(z)
    points_data = {slice_num: np.hstack((y[z == slice_num].reshape(-1, 1), x[z == slice_num].reshape(-1, 1))) for slice_num in unique_slices}
    return points_data

def read_image(path):
    if path.endswith(".png") or path.endswith(".jpg"):
        print("Warning: The image is not grayscale. The image will be converted to grayscale.")
    im = io.imread(path, as_gray=True).astype(np.float32)
    im = np.around((im - np.min(im)) / (np.max(im) - np.min(im)) * 255, 0).astype(np.uint8)
    return im.reshape((1,) + im.shape)


##########
def read_data(ebsd_path, bse_path, ebsd_points_path, bse_points_path):
    if ebsd_path.endswith(".ang"):
        ebsd_data = read_ang(ebsd_path)[0]
    elif ebsd_path.endswith(".h5"):
        ebsd_data = read_h5(ebsd_path)[0]
    elif ebsd_path.endswith(".dream3d"):
        ebsd_data = read_dream3d(ebsd_path)[0]
    elif ebsd_path.endswith(".tif") or ebsd_path.endswith(".tiff"):
        ebsd_data = read_image(ebsd_path)
        ebsd_data = {"Intensity": ebsd_data}
    else:
        raise ValueError(f"Unknown file type: {ebsd_path}")
    if "*" in bse_path:
        ext = os.path.splitext(bse_path)[1]
        folder = os.path.dirname(bse_path)
        bse_data = read_many_images(folder, ext)
        if bse_data.shape[0] != ebsd_data[list(ebsd_data.keys())[0]].shape[0]:
            raise ValueError("The number of BSE images does not match the number of EBSD images.")
    else:
        bse_data = read_image(bse_path)
    if ebsd_points_path == "":
        ebsd_points = None
    else:
        ebsd_points = read_points(ebsd_points_path)
    if bse_points_path == "":
        bse_points = None
    else:
        bse_points = read_points(bse_points_path)
    return ebsd_data, bse_data, ebsd_points, bse_points

def rescale_control(bse_data, bse_res, ebsd_res):
    downscale = bse_res / ebsd_res
    print("Current BSE resolution:", bse_res, "Target EBSD resolution:", ebsd_res)
    print("BSE needs to be downscaled by a factor of", downscale, "to match EBSD resolution.")
    rescaled_bse = np.array([tf.rescale(bse_data[i], downscale, anti_aliasing=True) for i in range(bse_data.shape[0])])
    rescaled_bse = np.around(255 * (rescaled_bse - rescaled_bse.min()) / (rescaled_bse.max() - rescaled_bse.min()), 0).astype(np.uint8)
    return rescaled_bse

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

if __name__ == "__main__":
    # root = tk.Tk()
    # _style_call(root, style='dark')
    # a = DataInput(root, mode="2D")
    # root.mainloop()
    # Read in the data
    # ebsd_path = a.ebsd_path
    # bse_path = a.bse_path
    # ebsd_points_path = a.ebsd_points_path
    # bse_points_path = a.bse_points_path
    # ebsd_res = a.ebsd_res
    # bse_res = a.bse_res
    # if ebsd_path == "" or bse_path == "":
    #     exit()
    ebsd_path = "/Users/jameslamb/Documents/Research/scripts/EBSD-Correction/test_data/OIM.ang"
    bse_path = "/Users/jameslamb/Documents/Research/scripts/EBSD-Correction/test_data/BSE.tif"
    ebsd_points_path = ""
    bse_points_path = ""
    ebsd_res = 1.5
    bse_res = 1.3
    ebsd_data, bse_data, ebsd_points, bse_points = read_data(ebsd_path, bse_path, ebsd_points_path, bse_points_path)
    # Display the data summary
    root = tk.Tk()
    _style_call(root, style='dark')
    b = DataSummary(root, ebsd_data, bse_data, ebsd_points, bse_points, ebsd_res, bse_res)
    root.mainloop()
    # If the user exited cleanly, continue
    if b.clean_exit:
        if b.rescale:
            bse_data = rescale_control(bse_data, bse_res, ebsd_res)
        if b.crop:
            root = tk.Tk()
            _style_call(root, style='dark')
            c = CropControl(root, bse_data[0, :, :, 0])
            root.mainloop()
            if c.clean_exit:
                print(c.start, c.end)