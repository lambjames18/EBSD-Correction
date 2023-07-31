import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from skimage import io
import h5py
import numpy as np


class DataSummary(object):
    """Displays a summary of the data that was read in
    Needs to show the number of BSE images, EBSD images, and control points
    Should show the EBSD data modalities (phase, CI, IQ, etc.)
    Should show the BSE data modalities (BSE, SE)
    Should show the dataset dimensions
    Should show the slices that have points
    Shoud be two panels, one for EBSD and one for BSE
    Should use a dropdown box to view the modalities and the slices that have points"""
    def __init__(self, parent, ebsd_data, bse_data, ebsd_points, bse_points) -> None:
        self.parent = parent
        self.clean_exit = False
        self.parse_data(ebsd_data, bse_data, ebsd_points, bse_points)
        self.w = tk.Toplevel(parent)
        self.w.title("Data Summary")
        self.w.rowconfigure(0, weight=10)
        self.w.rowconfigure(1, weight=1)
        self.w.columnconfigure((0, 1), weight=1)
        # Create frames
        self.left = ttk.Frame(self.w)
        self.left.grid(row=0, column=0, sticky="nsew")
        self.right = ttk.Frame(self.w)
        self.right.grid(row=0, column=1, sticky="nsew")
        self.bot = ttk.Frame(self.w)
        self.bot.grid(row=1, column=0, columnspan=2, sticky="nse")
        # Setup frames
        self.left.rowconfigure((0, 1, 2, 3), weight=1)
        self.left.columnconfigure((0, 1), weight=1)
        self.right.rowconfigure((0, 1, 2, 3), weight=1)
        self.right.columnconfigure(0, weight=1)
        self.bot.columnconfigure((0, 1), weight=1)
        # EBSD labels
        self.ebsd = ttk.Label(self.left, text="EBSD Data")
        self.ebsd.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.ebsd_dims_l = ttk.Label(self.left, text="Dimensions: ")
        self.ebsd_dims_l.grid(row=1, column=0, sticky="nsew")
        self.ebsd_modalities_l = ttk.Label(self.left, text="Modalities: ")
        self.ebsd_modalities_l.grid(row=2, column=0, sticky="nsew")
        self.ebsd_points_l = ttk.Label(self.left, text="Control Points: ")
        self.ebsd_points_l.grid(row=3, column=0, sticky="nsew")
        # EBSD data
        self.ebsd_dims = ttk.Label(self.left, text=f"{self.ebsd_dims[0]} x {self.ebsd_dims[1]} x {self.ebsd_dims[2]}")
        self.ebsd_dims.grid(row=1, column=1, sticky="nsew")
        self.ebsd_modalities = ttk.Combobox(self.left, values=self.ebsd_modalities, state="readonly")
        self.ebsd_modalities.grid(row=2, column=1, sticky="nsew")
        self.ebsd_points = ttk.Combobox(self.left, values=self.ebsd_points, state="readonly")
        self.ebsd_points.grid(row=3, column=1, sticky="nsew")
        # BSE
        self.bse = ttk.Label(self.right, text="BSE Data")
        self.bse.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.bse_dims_l = ttk.Label(self.right, text="Dimensions: ")
        self.bse_dims_l.grid(row=1, column=0, sticky="nsew")
        self.bse_modalities_l = ttk.Label(self.right, text="Modalities: ")
        self.bse_modalities_l.grid(row=2, column=0, sticky="nsew")
        self.bse_points_l = ttk.Label(self.right, text="Control Points: ")
        self.bse_points_l.grid(row=3, column=0, sticky="nsew")
        # BSE data
        self.bse_dims = ttk.Label(self.right, text=f"{self.bse_dims[0]} x {self.bse_dims[1]} x {self.bse_dims[2]}")
        self.bse_dims.grid(row=1, column=1, sticky="nsew")
        self.bse_modalities = ttk.Combobox(self.right, values=self.bse_modalities, state="readonly")
        self.bse_modalities.grid(row=2, column=1, sticky="nsew")
        self.bse_points = ttk.Combobox(self.right, values=self.bse_points, state="readonly")
        self.bse_points.grid(row=3, column=1, sticky="nsew")
        # Bottom
        self.save = ttk.Button(self.bot, text="Continue", command=self.continue_)
        self.save.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.cancel = ttk.Button(self.bot, text="Cancel", command=self.cancel)
        self.cancel.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

    def parse_data(self, ebsd_data, bse_data, ebsd_points, bse_points):
        self.ebsd_modalities = list(ebsd_data.keys())
        self.bse_modalities = list(bse_data.keys())
        self.ebsd_dims = ebsd_data[self.ebsd_modalities[0]].shape
        self.bse_dims = bse_data[self.bse_modalities[0]].shape
        self.ebsd_points = list(ebsd_points.keys())
        self.bse_points = list(bse_points.keys())
    
    def continue_(self):
        self.clean_exit = True
        self.w.destroy()
    
    def cancel(self):
        self.w.destroy()



class DataInput(object):
    def __init__(self, parent, mode="3D"):
        self.parent = parent
        self.mode = mode
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
        if self.mode == "3D":
            self.ebsd = ttk.Label(self.master, text="Dream3D File")
        elif self.mode == "2D":
            self.ebsd = ttk.Label(self.master, text="EBSD File")
        self.ebsd.grid(row=1, column=0, sticky="nse")
        self.ebsd_entry = ttk.Entry(self.master, width=60)
        self.ebsd_entry.grid(row=1, column=1, sticky="nsew")
        self.ebsd_browse = ttk.Button(self.master, text="...", command=self.ebsd_browse)
        self.ebsd_browse.grid(row=1, column=2, sticky="ns", padx=4, pady=2)
        # BSE: "BSE File" label, Entry for file path (3 columns), "Browse" button
        if self.mode == "3D":
            self.bse = ttk.Label(self.master, text="BSE Folder")
        elif self.mode == "2D":
            self.bse = ttk.Label(self.master, text="BSE File")
        self.bse.grid(row=2, column=0, sticky="nse")
        self.bse_entry = ttk.Entry(self.master, width=60)
        self.bse_entry.grid(row=2, column=1, sticky="nsew")
        self.bse_browse = ttk.Button(self.master, text="...", command=self.bse_browse)
        self.bse_browse.grid(row=2, column=2, sticky="ns", padx=4, pady=2)
        # EBSD Points: "EBSD control points" label, Entry for file path (3 columns), "Browse" button
        self.ebsd_points = ttk.Label(self.master, text="EBSD control points")
        self.ebsd_points.grid(row=3, column=0, sticky="nse")
        self.ebsd_points_entry = ttk.Entry(self.master, width=60)
        self.ebsd_points_entry.grid(row=3, column=1, sticky="nsew")
        self.ebsd_points_browse = ttk.Button(self.master, text="...", command=self.ebsd_points_browse)
        self.ebsd_points_browse.grid(row=3, column=2, sticky="ns", padx=4, pady=2)
        # BSE Points: "BSE control points" label, Entry for file path (3 columns), "Browse" button
        self.bse_points = ttk.Label(self.master, text="BSE control points")
        self.bse_points.grid(row=4, column=0, sticky="nse")
        self.bse_points_entry = ttk.Entry(self.master, width=60)
        self.bse_points_entry.grid(row=4, column=1, sticky="nsew")
        self.bse_points_browse = ttk.Button(self.master, text="...", command=self.bse_points_browse)
        self.bse_points_browse.grid(row=4, column=2, sticky="ns", padx=4, pady=2)
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

def read_ang(path):
    """Reads an ang file into a numpy array"""
    num_header_lines = 0
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
            else:
                break
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
    return out

def read_h5(path):
    h5 = h5py.File(path, "r")
    ebsd_data = h5["Scan 1/EBSD/Data"]
    keys = list(ebsd_data.keys())
    nrows = h5["Scan 1/EBSD/Header/nRows"][0]
    ncols = h5["Scan 1/EBSD/Header/nColumns"][0]
    ebsd_data = {key: np.squeeze(ebsd_data[key][...].reshape(nrows, ncols, -1)) for key in keys}
    return ebsd_data

def read_dream3d(path):
    h5 = h5py.File(path, "r")
    ebsd_data = h5["DataContainers/ImageDataContainer/CellData"]
    ebsd_keys = list(ebsd_data.keys())
    ebsd_data = {key: np.squeeze(ebsd_data[key][...]) for key in ebsd_keys}
    return ebsd_data

def read_many_images(path, ext):
    paths = sorted(
        [path for path in os.listdir(path) if os.path.splitext(path)[1] == ext],
        key=lambda x: int(x.replace(ext, "")),
    )
    imgs = []
    for i in range(len(paths)):
        p = os.path.join(imgs, paths[i])
        im = io.imread(p, as_gray=True).astype(np.float32)
        im = np.around(255 * (im - im.min()) / (im.max() - im.min()), 0).astype(np.uint8)
        imgs.append(im)
    imgs = np.array(imgs, dtype=np.uint8)
    return imgs

def read_points(path):
    points = np.loadtxt(path, delimiter=",", dtype=int)
    z, y, x = points[:, 0], points[:, 1], points[:, 2]
    unique_slices = np.unique(z)
    points_data = {slice_num: np.hstack((y[z == slice_num].reshape(-1, 1), x[z == slice_num].reshape(-1, 1))) for slice_num in unique_slices}
    return points_data

def read_image(path):
    if path.endswith(".png") or path.endswith(".jpg"):
        print("Warning: The image is not grayscale. The image will be converted to grayscale.")
    im = io.imread(path, as_gray=True).astype(np.float32)
    im = np.around((im - np.min(im)) / (np.max(im) - np.min(im)) * 255, 0).astype(np.uint8)
    return im

def read_data(ebsd_path, bse_path, ebsd_points_path, bse_points_path):
    if ebsd_path.endswith(".ang"):
        ebsd_data = read_ang(ebsd_path)
    elif ebsd_path.endswith(".h5"):
        ebsd_data = read_h5(ebsd_path)
    elif ebsd_path.endswith(".dream3d"):
        ebsd_data = read_dream3d(ebsd_path)
    else:
        raise ValueError(f"Unknown file type: {ebsd_path}")
    if os.path.isdir(bse_path):
        bse_data = read_many_images(bse_path, ".png")
    else:
        bse_data = read_image(bse_path)

if __name__ == "__main__":
    root = tk.Tk()
    DataInput(root, mode="2D")
    root.mainloop()