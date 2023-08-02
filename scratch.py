import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio.v3 as iio
from skimage import transform, io
import core
import tifffile


def _open_ang(ang_path: str):
    """Reads an ang file into a numpy array"""
    num_header_lines = 0
    col_names = None
    with open(ang_path, "r") as f:
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
    if col_names is None:
        col_names = ["phi1", "PHI", "phi2", "x", "y", "IQ", "CI", "Phase index"]
    raw_data = np.genfromtxt(ang_path, skip_header=num_header_lines)
    n_entries = raw_data.shape[-1]
    if raw_data.shape[0] == ncols * nrows:
        data = raw_data.reshape((nrows, ncols, n_entries))
    elif raw_data.shape[0] == ncols * (nrows - 1):
        data = raw_data.reshape((nrows - 1, ncols, n_entries))
        print("Warning: The number of data points does not match number of rows and columns. Automatic adjustments succeeded with (nrows - 1, ncols).")
    elif raw_data.shape[0] == ncols * (nrows + 1):
        data = raw_data.reshape((nrows + 1, ncols, n_entries))
        print("Warning: The number of data points does not match number of rows and columns. Automatic adjustments succeeded with (nrows + 1, ncols).")
    elif raw_data.shape[0] == (ncols - 1) * nrows:
        data = raw_data.reshape((nrows, ncols - 1, n_entries))
        print("Warning: The number of data points does not match number of rows and columns. Automatic adjustments succeeded with (nrows, ncols - 1).")
    elif raw_data.shape[0] == (ncols + 1) * nrows:
        data = raw_data.reshape((nrows, ncols + 1, n_entries))
        print("Warning: The number of data points does not match number of rows and columns. Automatic adjustments succeeded with (nrows, ncols + 1).")
    else:
        raise ValueError("The number of data points does not match number of rows and columns. Automatic adjustments failed (tried +-1 for both rows and columns).")
        
    out = {col_names[i]: data[:, :, i] for i in range(n_entries)}
    out["EulerAngles"] = np.array([out["phi1"], out["PHI"], out["phi2"]]).T.astype(np.float32)
    out["Phase"] = out["Phase index"].astype(np.int32)
    out["XPos"] = out["x"].astype(np.float32)
    out["YPos"] = out["y"].astype(np.float32)
    out["IQ"] = out["IQ"].astype(np.float32)
    out["CI"] = out["CI"].astype(np.float32)
    for key in ["phi1", "PHI", "phi2", "Phase index", "PRIAS Bottom Strip", "PRIAS Top Strip", "PRIAS Center Square", "SEM", "Fit", "x", "y"]:
        try:
            del out[key]
        except KeyError:
            pass
    for key in out.keys():
        if key != "EulerAngles":
            out[key] = np.fliplr(np.rot90(out[key], k=3)).T
        else:
            out[key] = out[key].transpose((1, 0, 2))
    return out


d = _open_ang("test_data/EBSD.ang")

io.imsave("test_data/EBSD.tif", d["Phase"])
d_read = io.imread("test_data/EBSD.tif")
print(np.all(d_read == d["Phase"]), d_read.dtype, d["Phase"].dtype)
tifffile.imwrite("test_data/EBSD2.tif", d["Phase"])
d_read = tifffile.imread("test_data/EBSD2.tif")
print(np.all(d_read == d["Phase"]), d_read.dtype, d["Phase"].dtype)
iio.imwrite("test_data/EBSD3.tif", d["Phase"])
d_read = iio.imread("test_data/EBSD3.tif")
print(np.all(d_read == d["Phase"]), d_read.dtype, d["Phase"].dtype)