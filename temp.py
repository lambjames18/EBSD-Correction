import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def read_ang(path):
    """Reads an ang file into a numpy array"""
    num_header_lines = 0
    col_names = None
    header = []
    with open(path, "r") as f:
        for line in f:
            if line[0] == "#":
                header.append(line)
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
    out["header"] = header
    return out

def write_ang(data:dict):
    header = data["header"]
    ncols = data["IQ"].shape[1]
    nrows = data["IQ"].shape[0]
    n_keys = len([key for key in data.keys() if key != "header"])
    arr = np.zeros((nrows * ncols, 8))
    arr[:, 0] = data["EulerAngles"][:, :, 0].flatten()
    arr[:, 1] = data["EulerAngles"][:, :, 1].flatten()
    arr[:, 2] = data["EulerAngles"][:, :, 2].flatten()
    arr[:, 3] = data["XPos"].flatten()
    arr[:, 4] = data["YPos"].flatten()
    arr[:, 5] = data["IQ"].flatten()
    arr[:, 6] = data["CI"].flatten()
    arr[:, 7] = data["Phase"].flatten()
    data_out = []
    for i in range(nrows * ncols):
        data_out.append(f"  {arr[i, 0]:.5f}   {arr[i, 1]:.5f}   {arr[i, 2]:.5f}"
                        + " "*(7-len(str(arr[i, 3]).split(".")[0])) + f"{arr[i, 3]:.5f}"
                        + " "*(7-len(str(arr[i, 4]).split(".")[0])) + f"{arr[i, 4]:.5f}"
                        + f"     {arr[i, 5]:.5f}  {arr[i, 6]:.5f}  {arr[i, 7]:.0f}")
    data_out = "\n".join(data_out)
    with open("D:/Research/CoNi67/Data/temp/1_new.ang", "w") as File:
        File.write("".join(header) + data_out)

iq = np.load("D:/Research/CoNi67/Data/3D/iq.npy")[0, :, :, 0]
xc = np.load("D:/Research/CoNi67/Data/3D/xc.npy")[0, :, :, 0]
ang = read_ang("D:/Research/CoNi67/Data/temp/1.ang")
ang["IQ"] = iq.copy()
ang["CI"] = xc.copy()
write_ang(ang)


ang = read_ang("D:/Research/CoNi67/Data/temp/1_new.ang")
iq = ang["IQ"]
xc = ang["CI"]
# Save CI and IQ as tif images using float32
io.imsave("D:/Research/CoNi67/Data/temp/CI.tif", xc.astype(np.float32))
io.imsave("D:/Research/CoNi67/Data/temp/IQ.tif", iq.astype(np.float32))
