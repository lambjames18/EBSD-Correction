import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


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
    for key in out.keys():
        if key not in ["EulerAngles"]:
            out[key] = np.fliplr(np.rot90(out[key], k=3))
        if len(out[key].shape) == 2:
            out[key] = out[key].T
        else:
            out[key] = out[key].transpose((1, 0, 2))
        out[key] = out[key].reshape((1,) + out[key].shape)
    
    # Get the grain file if it exists
    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0] + "_Grain.txt"
    if os.path.exists(os.path.join(dirname, basename)):
        grain_path = os.path.join(dirname, basename)
        grain_data = read_grainFile(grain_path)
        out["GrainIDs"] = grain_data.reshape((1,) + (nrows, ncols))
    return out, res


def read_grainFile(path):
    with open(path, "r") as f:
        for line in f:
            if line[0] == "#" and "Grain ID" in line:
                column = int(line.split(": ")[0].replace("#", "").replace("Column", "").strip())
                break

    grain_data = np.genfromtxt(path, comments="#", delimiter="\n", skip_header=1, dtype=str)
    f = lambda x: x.replace("      ", " ").replace("     ", " ").replace("    ", " ").replace("   ", " ").replace("  ", " ").split(" ")
    grainIDs = np.array([f(x)[column - 1] for x in grain_data]).reshape(-1).astype(np.uint32)
    grainIDs[grainIDs <= 0] = 0
    return grainIDs



path = "C:\\Users\\PollockGroup\\Desktop\\Leah to James\\Stitch\\C-103_50um_DIC\\Corrected\\crop20000\\EBSD-Corrected\\James Use This\\map20231022124105186_Spherical_cleaned_cropped-20000-DIC.ang"

data, _ = read_ang(path)

print(data.keys())
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
ax[0].imshow(data["IQ"][0], cmap="gray")
ax[1].imshow(data["GrainIDs"][0], cmap="viridis")
plt.show()
