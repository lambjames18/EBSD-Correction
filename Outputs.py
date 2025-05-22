import os
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import h5py

import Inputs


def write_ang_file(reference_path, save_path, data_dict):
    _, _, col_names, res, header_string, _ = Inputs.read_ang_header(reference_path)

    data_dict.pop("EulerAngles")
    ebsd_keys = list(data_dict.keys())
    print(data_dict.keys(), ebsd_keys)
    src_imgs = np.array([data_dict[key][0] for key in ebsd_keys], dtype=float)

    # Create the grid for the ang file
    x, y = np.meshgrid(np.arange(src_imgs[0].shape[1]), np.arange(src_imgs[0].shape[0]))
    x.flatten().astype(np.float32) * res
    y.flatten().astype(np.float32) * res
    src_imgs[col_names.index("x")] = x
    src_imgs[col_names.index("y")] = y

    # Modify the header according to the new grid
    nrows_index = header_string.index("# NROWS: ")
    end_index = header_string[nrows_index:].index("\n") + nrows_index
    header_string = header_string.replace(
        header_string[nrows_index:end_index], f"# NROWS: {src_imgs[0].shape[0]}"
    )
    ncolsE_index = header_string.index("# NCOLS_EVEN: ")
    end_index = header_string[ncolsE_index:].index("\n") + ncolsE_index
    header_string = header_string.replace(
        header_string[ncolsE_index:end_index], f"# NCOLS_EVEN: {src_imgs[0].shape[1]}"
    )
    ncolsO_index = header_string.index("# NCOLS_ODD: ")
    end_index = header_string[ncolsO_index:].index("\n") + ncolsO_index
    header_string = header_string.replace(
        header_string[ncolsO_index:end_index], f"# NCOLS_ODD: {src_imgs[0].shape[1]}"
    )

    # Save the data based on the extension
    data_out = []
    for i, key in enumerate(ebsd_keys):
        if key in col_names:
            data_out.append(src_imgs[i].reshape(-1, 1).astype(np.float32))
    data_out = np.hstack(data_out)
    np.savetxt(
        save_path,
        data_out,
        header=header_string,
        fmt="  %.5f   %.5f   %.5f    %.5f    %.5f %.1f  %0.3f %d %d %0.3f %0.5f %0.5f %0.5f",
        comments="",
    )


def format_with_variable_spacing(data, file_path, spacing_rules=None):
    """
    Save data to a text file with variable spacing between columns.

    Parameters:
    -----------
    data : numpy.ndarray
        The data to save
    file_path : str
        Path to the output file
    spacing_rules : list of dict, optional
        List of dictionaries defining spacing rules for each column.
        Each dict should have 'max_val' keys and corresponding space counts.
        If None, standard spacing is used.
    """
    fmt_strs = [
        "%.5f",
        "%.5f",
        "%.5f",
        "%.5f",
        "%.5f",
        "%.1f",
        "%0.3f",
        "%d",
        "%d",
        "%0.3f",
        "%0.5f",
        "%0.5f",
        "%0.5f",
    ]
    n_digits = data.max(axis=1) // 10 + 1

    if spacing_rules is None:
        spacing_rules = [None] * data.shape[1]

    with open(file_path, "w") as f:
        for row in data:
            line = ""

            # Process each column
            for i, value in enumerate(row):
                # Add spacing before all columns except the first
                if i > 0:
                    # Get spacing rule for this column
                    rule = spacing_rules[i]

                    if rule is None:
                        # Standard spacing (1 space)
                        line += " "
                    else:
                        # Apply variable spacing based on value
                        spaces = 1  # Default spacing
                        for max_val, space_count in sorted(rule.items()):
                            if value < max_val:
                                spaces = space_count
                                break
                        line += " " * spaces

                # Add formatted value
                if isinstance(value, float) and value.is_integer():
                    line += f"{value:.0f}"
                elif isinstance(value, float):
                    line += f"{value:.4f}"
                else:
                    line += f"{value}"

            line += "\n"
            f.write(line)
