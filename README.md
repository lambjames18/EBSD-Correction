# Distortion correction for EBSD/BSE images

Contains python files for correcting distorted EBSD images using reference BSE images.

CONDA ENV: `conda create -n align python numpy matplotlib h5py imageio scipy scikit-learn scikit-image`


## Usage

Simply run `python ui.py` to run the GUI. The initial window shows two blank image viewers, an array of buttons/dropdowns that are used during control point selection.

### Importing data

The "File" tab provides the ability to open data. You will be prompted to select a distorted file, typically an EBSD file, and a control file, typically a BSE image. The distorted file can be an image (tif, tiff, png) or an EBSD data file (ang). The control file needs to be an image (tif, tiff, png).

### Selecting control points

Once the data is imported, you can select the EBSD data modality (assuming you read in an EBSD data file) using the second drop down. The first dropdown is for selecting the slice number (in the case of 3D data, not applicable to 2D data). The "apply CLAHE" button will turn on local histogram equalization for the control image (sometimes helpful in selecting control points).
