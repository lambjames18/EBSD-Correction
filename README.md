# Distortion correction for EBSD/BSE images

Contains python files for correcting distorted EBSD images using reference BSE images.

CONDA ENV: `conda create -n align python numpy matplotlib h5py imageio scipy scikit-learn scikit-image tifffile`


## Usage

Simply run `python ui.py` to run the GUI.

### Initial window

The initial window (shown below) shows two blank image viewers (A and B). These will contiaon the distorted (left) and the control (right) images. On the top row:
- C: a checkbutton that toggles showing the selected control points on the images or hiding them.
- D: a slice number selection dropdown that is used for 3D data.
- E: EBSD modality selection dropdown. If EBSD data is selected (dream3d file for 3D, ang file for 2D), this will allow one to toggle between confidence index, image quality, euler angles, etc. If just an EBSD image is selected, it will remain as "Intensity".
- F: Export control point images upon button click. This is largely for illustrative purposes. It will save two images that have the identified control points overlaid on them.
- G: Control point inheritance button. This is only applicable for 3D, but it allows you to copy over identified control points from another slice to the current slice.
- H: Activate CLAHE. This will apply contrast local histogram equalization to the control image. Often useful for making it easier to identify control points.
- I: View slice. This button will bring up a new window that allows one to view the corrected data on the current slice only. Control points must be selected before viewing the slice or else the correction algorithm doesn't know what to do. It has two slices that allows one to view both the corrected image and the control image (to inspect alignment fidelity).
- J: View the full 3D stack. This will bring up a similar window that allows one to view the corrected data for the full 3D dataset (it is only applicable for 3D data). It has a slider that lets one scroll through the slices of the dataset in addition to the overlay slices.

![image](./theme/UI-Annotated.png "GUI")

### Importing data

The "File" tab provides the ability to open data. You will be prompted to select a control file first, typically a BSE image, and a distorted file second, typically an EBSD file. The distorted file can be an image (tif, tiff) or an EBSD data file (ang). The control file needs to be an image (tif, tiff). A third window will show that allows one to select a file containing identified reference points for the control image. If one wants to start from scratch and not import any points, click "cancel". If one selects a text file of control points, a fourth window will appear for importing points for the distorted image.

**NOTE**: The reference points files are space delimited with 2 columns and N rows (N number of points). The first column is the x position and the second column is the y position. Control point files are automatically saved by the UI during point selection, so you will rarely need to make a file yourself.

**NOTE 2**: The distorted and control images need to have the same pixel resolution. If they do not, higher resolution image will blow up in the UI (adaptive resolution and scaling in the UI is not supported yet). Therefore, it is advised to rescale the higher resolution image (ttypicaly the control image) down to the lower resolution. See the script `rescale.py` for rescaling the data.

### Selecting control points

Simply click on either image to place a reference point. Make sure that the order of the points is consistent between the two viewers (point 1 on distorted should connect to point 1 on control). You can drag points by clicking and dragging on existing points while holding SHIFT. Remove points by right cicking on them.

### Viewing the alignment

Click "View slice" to view the correction generated using the current set of points on the current two images (applicable for both 2D and 3D). Click "Apply to stack" to view the corrections for all slices in a 3D stack (only applicable for 3D data). The popup windows let the user view both the corrected image and the control image.

### Saving the data

Under the "File" tab, "Export 2D" will compute the alignment solution and save the corrected data. The first window will be to save the disorted data (corrected). The second window is for saving the control data (press cancel if you do not want to save either one). The outputs will match what one sees in the interaview view when "View Slice" is selected. "Export 3D" behaves similar, but it creates a new Dream3D file and applies the correction to the data within the new Dream3D file. All outputs are prenamed and will be saved in the location of the distorted data folder.

**NOTE**: The distorted file is only saved as an image currently. In the future, writing an .ang or .h5 file with the corrected data will be supported, but the current state is only to output individual images. Therefore, if one wanted to apply alignment to Confidence Index and Image Quality, one would have to run "Export 2D" twice, each time with a different EBSD mode selected from the dropdown. **The data type is preserved in the saved image, this means that the preview of the image might look wrong, but reading in the data (python, FIJI, etc.) will show the correct data.** For integer data types, the alignemt requires that they go to a float, so the output transforms the float back into the target data type.


## Future plans

Currently overhauling the UI to be more user friendly...

- Adding h5 support for 2D EBSD data
- Adding cropping/rescaling functionality within the UI
- Planning to incorporate automatic control point detection algorithms
- Need to integrate the ability to have multiple control images for each distorted image (imagine toggling between BSE and SE images)
- Plenty of other things...
