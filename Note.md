# Dream3d file alignment

*This is to align a dream3d dataset onto BSE images using manual control points.*


1. pick a slide #, ex 76

2. run dream3d_to_img.py. will create slice_76.tiff

3. Use slice76.tiff and the BSE image to pick control points (ex in ImageJ). report coordinates in ImageJ convention (opposite to numpy array convention). Control points files are called ctr_pts_ebsd.txt and ctr_pts_bse.txt

4. calibrate and test alignment function using distortion_calib.py. the distortion is stored as 'transform'.

5. run distortion_apply.py to the dream3d file. that will modify the arrays within CellData
