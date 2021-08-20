import core
from skimage import io

bse = "8_bse_resized.png"
ebsd = "8_ebsd.png"
# bse_ctr = core.SelectCoords(bse, return_path=True)
# ebsd_ctr = core.SelectCoords(ebsd, return_path=True)
bse_ctr = "ctr_pts_8_bse_resized.txt"
ebsd_ctr = "ctr_pts_8_ebsd.txt"
align = core.Alignment(bse_ctr, ebsd_ctr)
align.TPS(bse, saveSolution=True)
ebsd = io.imread(ebsd)
align.TPS_apply(ebsd)
