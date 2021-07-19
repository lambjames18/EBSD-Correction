# -*- coding: utf-8 -*-


"""
Created on Thu Jan 24 14:51:36 2019

@author: Arsenic
"""

import numpy as np
import scipy.ndimage
from PIL import Image
from skimage import transform as tf
from skimage import util 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from skimage.io import imread
from skimage.transform import resize
import toolsDistortions as tools


# parameters of the polynomial function
deg = 2


# Read 3d feats map map
featmap = imread("D:\\MariePython\\Ti7_Chess\\alignment\\slice_76.tiff")

featmap_trans_im = Image.fromarray(featmap)
featmap_trans_im.save("D:\\MariePython\\Ti7_Chess\\alignment\\feat_ids_76.tiff")

# Read control points from files
coord_ebsd = np.loadtxt(open("D:\\MariePython\\Ti7_Chess\\alignment\\ctr_pts_ebsd.txt", "rb"), 
                  delimiter="\t").astype(int)
coord_bse = np.loadtxt(open("D:\\MariePython\\Ti7_Chess\\alignment\\ctr_pts_bse.txt", "rb"), 
                  delimiter="\t").astype(int)
nb_of_points = coord_ebsd.shape[0]

def swap_cols(arr, frm, to):  # swap columns to get from imageJ to Python coordinates system
    arr[:,[frm, to]] = arr[:,[to, frm]]

# swap_cols(coord_ebsd, 0, 1)
# swap_cols(coord_bse, 0, 1)

dist_ebsd = tools.successiveDistances(coord_ebsd)
dist_dic = tools.successiveDistances(coord_bse)
ratio = tools.findRatio(dist_dic, dist_ebsd)

# Convert control points coordinates into same reference frame
coord_bse_rescaled = coord_bse/ratio
src = coord_ebsd  # sources
dst = coord_bse_rescaled  # targets



# Define the polynomial regression
model_i = Pipeline([('poly', PolynomialFeatures(degree=deg, include_bias=True)),
                            ('linear', LinearRegression(fit_intercept=False, normalize=False))])

model_j = Pipeline([('poly', PolynomialFeatures(degree=deg, include_bias=True)),
                            ('linear', LinearRegression(fit_intercept=False, normalize=False))])

# Solve the regression system
model_i.fit(src, dst[:, 0])
model_j.fit(src, dst[:, 1])

# Define the image transformation
params = np.stack([model_i.named_steps['linear'].coef_,
                   model_j.named_steps['linear'].coef_], axis=0)

# Distort image
transform = tf._geometric.PolynomialTransform(params)
featmap_align = tf.warp(featmap, transform,
                                              cval=0,  # new pixels are black pixel
                                              order=0,  # k-neighbour
                                              preserve_range=True)

featmap_trans_im = Image.fromarray(featmap_align)
featmap_trans_im.save("D:\\MariePython\\Ti7_Chess\\alignment\\feat_ids_76_align.tiff")

