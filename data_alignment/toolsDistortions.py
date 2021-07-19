# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:05:12 2019

@author: Arsenic
"""

import numpy as np
from skimage import transform as tf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# functions and tools for finding and applying various distortions functions btw EBSD and DIC

# Scaling / Resizing functions
def successiveDistances(array):
    """ Calculates euclidian distance btw sets of points in an array    """
    nb_of_points = int(np.ma.size(array)/2)
    diffs = np.zeros(nb_of_points-1, dtype = float)
    for i in range (0, nb_of_points-1):
        diffs[i] = (array[i][0]-array[i+1][0])**2+(array[i][1]-array[i+1][1])**2
    dists = np.sqrt(diffs, dtype=float)
    return dists

def findRatio(array1, array2):
    """ Finds scaling ratio btw 2 arrays of reference points on their own grids    """
    ratios = np.divide(array1, array2)
    ratio = np.average(ratios)
    return ratio

def reshape(array, parameter):
    reshapedArray = np.reshape(array, (-1, parameter))
    return reshapedArray

# Finding parameters and applying distortion functions
def findPolyTransform(src, dst, deg, params): 
    """ Finds polynomial transformation for given set of points in the same ref grid    """

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
    print("Polynomial regression parameters:")
    print(" - xx : {} \n - yy : {}".format(params[0], params[1]))
    # Distort image
    transform = tf._geometric.PolynomialTransform(params)
    return transform    

def applyPolyTransform(array, params):
        """ Applies known polynomial transform    """
        transform = tf._geometric.PolynomialTransform(params)
        arrayPolyDistort = tf.warp(array, transform,
                                              cval=0,  # new pixels are black pixel
                                              order=0,  # k-neighbour
                                              preserve_range=True)
        return arrayPolyDistort

def applyTranslation(array, translation):
        """ Applies known translation    """
        affine_tform = tf.AffineTransform(matrix=None, scale=None, rotation=None, shear=None, translation=translation)
        arrayTranslated= tf.warp(array, affine_tform, cval=0,  # new pixels are black pixel
                                              order=0,  # k-neighbour
                                              preserve_range=True)
        return arrayTranslated



