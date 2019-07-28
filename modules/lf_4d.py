#!/usr/bin/env python

'''
    I am trying to test operations on lightfield images.
'''

import os
import sys
import time

import numpy as np
import scipy.linalg as lin
import scipy.ndimage as ndim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Image

def LF_shape_check(LF):
    '''
        Cute function to check if the light-field is a grayscale one or RGB.

        Inputs:
            LF: Light-field object, 4D if grayscale and 5D if RGB

        Outputs:
            lf_shape: First four dimensions of the lightfield
            isrgb: If LF is an RGB object, then return True
    '''
    lf_shape = LF.shape
    if len(lf_shape) == 4:
        return lf_shape, False
    else:
        return lf_shape[:4], True

def get_lenslet(LF):
    '''
        Create a lenslet image from the Light field object.
    '''
    [lf_shape, isrgb] = LF_shape_check(LF)
    [U, V, H, W] = lf_shape

    if isrgb:
        lenslet_image = np.zeros((U*H, V*W, 3))
    else:
        lenslet_image = np.zeros((U*H, V*W))

    for idx1 in range(U):
        for idx2 in range(V):
            if isrgb:
                lenslet_image[idx1:U*H:U, idx2:V*W:V, :] =\
                        LF[idx1, idx2, :, :, :]
            else:
                lenslet_image[idx1:U*H:U, idx2:V*W:V] = LF[idx1, idx2, :, :]

    return lenslet_image

def af_slice(AF, x, y, zoom_factor=1.0, save=False, fname=None):
    '''
        Display an AF image at given coordinates.

        Inputs:
            AF: Aperture-Focus stack.
            x: Horizontal coordinate of the image.
            y: Vertical coordinate of the iamge.
            zoom_factor: Factor to resize the image by.
            save: If True, save the image.
            fname: If save is True, save the image using this name.

        Outputs:
            None
    '''
    [af_shape, isrgb] = LF_shape_check(AF)

    if isrgb:
        AF_img = ndim.zoom(AF[:, :, x, y, :], (zoom_factor, zoom_factor, 1))
    else:
        AF_img = ndim.zoom(AF[:, :, x, y], zoom_factor)

    if save:
        Image.fromarray(AF_img.astype(np.uint8)).save(fname)

    Image.fromarray(AF_img.astype(np.uint8)).show()

def LF2AF(LF):
    '''
        Convert Light-field stack to Aperture-Focus stack.
    '''
    [lf_shape, isrgb] = LF_shape_check(LF)
    [U, V, H, W] = lf_shape

    AF = np.zeros_like(LF)

    # Assume U = V for now.
    c = U//2
    aperture_array = np.linspace(0, c*np.sqrt(2), U+1)[1:]
    focus_array = np.linspace(-1, 1, U)

    for a_idx, aperture in enumerate(aperture_array):
        for f_idx, focus in enumerate(focus_array):
            AF[a_idx, f_idx] = get_af_image(LF, aperture, focus)

    return AF

def get_af_image(LF, aperture=None, focus=0):
    '''
        Get an image from light field with a specified aperture and focus
        settings.
    '''
    [lf_shape, isrgb] = LF_shape_check(LF)
    [U, V, H, W] = lf_shape

    # Default aperture setting is full open.
    if aperture is None:
        aperture = np.sqrt(2)*(U//2)

    # Get aperture indices
    c = U//2
    x = np.arange(U, dtype=float)
    y = np.arange(V, dtype=float)

    [X, Y] = np.meshgrid(x, y)
    [aperture_x, aperture_y] = np.where(np.hypot(X-c, Y-c) <= aperture)

    # Now get Focused lightfield.
    Xoffset_array = np.linspace(-0.5, 0.5, U)*focus*U
    Yoffset_array = np.linspace(-0.5, 0.5, V)*focus*V

    LF_shift = np.zeros_like(LF)
    for idx1 in range(U):
        Xoffset = Xoffset_array[idx1]
        for idx2 in range(V):
            Yoffset = Yoffset_array[idx2]

            # Shift the image and reassign it to LF.
            if isrgb:
                LF_shift[idx1, idx2, :, :, :] = ndim.shift(
                                                    LF[idx1, idx2, :, :, :],
                                                    [Xoffset, Yoffset, 0])
            else:
                LF_shift[idx1, idx2, :, :] = ndim.shift(LF[idx1, idx2, :, :],
                                                        [Xoffset, Yoffset])

    # Done. Now get the image.
    if isrgb:
        af_image = LF_shift[aperture_x, aperture_y, :, :, :].mean(0)
    else:
        af_image = LF_shift[aperture_x, aperture_y, :, :].mean(0)

    return af_image

def aperture_change(LF):
    '''
        Create images with changing aperture settings.
    '''
    [lf_shape, isrgb] = LF_shape_check(LF)
    [U, V, H, W] = lf_shape

    # No sanity check done now. Assume U = v
    c = U//2

    radii_array = np.linspace(0, c*np.sqrt(2), U+1)[1:]

    # Create mesh grid of indices to sweep through radii and find the indices.
    x = np.arange(U, dtype=float)
    y = np.arange(V, dtype=float)

    [X, Y] = np.meshgrid(x, y)

    # Store the variable aperture images here.
    images = []

    for idx, radius in enumerate(radii_array):
        [idx_x, idx_y] = np.where(np.hypot(X-c, Y-c) <= radius)
        if isrgb:
            im = LF[idx_x, idx_y, :, :, :].sum(0)/len(idx_x)
        else:
            im = LF[idx_x, idx_y, :, :].sum(0)/len(idx_x)
        images.append(im)

    return images

def focus_change(LF):
    '''
        Create images with changing focus settings.
    '''
    U = LF.shape[0]     # Assuming aperture dimensions are same in both
                        # directions.
    focus_array = np.linspace(-1, 1, U, endpoint=True)
    images = []

    for focus in focus_array:
        images.append(_focus(LF, focus))

    return images

def _focus(LF, focus_val):
    '''
        Focus the light-field at a given focal value.
    '''
    # I am writing this mostly from LFFiltShiftSum from LFToolbox0.4
    [lf_shape, isrgb] = LF_shape_check(LF)
    [U, V, H, W] = lf_shape

    # Image indices
    [X, Y] = np.meshgrid(range(H), range(W))

    # Shifts in each direction.
    Xoffset_array = np.linspace(-0.5, 0.5, U)*focus_val*U
    Yoffset_array = np.linspace(-0.5, 0.5, V)*focus_val*V

    LF_shift = np.zeros_like(LF)
    for idx1 in range(U):
        Xoffset = Xoffset_array[idx1]
        for idx2 in range(V):
            Yoffset = Yoffset_array[idx2]

            # Shift the image and reassign it to LF.
            if isrgb:
                LF_shift[idx1, idx2, :, :, :] = ndim.shift(
                                                    LF[idx1, idx2, :, :, :],
                                                    [Xoffset, Yoffset, 0])
            else:
                LF_shift[idx1, idx2, :, :] = ndim.shift(LF[idx1, idx2, :, :],
                                                        [Xoffset, Yoffset])

    # Now add up the light-field slices.
    return LF_shift.mean(0).mean(0)

if __name__ == '__main__':
    # Load the Aperture focus stack
    AF = np.load('results/lego_af.npy')
