#!/usr/bin/env python

'''
    Module for rendering simple optical outputs using raytracing.
'''

# System imports
import os
import sys
import pdb

# Scipy and related imports
import numpy as np
import scipy.linalg as lin
import scipy.ndimage as ndim
import matplotlib.pyplot as plt
import Image

# ------------------- Aperture Focus Image routines ---------------------------#
def render_point(f, beta, d, screen_lim, res, h, x, gaussian_blur=True,
                 gamma=0.7):
    '''
        Render the image of a single point in a 2D setting with a thin lens
        model.

        Inputs:
            f: Focal length of the lens.
            beta: Aperture radius from optical axis.
            d: Distance of the screen from the lens.
            screen_lim: Limits of the screen along vertical axis.
            res: Resolution of the screen in terms of distance between two
                consecutive points.
            h: Height of the point from optical axis.
            x: Distance of the point from the lens.
            gaussian_blur: If True, use a gaussian blur for out of focus point,
                else use a pillbox blur model.
            gamma: Factor for gamma correction of the image.

        Outputs:
            screen: Array representing screen coordinates.
            image: Projection of the point on the screen.
            h0: Height of the image formed at the focal point.
            d0: Distance of the image formed at the focal point.
    '''
    # Numerical stability
    f += 1e-15

    # Compute the point at which the image will be formed using thin lens
    # model.
    d0 = f*x/(x-f)

    # compute the height at which the image will actually be formed.
    h0 = -f*h/(x-f)

    # compute the position of the image on the sensor
    hs = -h*d/x

    # Compute the extent of blur.
    h1 = -d*(-h0+beta)/d0 + beta
    h2 = -d*(-h0-beta)/d0 - beta
    dh = abs(h2-h1) + 1e-10

    [llim, ulim] = screen_lim

    # Construct the screen and image.
    screen = np.arange(llim, ulim, res)
    image = np.zeros_like(screen)

    # Assume unit intensity for now.
    if gaussian_blur:
        image = np.exp(-pow(screen-hs, 2)/pow(dh, 2))/np.sqrt(2*np.pi)/dh
        image -= image[abs(screen-h1).argmin()]
    else:
        image[:] = 1

    # Clip and normalize. See if screen is in front or behind focus point.

    if d > d0:
        image *= (screen >= h1)*(screen <= h2)
    else:
        image *= (screen >= h2)*(screen <= h1)

    imsum = image.sum()

    # If imsum is zero, it means that the image is focussed.
    if imsum == 0:
        image[abs(screen-hs).argmin()] = 1/np.sqrt(2*np.pi)/dh

    # Do a gamma correction.
    image = pow(abs(image), gamma)

    # Done. We can return the image.
    return screen, image, h0, d0

def afi_point(f_array, beta_array, d, screen_lim, res, h, x,
               gaussian_blur=True):
    '''
        Render an aperture focus stack for a single point using a thin lens
        model.

        Inputs:
            f_array: Focal length of the lens.
            beta_array: Array of aperture values.
            d: Distance between lens and sensor.
            screen_lim: Coordinate limits of sensor.
            res: Resolution of the sensor.
            h: Height of the point from optical axis.
            x: Distance of the point from the lens.
            gaussian_blur: If True, use the gaussian blur model, else use the
                pillbox blur model.

        Outputs:
            screen: Array representing screen coordinates.
            AF: Aperture focus stack.
            h0_array: Array of original focus point of the object.
            d0_array: Array of distances from lens of focus point.
    '''
    pdb.set_trace()
    # Compute screen for one configuration.
    [screen, img, h0, d0] = render_point(f_array[0], beta_array[0], d,
                                         screen_lim, res, h, x,
                                         gaussian_blur)

    # Create the AF object.
    AF = np.zeros((len(f_array), len(beta_array), len(screen)))
    h0_array = np.zeros(len(f_array))
    d0_array = np.zeros(len(f_array))

    for f_idx, f in enumerate(f_array):
        for b_idx, beta in enumerate(beta_array):
            [_, im, h0, d0] = render_point(f, beta, d, screen_lim, res, h, x,
                                           gaussian_blur)
            AF[f_idx, b_idx] = im
            h0_array[f_idx] = h0
            d0_array[f_idx] = d0

    # Compute the screen index where the image is focussed.
    focus_idx = abs(d0_array - d).argmin()
    im_idx = abs(screen - h0_array[focus_idx]).argmin()

    # Done.
    return screen, AF, im_idx

def afi_render(f_array, beta_array, d, screen_lim, res, scene,
               gaussian_blur=True):
    '''
        Render an aperture focus stack for a simple scene using a thin lens
        model.

        Inputs:
            f_array: Focal length of the lens.
            beta_array: Array of aperture values.
            d: Distance between lens and sensor.
            screen_lim: Coordinate limits of sensor.
            res: Resolution of the sensor.
            scene: A 3-tuple representing the scene geometry:
                scene[1]: Heights of points from optical axis
                scene[2]: Distance of poitns from the lens
                scene[3]: Intensities of the points (0-1).
            gaussian_blur: If True, use the gaussian blur model, else use the
                pillbox blur model.

        Outputs:
            screen: Array representing screen coordinates.
            AF: Aperture focus stack.
            im_idx_array: Array of screen coordinates where each scene point
                is best focused.
    '''
    # Unpack the scene
    [H, X, I] = scene

    # Compute an AFI for dimensions.
    [screen, AF, im_idx] = afi_point(f_array, beta_array, d, screen_lim,
                                     res, H[0], X[0], gaussian_blur)
    im_idx_array = np.zeros(len(H))

    # Use the above AFI and im_idx
    AF *= I[0]
    im_idx_array[0] = im_idx

    # For now, the AFIs are just added up. No obstacle based ray tracing done.
    for idx in range(1, len(H)):
        [_, AF_tmp, im_idx] = afi_point(f_array, beta_array, d,
                                        res, H[idx], X[idx],
                                        gaussian_blur)
        AF += AF_time*I[idx]
        im_idx_array[idx] = im_idx

    return screen, AF, im_idx_array

def plot_af(AF, screen):
    '''
        Plot the superimposed images of AF stack.

        Inputs:
            AF: Aperture focus stack.
            screen: Coordinates of the sensor screen.

        Outputs:
            None
    '''
    [nf, na, ns] = AF.shape

    for f_idx in range(nf):
        for a_idx in range(na):
            plt.plot(screen, AF[f_idx, a_idx, :])

    plt.show()

def show_af_slice(AF, im_idx, zoom_factor=1.0, save=False, fname=None,
                  show=True):
    '''
        Show the AF image slice for a 2D image setting.

        Inputs:
            AF: The aperture focus stack
            im_idx: Index to extract the AF slice from.
            zoom_factor: Amount to zoom the image by.
            save: If True, save the image.
            fname: If save is true, save the image using this name.
            show: If False, just save and don't show.

        Output:
            None.
    '''
    AF_im = abs(AF[:, :, im_idx])
    AF_im = ndim.zoom(AF_im.T*255/AF_im.max(), zoom_factor)
    if show:
        Image.fromarray(AF_im).show()

    if save:
        Image.fromarray(AF_im.astype(np.uint8)).save(fname)

# --------------------- Light field routines ----------------------------------#
def lf_point(f, d, lens_lim, screen_lim, lens_res, screen_res, h, x,
             intensity=1):
    '''
        Render the light field output of a single point.

        Inputs:
            f: Focal length of the lens.
            d: Distance of the screen from the lens.
            lens_lim: Limits of the lens plane(u).
            screen_lim: Limits of the screen plane(s).
            lens_res: Resolution of sampling of the lens plane.
            screen_res: REsolution of sampling of the screen plane.
            h: Distance of point from the lens.
            x: Height of point from the optical axis.
            intensity: Radiance emitted by the point. Default is 1.

        Outputs:
            lens: Coordinates of sampling on the lens.
            screen: Coordinates of sampling on the screen.
            h0: Focal point of the image.
            LF: Light field image, LF(u, s)
    '''
    # Compute the point at which the image will be formed using thin lens
    # model.
    d0 = f*x/(x-f)

    # compute the height at which the image will actually be formed.
    h0 = -f*h/(x-f)

    # compute the position of the image on the sensor
    hs = -h*d/x

    [llim_l, ulim_l] = lens_lim

    # Compute the extent of light-field.
    h1 = -d*(-h0+ulim_l)/d0 + ulim_l
    h2 = -d*(-h0-llim_l)/d0 - llim_l
    dh = abs(h2-h1) + 1e-10

    [llim_s, ulim_s] = screen_lim

    # Construct the screen, lens and light field.
    screen = np.arange(llim_s, ulim_s, screen_res)
    lens = np.arange(llim_l, ulim_l, lens_res)

    LF = np.zeros((len(screen), len(lens)))

    # Compute the light field coordinates.
    S = lens - (d/d0)*(lens-h0)

    for idx, u in enumerate(lens):
        s = S[idx]
        LF[abs(screen-s).argmin(), idx] = intensity

    return lens, screen, h0, LF

def lf_render(f, d, lens_lim, screen_lim, lens_res, screen_res, scene):
    '''
        Render the light field output of a single point.

        Inputs:
            f: Focal length of the lens.
            d: Distance of the screen from the lens.
            lens_lim: Limits of the lens plane(u).
            screen_lim: Limits of the screen plane(s).
            lens_res: Resolution of sampling of the lens plane.
            screen_res: Resolution of sampling of the screen plane.
            scene: A 3-tuple representing the scene geometry:
                scene[1]: Heights of points from optical axis
                scene[2]: Distance of poitns from the lens
                scene[3]: Intensities of the points(0-1).

        Outputs:
            lens: Coordinates of sampling on the lens.
            screen: Coordinates of sampling on the screen.
            h0: Focal point of the image.
            LF: Light field image, LF(u, s)
    '''
    # Extract the scene:
    [H, D, I] = scene

    # Sort the scene by distances from lens. A point nearer to the lens will
    # obstruct anything behind, hence the sorting.
    sort_idx = np.argsort(D)[::-1]

    # Render an example first point LF.
    idx = sort_idx[0]
    [lens, screen, h0, LF] = lf_point(f, d, lens_lim, screen_lim, lens_res,
                                      screen_res, H[idx], D[idx], I[idx])
    # We want to save the focus heights as well.
    h0_array = np.zeros_like(H)

    for idx in sort_idx[1:]:
        # Compute a new LF for each point.
        [_, _, h0, lf] = lf_point(f, d, lens_lim, screen_lim, lens_res,
                                  screen_res, H[idx], D[idx], I[idx])
        # The new computed light field covers the previous light field.
        LF[lf != 0] = lf[lf != 0]
        h0_array[idx] = h0

    # Done.
    return lens, screen, h0_array, LF
