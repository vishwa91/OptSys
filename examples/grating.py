#!/usr/bin/env python3

import os, sys
sys.path.append('../modules')

import numpy as np
import matplotlib.pyplot as plt

import raytracing as rt
import visualize as vis
import ray_utilities

if __name__ == '__main__':
    # Create a spectrometer using a simple 4f system and diffraction grating
    f = 50              # Focal length of all lenses
    aperture = 25.4     # Size of lenses
    npoints = 3         # Number of light source points
    nrays = 5           # Number of light rays per point
    ymax = -0.1         # Limit of source plane. Controls spectral resolution
    ymin = 0.1
    ngroves = 600       # Grove density of diffraction grating

    # Simulate system for these wavelengths
    lmb = list(np.linspace(400, 700, 11)*1e-9)

    components = []
    rays = []
    image_plane = -200
    nrays = 20

    # Create three scene points
    scene = np.zeros((2, npoints))
    scene[1, :] = np.linspace(ymin, ymax, npoints)

    # Place a collimation lens
    components.append(rt.Lens(f=f,
                              aperture=aperture,
                              pos=[f, 0],
                              theta=0))

    # Place a diffraction grating
    components.append(rt.Grating(ngroves=ngroves,
                                 aperture=aperture,
                                 pos=[2*f, 0],
                                 theta=0))

    # Place a lens such that the central wavelength is centered on the sensor
    theta_design = np.arcsin(lmb[len(lmb)//2]/(1e-3/ngroves))
    x1 = 2*f + f*np.cos(-theta_design)
    y1 = f*np.sin(-theta_design)

    components.append(rt.Lens(f=f,
                              aperture=aperture,
                              pos=[x1, y1],
                              theta=theta_design))

    # Place a sensor
    x2 = x1 + f*np.cos(-theta_design)
    y2 = y1 + f*np.sin(-theta_design)

    components.append(rt.Sensor(aperture=aperture,
                                pos=[x2, y2],
                                theta=theta_design))

    # Get the initial rays
    [rays, ptdict, colors] = ray_utilities.initial_rays(scene,
                                                        components[0],
                                                        nrays)
    # Create rainbow colors
    colors = vis.get_colors(len(lmb), nrays*npoints, cmap='rainbow')
    
    # Create a new canvas
    canvas = vis.Canvas([-5, 4.1*f], [-2*aperture, 2*aperture])

    # Draw the components
    canvas.draw_components(components)

    # Draw the rays for each wavelength
    for idx in range(len(lmb)):
        canvas.draw_rays(rt.propagate_rays(components, rays,
                                           lmb=lmb[idx]), colors[idx],
                        linewidth=0.2)

    # Show the system
    canvas.show()

    # Save a copy
    canvas.save('grating.png')
