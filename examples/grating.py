#!/usr/bin/env python3

import os, sys
sys.path.append('../modules')

import numpy as np
import matplotlib.pyplot as plt

import raytracing as rt
import visualize as vis
import ray_utilities

if __name__ == '__main__':
    # Create simple system to test diffraction grating
    components = []
    rays = []
    image_plane = -200
    nrays = 20

    # Create three scene points
    scene = np.zeros((2, 3))
    scene[0, :] = image_plane
    scene[1, 0] = 20
    scene[1, 1] = 0
    scene[1, 2] = -20

    # Imaging lens
    components.append(rt.Lens(f=100,
                              aperture=10,
                              pos=[0,0],
                              theta=0))
    # Grating
    components.append(rt.Grating(ngroves=600,
                                 aperture=100,
                                 pos=[200, 0],
                                 theta=0))
    # Reimaging lens
    components.append(rt.Lens(f=50,
                              aperture=100,
                              pos=[300, -25],
                              theta=np.pi/12))

    # Get the initial rays
    [rays, ptdict, colors] = ray_utilities.initial_rays(scene,
                                                        components[0],
                                                        nrays)
    # Colors and three wavelengths
    colors = ['r'*nrays*3, 'g'*nrays*3, 'b'*nrays*3];
    wavelengths = [488e-9, 514e-9, 633e-9]

    # Create a new canvas
    canvas = vis.Canvas([-200, 600], [-100, 100])

    # Draw the components
    canvas.draw_components(components)

    # Draw the rays for each wavelength
    for idx in range(len(wavelengths)):
        canvas.draw_rays(rt.propagate_rays(components, rays,
                                           lmb=wavelengths[idx]), colors[idx])

    # Show the system
    canvas.show()

    # Save a copy
    canvas.save('grating.png')
