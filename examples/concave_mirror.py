#!/usr/bin/env python3

import sys
import os
sys.path.append('../modules')

import numpy as np
import matplotlib.pyplot as plt

import raytracing as rt
import visualize as vis
import ray_utilities

if __name__ == '__main__':
    # Constants
    image_plane = -1e6  # Image plane from first lens
    fs = 100            # System focal length
    aperture = 25.4     # Diameter of each mirror
    npoints = 3         # Number of scene points
    ymax = 1e5          # Size of imaging area
    ymin = -1e5    
    nrays = 10          # Number of rays per scene point
    lmb = 500           # Design wavelength

    # Create a scene
    scene = np.zeros((2, npoints))
    scene[0, :] = image_plane
    scene[1, :] = np.linspace(ymin, ymax, npoints)

    components = []

    # Add a concave mirror
    components.append(rt.SphericalMirror(f=fs,
                                        aperture=aperture,
                                        pos=[0, 0],
                                        theta=0))
    
    # Place a detector just on the focal plane of the mirror
    components.append(rt.Sensor(aperture=aperture,
                                pos=[-fs, 0],
                                theta=np.pi))
    
    # Get initial rays
    [rays, ptdict, colors] = ray_utilities.initial_rays(scene,
                                                        components[0],
                                                        nrays)
    # Color code the rays emanating from each scene point
    colors = ['b']*nrays + ['g']*nrays + ['r']*nrays

    # Create a new canvas
    canvas = vis.Canvas(xlim=[-2*fs, 5],
                        ylim=[-aperture, aperture],
                        figsize=[12, 12])

    ray_bundles = rt.propagate_rays(components, rays, lmb)
    canvas.draw_rays(ray_bundles, colors)

    # Draw the components
    canvas.draw_components(components)

    # Show the canvas
    canvas.show()
