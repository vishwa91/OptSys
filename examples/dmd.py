#!/usr/bin/env python3

import sys
sys.path.append('../modules')

import numpy as np
import matplotlib.pyplot as plt

import raytracing as rt
import visualize as vis
import ray_utilities

if __name__ == '__main__':
    # System to test a DMD imager

    # Constants
    ffl = 45            # Flange focal length
    f_field = 40        # Focal length of field lens
    aperture = 25.4     # Aperture of each lens
    nrays = 20          # Number of rays per scene point
    npoints = 5         # Number of scene points
    image_plane = -300  # Position of image plane
    ymax = 50           # Limit of image plane
    ymin = -50

    # Create a scene parallel to DMD and completely in focus.
    scene = np.zeros((2, npoints))
    scene[0, :] = image_plane
    scene[1, :] = np.linspace(ymin, ymax, npoints)

    # Create a simple objective lens
    components = []
    components.append(rt.Lens(f=-image_plane,
                              aperture=aperture,
                              pos=[-20,0],
                              theta=0))
    components.append(rt.Lens(f=ffl,
                              aperture=aperture,
                              pos=[0,0],
                              theta=0))

    # Add a field lens to increase throughput
    components.append(rt.Lens(f=f_field,
                              aperture=aperture,
                              pos=[ffl,0],
                              theta=0))

    # Reimage using a relay pair
    components.append(rt.Lens(f=100,
                              aperture=aperture,
                              pos=[ffl+100, 0],
                              theta=0))
    components.append(rt.Lens(f=100,
                              aperture=aperture,
                              pos=[ffl+110, 0],
                              theta=0))

    # Add a field lens on DMD to increase throughput
    components.append(rt.Lens(f=100,
                              aperture=aperture,
                              pos=[ffl+210, 0],
                              theta=0))
    # And then a DMD
    components.append(rt.DMD(deflection=-12*np.pi/180,
                             aperture=aperture,
                             pos=[ffl+210, 0],
                             theta=0))

    # The field lens needs repetition
    components.append(rt.Lens(f=100,
                              aperture=aperture,
                              pos=[ffl+210, 0],
                              theta=-np.pi))

    # Refocusing lens
    components.append(rt.Lens(f=30,
                              aperture=aperture,
                              pos=[200, -25],
                              theta=-204*np.pi/180))

    # Get initial rays
    [rays, ptdict, colors] = ray_utilities.initial_rays(scene,
                                                        components[0],
                                                        nrays)

    # Propagate the rays
    ray_bundles = rt.propagate_rays(components, rays)

    # Create a new canvas
    canvas = vis.Canvas([image_plane, 300], [-100, ymax], figsize=[12, 6])

    # Create unique colors for each point
    colors = vis.get_colors(npoints, nrays)
    colors_list = []

    for color in colors:
        colors_list += color

    # Draw the rays
    canvas.draw_rays(ray_bundles, colors_list, linewidth=0.2)

    # Draw the components
    canvas.draw_components(components)

    # Show the canvas
    canvas.show()

    # Save the canvas
    canvas.save('dmd.png')
