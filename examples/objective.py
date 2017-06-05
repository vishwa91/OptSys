#!/usr/bin/env python3

import os, sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

import raytracing as rt
import visualize as vis
import ray_utilities

if __name__ == '__main__':
    # Create a relay lens system
    components = []
    rays = []
    image_plane = -300
    nrays = 10

    # Objective is simulated using two lenses
    components.append(rt.Lens(f=30,
                              aperture=100,
                              pos=[0,0],
                              theta=0))

    # Second lens creates the flange focal distance
    components.append(rt.Lens(f=-13,
                              aperture=50,
                              pos=[20,0],
                              theta=0))

    # Create three points and three rays from each point
    rays += ray_utilities.ray_fan([image_plane, 200], [-np.pi/5, -np.pi/6], nrays)
    rays += ray_utilities.ray_fan([image_plane, 0], [-np.pi/30, np.pi/30], nrays)
    rays += ray_utilities.ray_fan([image_plane, -200], [np.pi/6, np.pi/5], nrays)

    colors = 'r'*nrays + 'g'*nrays + 'b'*nrays

    # Propagate the rays
    ray_bundles = rt.propagate_rays(components, rays)

    # Create a new canvas
    canvas = vis.Canvas([-300, 100], [-200, 200])

    # Draw the components
    canvas.draw_components(components)

    # Draw the rays
    canvas.draw_rays(ray_bundles, colors)

    # Show the system
    canvas.show()

    # Save a copy
    canvas.save('objective.png')
