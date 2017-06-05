#!/usr/bin/env python3

import os, sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

import raytracing as rt
import visualize as vis

if __name__ == '__main__':
    # Create simple system to reimage using lens with scheimpflug
    components = []
    rays = []
    image_plane1 = -150
    image_plane2 = -140
    image_plane3 = -130

    # System contains just of one lens
    components.append(rt.Lens(f=100,
                              aperture=100,
                              pos=[0,0],
                              theta=0))

    # Create three points and three rays from each point
    rays.append([image_plane1, 10, -np.pi/20])
    rays.append([image_plane1, 10, 0])
    rays.append([image_plane1, 10, np.pi/20])

    rays.append([image_plane2, 0, -np.pi/20])
    rays.append([image_plane2, 0, 0])
    rays.append([image_plane2, 0, np.pi/20])

    rays.append([image_plane3, -10, -np.pi/20])
    rays.append([image_plane3, -10, 0])
    rays.append([image_plane3, -10, np.pi/20])

    colors = 'rrrgggbbb'

    # Propagate the rays
    ray_bundles = rt.propagate_rays(components, rays)

    # Create a new canvas
    canvas = vis.Canvas([-200, 600], [-100, 100])

    # Draw the components
    canvas.draw_components(components)

    # Draw the rays
    canvas.draw_rays(ray_bundles, colors)

    # Show the system
    canvas.show()

    # Save a copy
    canvas.save('schiempflug.png')
