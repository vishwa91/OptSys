#!/usr/bin/env python3

import sys
sys.path.append('../modules')

import numpy as np
import matplotlib.pyplot as plt

import raytracing as rt
import visualize as vis
import ray_utilities

if __name__ == '__main__':
    # System to test a field lens.
    # Set f_field to 4e1 to get maximum throughput, and 4e9 (or something large)
    # to disable its effects

    # Constants
    ffl = 45            # Flange focal length
    f_field = 4e1       # Focal length of field lens
    aperture = 100      # Aperture of each lens.
    nrays = 20          # Number of rays per scene point
    npoints = 5         # Number of scene points
    image_plane = -300  # Position of image plane
    ymax = 200          # Limit of image plane
    ymin = -200

    # Create a scene. Hehe
    scene = np.zeros((2, npoints))
    scene[0, :] = image_plane
    scene[1, :] = np.linspace(ymin, ymax, npoints)

    # Create an objective lens
    components = []
    components.append(rt.Lens(f=-image_plane,
                              aperture=aperture,
                              pos=[-20,0],
                              theta=0))
    components.append(rt.Lens(f=ffl,
                              aperture=aperture,
                              pos=[0,0],
                              theta=0))

    # Add a field lens.
    components.append(rt.Lens(f=f_field,
                              aperture=aperture,
                              pos=[ffl,0],
                              theta=0))

    # Add a relay pair
    components.append(rt.Lens(f=50,
                              aperture=aperture,
                              pos=[ffl+50, 0],
                              theta=0))
    components.append(rt.Lens(f=50,
                              aperture=aperture,
                              pos=[ffl+50+10, 0],
                              theta=0))

    # Add an image sensor
    components.append(rt.Sensor(aperture=aperture,
                                pos=[ffl+110, 0],
                                theta=0))

    # Get initial rays
    [rays, ptdict, colors] = ray_utilities.initial_rays(scene,
                                                        components[0],
                                                        nrays)

    # Propagate rays without sensor to compute vignetting
    ray_bundles = rt.propagate_rays(components[:-1], rays)
    vignetting = ray_utilities.vignetting(ray_bundles, ptdict)
    print('Vignetting: {}'.format(vignetting))

    # Propagate the rays to draw
    ray_bundles = rt.propagate_rays(components, rays)

    # Create a new canvas
    canvas = vis.Canvas([image_plane, ffl+130], [ymin, ymax])

    # Draw the rays
    colors = vis.get_colors(npoints, nrays, flatten=True)
    canvas.draw_rays(ray_bundles, colors)

    # Draw the components
    canvas.draw_components(components)

    # Show the canvas
    canvas.show()

    # Save the canvas
    canvas.save('field_lens.png')
