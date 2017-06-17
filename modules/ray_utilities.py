#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import raytracing as rt
import visualize as vis

def initial_rays(scene, objective, nrays=10):
    '''
        Function to get an initial set of light rays for a given scene
        configuration and objective lens

        Inputs:
            scene: 2xN matrix of points
            objective: Lens instance for objective lens
            nrays: Number of rays per scene point

        Outputs:
            rays: nrays.N list of rays
            point_ray_dict: List with point to rays correspondence
            colors: nrays.N list of colors
    '''
    rays = []
    colors = []
    point_ray_dict = []

    N = scene.shape[1]
    x0 = objective.pos[0]
    y0 = objective.pos[1]
    r = objective.aperture/2
    theta = np.pi/2-objective.theta

    # Compute extent of objective lens
    x1 = x0 + 0.5*r*np.cos(theta)
    y1 = y0 + 0.5*r*np.sin(theta)

    x2 = x0 - 0.5*r*np.cos(theta)
    y2 = y0 - 0.5*r*np.sin(theta)

    # Now create rays for each scene point
    for idx in range(N):
        theta_min = np.arctan2(y1-scene[1, idx], x1-scene[0, idx])
        theta_max = np.arctan2(y2-scene[1, idx], x2-scene[0, idx])

        rays += ray_fan(scene[:, idx], [theta_min, theta_max], nrays)
        colors += [np.random.rand(3, 1)]*nrays
        point_ray_dict.append(range(idx*nrays, (idx+1)*nrays))

    return rays, point_ray_dict, colors

def ray_fan(pos, theta_lim, nrays):
    '''
        Function to get a fan of rays from a point

        Inputs:
            orig: Origin of  the point
            theta_lim: 2-tuple of angle limits in radians
            nrays: Number of rays to generate

        Output:
            rays: List of rays in 3-tuple format: x-coordinte, y-coordinate and
                  angle
    '''
    rays = []
    angles = np.linspace(theta_lim[0], theta_lim[1], nrays)

    for angle in angles:
        rays.append([pos[0], pos[1], angle])

    return rays

def throughput(ray_bundles):
    '''
        Compute throughput of a system based on number of rays passing through

        Inputs:
            ray_bundles: List of rays. See raytracing.propagate_rays()

        Output:
            thp: Total fraction of light energy that goes through the system
    '''
    nrays = len(ray_bundles)
    propagated = [~np.isnan(ray_bundles[idx][-1, -1]) for idx in range(nrays)]

    return sum(propagated)/(1.0*nrays)

def vingetting(ray_bundles, ptdict):
    '''
        Estimate vingetting by computing point wise throughput

        Inputs:
            ray_bundles: List of rays. See raytracing.propagate_rays()
            ptdict: List of indices of rays for each point

        Outputs:
            ving: List of throughputs per point
    '''
    # Compute throughput
    nrays = len(ray_bundles)
    propagated = [~np.isnan(ray_bundles[idx][-1, -1]) for idx in range(nrays)]
    propagated = np.array(propagated)

    ving = np.zeros(len(ptdict))

    for idx in range(len(ptdict)):
        pt_indices = propagated[list(ptdict[idx])]
        ving[idx] = sum(pt_indices)/len(pt_indices)

    return ving
