OptSys -- Optical systems simulator

# About
Simulate ray-tracing of optical system with first order approximation. This will
 be a handy tool to get an intial estimate for various components for your
 optical system. 

# Components
The following components can be used:

* **Lens**: Convex/concave lens with finite aperture. Uses lens maker's formula for
computing output ray

* **Mirror**: Flat mirror with finite aperture. 

* **Grating**: Diffraction grating with set number of groves and finite aperture. 
As of now, only the first order can be simulated

* **DMD**: Digital Micromirror device with zero pitch size and finite aperture.

# Extra functions
Apart from placement and viewing of optical elements and rays, you can also:

1. Compute the light throughput of the system

2. Simulate system for different wavelengths

# Usage
Broadly, simulation consists of two parts

* **Components**: A (python) list of various optical components with component
specific paramenter, position, aperture and angle w.r.t y-axis. An example:
```python
import raytracing as rt

components = []
components.append(rt.Lens(f=100,
			  aperture=25.4,
			  pos=[0,0],
			  angle=0))
```	

* **Rays**: A (python) list of 3-tuple of rays with x-coordinate, y-coordinate
and angle w.r.t x-axis. An example:
```python
import raytracing as rt

rays = []
rays.append([-10, 0, -np.pi/6])
```						 

Once you configure your system with components and rays, you can propagate the 
rays using the following command:
```python
ray_bundles = propagate_rays(components, rays)

```

In order to view the output, you create a canvas and draw the components and
rays. 
```python
import visualize as vis

# Color for the rays
colors = 'r'

# Create a new canvas
canvas = vis.Canvas([-200, 600], [-100, 100])

# Draw the components
canvas.draw_components(components)

# Draw the rays
canvas.draw_rays(ray_bundles, colors)

# Show the system
canvas.show()

# Save a copy
canvas.save('example.png')
```

See examples folder for more information.

# TODO
1. Implement convex/concave mirror
2. Implement vignetting computation 
3. Implement image plane computation

Authors:
*	Vishwanath Saragadam (PhD candidate, ECE, Carnegie Mellon University)

*	Aswin Sankaranarayanan (Asst. Prof., ECe, Carnegie Mellon University)
	
