import numpy as np
from geomdl import NURBS
from geomdl.visualization import VisMPL

# Step 1: Define the 180-degree NURBS Arc (in the XZ-plane)
arc_ctrlpts = np.array([
    [1.0, 0.0, 0.0],   # Rightmost point
    [0.0, 0.0, 1.0],   # Top middle
    [-1.0, 0.0, 0.0]   # Leftmost point
])

arc_weights = np.array([1.0, np.sqrt(2)/2, 1.0])  # Quadratic weights

# Define rotation angles for 360-degree revolution around Y-axis
angles = [0, 90, 180, 270, 360]  # Full revolution

# Convert to radians
angles_rad = np.radians(angles)

# Step 2: Generate control points for the sphere by rotating the arc
ctrlpts_2d = []
for theta in angles_rad:
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],  # Y remains unchanged
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    # Rotate each control point
    rotated_ctrlpts = np.dot(arc_ctrlpts, rotation_matrix.T)
    ctrlpts_2d.append(rotated_ctrlpts.tolist())

# Flatten the control points for geomdl
ctrlpts_cartesian = [list(pt) for row in ctrlpts_2d for pt in row]

# Step 3: Create the NURBS Sphere Surface
sphere = NURBS.Surface()
sphere.degree_u = 2  # Arc degree
sphere.degree_v = 2  # Revolution degree

# Define the number of control points in U and V directions
sphere.ctrlpts_size_u = 3  # Arc (3 control points)
sphere.ctrlpts_size_v = 5  # Full sphere revolution (5 control points)

# Assign control points to the surface
sphere.ctrlpts = ctrlpts_cartesian

# Define Corrected Knot Vectors
sphere.knotvector_u = [0, 0, 0, 1, 1, 1]  # Arc (degree 2, 3 control points)
sphere.knotvector_v = [0, 0, 0, 0.33, 0.66, 1, 1, 1]  # Revolution (degree 2, 5 control points)

# Step 4: Visualize the NURBS Sphere
sphere.vis = VisMPL.VisSurface()
sphere.render()