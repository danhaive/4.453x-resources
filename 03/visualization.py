from trimesh.util import concatenate
from trimesh.creation import cylinder
import numpy as np
from .analysis import MAX_RADIUS


def generate_cylinder_meshes(vertices, edges, forces, radii):
    """
    Creates cylindrical meshes using the trimesh library
    Returns an array of mesh cylinders for each bar and its corresponding
    color attribute based on the internal axial force it experiences due to the applied load.
    """
    meshes = []
    color_attributes = []
    for e, r, f in zip(edges, radii, forces):
        cylinder_mesh = cylinder(radius=min(r, MAX_RADIUS), segment=vertices[e])
        meshes.append(cylinder_mesh)
        color_attributes += [f] * cylinder_mesh.vertices.shape[0]
    return concatenate(meshes), np.array(color_attributes).flatten()


        