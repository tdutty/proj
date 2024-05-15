import numpy as np
from scipy.interpolate import griddata
import pymesh

def infer_missing_data(mesh_path, output_mesh_path):
    # Load mesh
    mesh = pymesh.load_mesh(mesh_path)
    vertices = mesh.vertices
    faces = mesh.faces

    # Extract x, y, z coordinates of vertices
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    # Generate grid of points covering the bounding box of the mesh
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    # Adjust bounding box to ensure interpolation covers entire mesh
    x_range = np.linspace(x_min, x_max, num=100)
    y_range = np.linspace(y_min, y_max, num=100)
    z_range = np.linspace(z_min, z_max, num=100)

    # Generate meshgrid for interpolation
    X, Y, Z = np.meshgrid(x_range, y_range, z_range)

    # Interpolate missing data points
    inferred_Z = griddata((x, y, z), z, (X, Y, Z), method='cubic')

    # Flatten meshgrid and inferred Z values
    inferred_points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten(), inferred_Z.flatten()))

    # Remove NaN values (outside of mesh bounds)
    inferred_points = inferred_points[~np.isnan(inferred_points).any(axis=1)]

    # Save the inferred mesh
    inferred_mesh = pymesh.form_mesh(inferred_points[:, :3], faces)
    pymesh.save_mesh(output_mesh_path, inferred_mesh)

    print("Missing data inference completed and mesh saved successfully.")

if __name__ == "__main__":
    mesh_path = "input_mesh.obj"  # Replace with the path to your mesh file
    output_mesh_path = "output_inferred_mesh.obj"  # Output mesh with inferred data

    infer_missing_data(mesh_path, output_mesh_path)
