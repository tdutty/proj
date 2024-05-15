import numpy as np

def render_mesh_cloud_to_obj(vertices, faces, output_obj_path):
    with open(output_obj_path, 'w') as obj_file:
        # Write vertices
        for vertex in vertices:
            obj_file.write("v {} {} {}\n".format(vertex[0], vertex[1], vertex[2]))

        # Write faces
        for face in faces:
            obj_file.write("f")
            for vertex_index in face:
                obj_file.write(" {}".format(vertex_index + 1))  # OBJ format uses 1-based indexing
            obj_file.write("\n")

    print("Mesh cloud rendered to OBJ successfully.")

if __name__ == "__main__":
    # Example vertices and faces (replace with your data)
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0]])
    faces = np.array([[0, 1, 2],
                      [0, 2, 3]])

    output_obj_path = "output_mesh.obj"  # Output OBJ file path

    render_mesh_cloud_to_obj(vertices, faces, output_obj_path)
