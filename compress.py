import open3d as o3d

# Use your exact paths
input_path = "./models/Task_Board_TBv2023_m3b.stl"
output_path = "./models/task_board_perfect.stl"

print("Loading ASCII board...")
mesh = o3d.io.read_triangle_mesh(input_path)

# 1. Center the origin for MuJoCo
center = mesh.get_center()
mesh.translate(-center)

# 2. Convert from Millimeters to Meters
mesh.scale(1.0, center=(0, 0, 0))

# 3. Compute missing physics normals
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

# 4. Save as a Binary STL (write_ascii=False is the magic command!)
o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=False)
print("Success! File saved as Binary.")