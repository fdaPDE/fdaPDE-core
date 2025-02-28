import matplotlib.pyplot as plt
import sys
import os

def read_node_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    num_points = int(header[0])
    points = {}
    for line in lines[1:num_points+1]:
        parts = line.strip().split()
        idx = int(parts[0])
        x, y = float(parts[1]), float(parts[2])
        points[idx] = (x, y)
    return points

def read_ele_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    num_triangles = int(header[0])
    triangles = []
    for line in lines[1:num_triangles+1]:
        parts = line.strip().split()
        v1, v2, v3 = int(parts[1]), int(parts[2]), int(parts[3])
        triangles.append((v1, v2, v3))
    return triangles

def plot_mesh(points, triangles, basename):
    fig, ax = plt.subplots()
    for tri in triangles:
        x = [points[i][0] for i in tri] + [points[tri[0]][0]]
        y = [points[i][1] for i in tri] + [points[tri[0]][1]]
        color = "black"
        ax.plot(x, y, color=color, linewidth=1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    out_path = f"Meshes/Test_triangle/{basename}_mesh.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    print(f"Mesh saved as {out_path}")

# === MAIN ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_triangle_mesh.py <basename>")
        sys.exit(1)

    basename = sys.argv[1]  # es: "star"
    node_file = f"Meshes/Test_triangle/{basename}.1.node"
    ele_file = f"Meshes/Test_triangle/{basename}.1.ele"

    points = read_node_file(node_file)
    triangles = read_ele_file(ele_file)
    plot_mesh(points, triangles, basename)
