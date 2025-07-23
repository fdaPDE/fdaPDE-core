import json
import matplotlib.pyplot as plt
import numpy as np

def plot_dcel(filename):
    # Carica il file JSON esportato dalla DCEL
    with open(filename, 'r') as file:
        data = json.load(file)

    # Estrai i nodi
    nodes = {node["id"]: np.array(node["coords"]) for node in data["nodes"]}

    # Crea figura senza assi e sfondo
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.axis('off')  # Nasconde assi e numeri

    # Disegna ogni arco
    for edge in data["edges"]:
        from_id = edge["from"]
        to_id = edge["to"]
        is_sub = edge.get("segment", False)

        p1 = nodes[from_id]
        p2 = nodes[to_id]

        color = "black" if is_sub else "green"
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.5)

    # Disegna triangoli (celle)
    halfedge_to_node = {edge["id"]: edge["from"] for edge in data["edges"]}
    for cell in data["cells"]:
        try:
            cell_nodes = [nodes[halfedge_to_node[edge]] for edge in cell["edges"]]
            x_values = [pt[0] for pt in cell_nodes] + [cell_nodes[0][0]]
            y_values = [pt[1] for pt in cell_nodes] + [cell_nodes[0][1]]
            ax.plot(x_values, y_values, 'g-', linewidth=0.5)
        except KeyError:
            print(f"Error in drawing cell with edges {cell['edges']}")

    # Salvataggio immagine pulita
    plt.savefig("Meshes/Delaunay/delaunay_plot.png", dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    print("Plot saved as 'Meshes/Delaunay/delaunay_plot.png'")

if __name__ == "__main__":
    plot_dcel("Meshes/Delaunay/delaunay_output.json")


