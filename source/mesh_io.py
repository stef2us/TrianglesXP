import numpy as np

def parse_ortho_mesh(file_path):
    """
    Parse un fichier .mesh Ortho4XP.
    Retourne les tableaux numpy bruts : vertices, uvs, faces, tri_types, nbr_nodes, nbr_tris
    """
    with open(file_path, "r") as f:
        mesh_version = float(f.readline().strip().split()[-1])

        for _ in range(3): f.readline() # skip 3 lines

        nbr_nodes = int(f.readline())
        node_coords = np.zeros(5 * nbr_nodes, dtype=np.float32)

        # read positions
        for i in range(nbr_nodes):
            node_coords[5 * i : 5 * i + 3] = [float(x) for x in f.readline().split()[:3]]

        # altitutes scaling factor
        node_coords[2::5] *= 100000

        for _ in range(3): f.readline() # skip 3 lines

        # read normals (UVs)
        for i in range(nbr_nodes):
            node_coords[5 * i + 3 : 5 * i + 5] = [float(x) for x in f.readline().split()[:2]]

        for _ in range(2): f.readline() # skip 2 lines

        # read nbr of tris
        nbr_tris = int(f.readline())

        tri_idx = np.zeros((nbr_tris, 3), dtype=np.uint32)
        tri_types = np.zeros(nbr_tris, dtype=np.uint32)

        for i in range(nbr_tris):
            parts = f.readline().split()[:4]
            tri_idx[i] = (int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1)
            tri_types[i] = int(parts[3])

    # Préparation des données
    all_data = node_coords.reshape(-1, 5)
    vertices = all_data[:, :3]
    uvs = all_data[:, 3:5]
    faces = tri_idx

    return vertices, uvs, faces, tri_types, nbr_nodes, nbr_tris


def write_ortho_mesh(file_path, export_vertices, uvs, faces, tri_types):
    """
    Écrit les données numpy dans un fichier .mesh au format Ortho4XP.
    """
    nbr_vert = len(export_vertices)
    nbr_tri = len(faces)

    with open(file_path, "w") as f:
        f.write("MeshVersionFormatted 2\n")
        f.write("Dimension 3\n\n")

        # --- SOMMETS ---
        f.write("Vertices\n")
        f.write(f"{nbr_vert}\n")
        for i in range(nbr_vert):
            lon = export_vertices[i, 0]
            lat = export_vertices[i, 1]
            alt = export_vertices[i, 2] / 100000.0
            f.write(f"{lon:.15f} {lat:.15f} {alt:.15f} 0\n")
        f.write("\n")

        # --- NORMALES (Coordonnées UV) ---
        f.write("Normals\n")
        f.write(f"{nbr_vert}\n")
        for i in range(nbr_vert):
            u = uvs[i, 0]
            v = uvs[i, 1]
            f.write(f"{u:.2f} {v:.2f} 0\n")
        f.write("\n")

        # --- TRIANGLES ---
        f.write("Triangles\n")
        f.write(f"{nbr_tri}\n")
        for i in range(nbr_tri):
            n1 = faces[i, 0] + 1
            n2 = faces[i, 1] + 1
            n3 = faces[i, 2] + 1
            attr = tri_types[i]
            f.write(f"{n1} {n2} {n3} {attr}\n")