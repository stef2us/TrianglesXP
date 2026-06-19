import numpy as np
import triangle as tr
import logging
from collections import defaultdict, deque

def subdivide_mesh_selection(vertices, uvs, faces, tri_types, selected_faces_indices, tri_levels=None, max_subdiv_level=None):
    """
    Subdivise les triangles sélectionnés en évitant les déchirures aux frontières.
    Retourne les nouveaux tableaux et la nouvelle liste de sélection.
    Transmet le 'tri_levels' et gère un niveau maximum de subdivision.
    """
    if tri_levels is None:
        tri_levels = np.zeros(len(faces), dtype=np.uint8)

    # --- 1. Filtrage par niveau maximum ---
    selected_array = np.array(selected_faces_indices, dtype=int)
    if max_subdiv_level is not None:
        mask_subdiv = tri_levels[selected_array] < max_subdiv_level
        to_subdivide = selected_array[mask_subdiv].tolist()
        to_keep = selected_array[~mask_subdiv].tolist() # Ceux qui ont atteint la limite
    else:
        to_subdivide = selected_array.tolist()
        to_keep = []

    # On ne travaille plus que sur les triangles autorisés
    selected_faces = faces[to_subdivide]
    edge_counts = {}

    for face in selected_faces:
        edges = [tuple(sorted((face[0], face[1]))),
                 tuple(sorted((face[1], face[2]))),
                 tuple(sorted((face[2], face[0])))]
        for edge in edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    edges_to_cut = {edge for edge, count in edge_counts.items() if count == 2}

    # Si aucune arête à couper, on retourne la sélection d'origine INTACTE
    if not edges_to_cut:
        return vertices, uvs, faces, tri_types, selected_faces_indices, 0, tri_levels

    new_vertices_list = []
    new_uvs_list = []
    edge_to_new_vertex_idx = {}
    current_vertex_count = len(vertices)

    for edge in edges_to_cut:
        v1, v2 = edge
        new_vertices_list.append((vertices[v1] + vertices[v2]) / 2.0)
        new_uvs_list.append((uvs[v1] + uvs[v2]) / 2.0)
        edge_to_new_vertex_idx[edge] = current_vertex_count
        current_vertex_count += 1

    new_vertices = np.vstack((vertices, new_vertices_list))
    new_uvs = np.vstack((uvs, new_uvs_list))

    new_faces_list = []
    new_tri_types_list = []
    new_levels_list = []

    # --- 2. Isolation des triangles (y compris ceux qu'on épargne) ---
    mask_unselected = np.ones(len(faces), dtype=bool)
    mask_unselected[to_subdivide] = False

    new_faces_list.extend(faces[mask_unselected])
    new_tri_types_list.extend(tri_types[mask_unselected])
    new_levels_list.extend(tri_levels[mask_unselected])

    start_new_selection_idx = len(new_faces_list)

    # On calcule la nouvelle position des triangles qu'on a épargnés
    # pour pouvoir les réintégrer à la sélection à la fin !
    mapping = np.cumsum(mask_unselected) - 1
    kept_new_indices = mapping[to_keep].tolist()

    # --- 3. Subdivision des triangles autorisés ---
    for face_idx in to_subdivide:
        face = faces[face_idx]
        tri_type = tri_types[face_idx]
        current_lvl = tri_levels[face_idx]

        v0, v1, v2 = face
        e0 = tuple(sorted((v0, v1)))
        e1 = tuple(sorted((v1, v2)))
        e2 = tuple(sorted((v2, v0)))

        cut0 = e0 in edges_to_cut
        cut1 = e1 in edges_to_cut
        cut2 = e2 in edges_to_cut

        cuts = sum([cut0, cut1, cut2])

        if cuts == 0:
            new_faces_list.append([v0, v1, v2])
            new_tri_types_list.extend([tri_type])
            new_levels_list.append(current_lvl)

        elif cuts == 3:
            m0 = edge_to_new_vertex_idx[e0]
            m1 = edge_to_new_vertex_idx[e1]
            m2 = edge_to_new_vertex_idx[e2]
            new_faces_list.append([v0, m0, m2])
            new_faces_list.append([v1, m1, m0])
            new_faces_list.append([v2, m2, m1])
            new_faces_list.append([m0, m1, m2])
            new_tri_types_list.extend([tri_type] * 4)
            new_levels_list.extend([current_lvl + 1] * 4)

        elif cuts == 1:
            if cut0:
                m = edge_to_new_vertex_idx[e0]
                new_faces_list.append([v2, v0, m])
                new_faces_list.append([v2, m, v1])
            elif cut1:
                m = edge_to_new_vertex_idx[e1]
                new_faces_list.append([v0, v1, m])
                new_faces_list.append([v0, m, v2])
            elif cut2:
                m = edge_to_new_vertex_idx[e2]
                new_faces_list.append([v1, v2, m])
                new_faces_list.append([v1, m, v0])
            new_tri_types_list.extend([tri_type] * 2)
            new_levels_list.extend([current_lvl + 1] * 2)

        elif cuts == 2:
            if not cut0:
                m1 = edge_to_new_vertex_idx[e1]; m2 = edge_to_new_vertex_idx[e2]
                new_faces_list.append([v0, v1, m1])
                new_faces_list.append([v0, m1, m2])
                new_faces_list.append([v2, m2, m1])
            elif not cut1:
                m0 = edge_to_new_vertex_idx[e0]; m2 = edge_to_new_vertex_idx[e2]
                new_faces_list.append([v1, v2, m2])
                new_faces_list.append([v1, m2, m0])
                new_faces_list.append([v0, m0, m2])
            elif not cut2:
                m0 = edge_to_new_vertex_idx[e0]; m1 = edge_to_new_vertex_idx[e1]
                new_faces_list.append([v2, v0, m0])
                new_faces_list.append([v2, m0, m1])
                new_faces_list.append([v1, m1, m0])
            new_tri_types_list.extend([tri_type] * 3)
            new_levels_list.extend([current_lvl + 1] * 3)

    new_faces = np.array(new_faces_list, dtype=np.uint32)
    new_tri_types = np.array(new_tri_types_list, dtype=np.uint32)
    new_levels = np.array(new_levels_list, dtype=np.uint8)

    # --- 4. Fusion des sélections ---
    # On rassemble les triangles intacts (limite atteinte) et les nouveaux (fraîchement coupés)
    new_selection = kept_new_indices + list(range(start_new_selection_idx, len(new_faces)))

    return new_vertices, new_uvs, new_faces, new_tri_types, new_selection, len(edges_to_cut), new_levels

def apply_cotangent_smooth(vertices, faces, selected_faces_indices, iterations=5, factor=0.5, feather_radius=2):
    """
    Lissage par poids cotangents optimisé pour ne traiter que la sélection locale.
    Évite l'effet d'étirement dû aux triangles effilés ("slivers") générés par la subdivision.

    :param vertices: Tableau numpy (N_global, 3) des sommets.
    :param faces: Tableau numpy (F_global, 3) des faces.
    :param selected_faces_indices: Liste ou tableau des index des faces sélectionnées.
    :param iterations: Nombre de passes de lissage.
    :param factor: Force du lissage (0.0 à 1.0).
    :param feather_radius: Rayon (en nombre de sommets/arêtes) de la zone de transition pour le lissage des bords.
    """
    if len(selected_faces_indices) == 0:
        return vertices

    # === ÉTAPE 1 à 4 : Isolation, identification et Mapping ===
    local_faces = faces[selected_faces_indices]
    global_vert_indices, local_faces_mapped_1d = np.unique(local_faces, return_inverse=True)
    local_faces_mapped = local_faces_mapped_1d.reshape(local_faces.shape)
    N = len(global_vert_indices)

    # === ÉTAPE 5 : Calcul du Feathering (Zone de transition par BFS) ===
    # Extraction de toutes les arêtes locales
    edges = np.vstack((
        local_faces_mapped[:, [0, 1]],
        local_faces_mapped[:, [1, 2]],
        local_faces_mapped[:, [2, 0]]
    ))
    edges = np.sort(edges, axis=1)

    # Les arêtes qui n'apparaissent qu'une fois sont les frontières de la sélection
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    boundary_vertices = np.unique(boundary_edges)

    # Construction d'un graphe d'adjacence local pour le parcours (BFS)
    adj = [[] for _ in range(N)]
    for v1, v2 in unique_edges:
        adj[v1].append(v2)
        adj[v2].append(v1)

    # Parcours en largeur (BFS) pour calculer la "distance topologique" au bord
    distances = np.full(N, -1, dtype=np.int32)
    queue = deque(boundary_vertices)
    for bv in boundary_vertices:
        distances[bv] = 0

    while queue:
        curr = queue.popleft()
        d = distances[curr]
        for neighbor in adj[curr]:
            if distances[neighbor] == -1:
                distances[neighbor] = d + 1
                queue.append(neighbor)

    # Calcul des multiplicateurs (alpha) pour adoucir les bords (cosinus)
    alphas = np.zeros(N, dtype=np.float32)
    for i in range(N):
        d = distances[i]
        if d <= 0:
            alphas[i] = 0.0 # Strictement sur le bord : on ne bouge pas
        elif d >= feather_radius:
            alphas[i] = factor # Au centre de la sélection : force maximale
        else:
            # Transition douce (interpolation cosinus)
            normalized = d / feather_radius
            alphas[i] = factor * (1.0 - np.cos(normalized * np.pi)) / 2.0

    # === ÉTAPE 6 : Exécution de la logique de lissage Cotangent ===
    # On travaille sur une copie locale des sommets
    local_vertices = vertices[global_vert_indices].copy()

    for _ in range(iterations):
        # Extraire les coordonnées 3D des 3 sommets pour chaque face locale
        v0 = local_vertices[local_faces_mapped[:, 0]]
        v1 = local_vertices[local_faces_mapped[:, 1]]
        v2 = local_vertices[local_faces_mapped[:, 2]]

        # Vecteurs directeurs des arêtes (opposées aux sommets v0, v1, v2)
        e0 = v2 - v1
        e1 = v0 - v2
        e2 = v1 - v0

        # Calcul des aires des triangles (norme du produit vectoriel)
        cross_norm = np.linalg.norm(np.cross(e0, -e1), axis=1)
        cross_norm[cross_norm < 1e-8] = 1e-8 # Éviter la division par zéro

        # Calcul des cotangentes pour chaque angle du triangle (produit scalaire / norme du produit vectoriel)
        cot0 = np.sum(-e2 * e1, axis=1) / cross_norm
        cot1 = np.sum(-e0 * e2, axis=1) / cross_norm
        cot2 = np.sum(-e1 * e0, axis=1) / cross_norm

        # Sécurité : limiter l'amplitude des cotangentes pour les triangles quasi-dégénérés
        cot0 = np.clip(cot0, -10.0, 10.0)
        cot1 = np.clip(cot1, -10.0, 10.0)
        cot2 = np.clip(cot2, -10.0, 10.0)

        # Tableaux d'accumulation
        sum_weights = np.zeros(N, dtype=np.float32)
        sum_z = np.zeros(N, dtype=np.float32)

        # Accumuler les poids et les altitudes pondérées pour chaque sommet de la face
        # Pour v0
        np.add.at(sum_weights, local_faces_mapped[:, 0], cot1 + cot2)
        np.add.at(sum_z, local_faces_mapped[:, 0], cot1 * local_vertices[local_faces_mapped[:, 2], 2] + cot2 * local_vertices[local_faces_mapped[:, 1], 2])

        # Pour v1
        np.add.at(sum_weights, local_faces_mapped[:, 1], cot2 + cot0)
        np.add.at(sum_z, local_faces_mapped[:, 1], cot2 * local_vertices[local_faces_mapped[:, 0], 2] + cot0 * local_vertices[local_faces_mapped[:, 2], 2])

        # Pour v2
        np.add.at(sum_weights, local_faces_mapped[:, 2], cot0 + cot1)
        np.add.at(sum_z, local_faces_mapped[:, 2], cot0 * local_vertices[local_faces_mapped[:, 1], 2] + cot1 * local_vertices[local_faces_mapped[:, 0], 2])

        # Appliquer les nouvelles altitudes avec le Feathering (alpha)
        valid = sum_weights > 1e-8
        new_z = np.copy(local_vertices[:, 2])
        new_z[valid] = sum_z[valid] / sum_weights[valid]

        # Seul l'axe Z est modifié pour préserver les coordonnées X,Y de la grille Ortho4XP
        local_vertices[:, 2] = (1.0 - alphas) * local_vertices[:, 2] + alphas * new_z

    # === ÉTAPE 7 : Réintégration dans le tableau global ===
    vertices[global_vert_indices, 2] = local_vertices[:, 2]

    return vertices

def find_triangles_in_rect(screen_verts_4d, faces, x_min, x_max, y_min, y_max):
    """
    Algorithme SAT pour trouver les triangles à l'intérieur d'un rectangle 2D à l'écran.
    """
    sw = screen_verts_4d[:, 3]
    valid_w = sw > 1e-6
    valid_faces_mask = valid_w[faces[:, 0]] & valid_w[faces[:, 1]] & valid_w[faces[:, 2]]
    safe_sw = np.where(valid_w, sw, 1.0)
    sx = screen_verts_4d[:, 0] / safe_sw
    sy = screen_verts_4d[:, 1] / safe_sw

    faces_x = sx[faces]
    faces_y = sy[faces]
    tri_x_min, tri_x_max = np.min(faces_x, axis=1), np.max(faces_x, axis=1)
    tri_y_min, tri_y_max = np.min(faces_y, axis=1), np.max(faces_y, axis=1)

    aabb_mask = valid_faces_mask & (tri_x_max >= x_min) & (tri_x_min <= x_max) & \
                (tri_y_max >= y_min) & (tri_y_min <= y_max)
    candidate_indices = np.where(aabb_mask)[0]

    if len(candidate_indices) == 0:
        return []

    candidates = faces[candidate_indices]
    V = np.column_stack((sx, sy))
    v0, v1, v2 = V[candidates[:, 0]], V[candidates[:, 1]], V[candidates[:, 2]]
    edges = [v1 - v0, v2 - v1, v0 - v2]
    rect_corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    collision_mask = np.ones(len(candidate_indices), dtype=bool)

    for edge in edges:
        normals = np.column_stack((-edge[:, 1], edge[:, 0]))
        p_v0 = np.einsum('ij,ij->i', v0, normals)
        p_v1 = np.einsum('ij,ij->i', v1, normals)
        p_v2 = np.einsum('ij,ij->i', v2, normals)
        tri_min = np.minimum.reduce([p_v0, p_v1, p_v2])
        tri_max = np.maximum.reduce([p_v0, p_v1, p_v2])

        p_rect = np.dot(rect_corners, normals.T)
        rect_min, rect_max = np.min(p_rect, axis=0), np.max(p_rect, axis=0)
        overlap = (tri_max >= rect_min) & (tri_min <= rect_max)
        collision_mask &= overlap

    return candidate_indices[collision_mask].tolist()

def split_closest_edge(vertices, uvs, faces, tri_types, tri_levels, pivot_x, pivot_y):
    """
    Trouve l'arête la plus proche d'un point 2D, crée un sommet en son milieu et adapte la topologie.
    Retourne les nouveaux tableaux (dont tri_levels) et l'index du sommet créé.
    """
    # --- OPTIMISATION EXTRÊME : Restreindre la recherche aux faces proches ---
    dist_sq_to_pivot = (vertices[:, 0] - pivot_x)**2 + (vertices[:, 1] - pivot_y)**2
    # On prend les 10 sommets les plus proches (instantané avec argpartition)
    closest_verts = np.argpartition(dist_sq_to_pivot, 10)[:10]

    # On isole uniquement les triangles rattachés à ces 10 sommets
    mask = np.isin(faces[:, 0], closest_verts) | np.isin(faces[:, 1], closest_verts) | np.isin(faces[:, 2], closest_verts)
    candidate_faces_idx = np.where(mask)[0]

    if len(candidate_faces_idx) == 0:
        candidate_faces_idx = np.arange(len(faces)) # Fallback de sécurité

    edge_to_faces = {}
    edges = []

    # On ne boucle QUE sur les quelques dizaines de triangles candidats !
    for f_idx in candidate_faces_idx:
        face = faces[f_idx]
        for j in range(3):
            e = tuple(sorted((face[j], face[(j+1)%3])))
            if e not in edge_to_faces:
                edge_to_faces[e] = []
                edges.append(e)
            edge_to_faces[e].append(f_idx)

    edges = np.array(edges)
    p = np.array([pivot_x, pivot_y])
    v1_2d = vertices[edges[:, 0], :2]
    v2_2d = vertices[edges[:, 1], :2]

    ab = v2_2d - v1_2d
    ap = p - v1_2d
    ab_ab = np.einsum('ij,ij->i', ab, ab)
    ab_ab[ab_ab == 0] = 1e-8
    ap_ab = np.einsum('ij,ij->i', ap, ab)

    t = np.clip(ap_ab / ab_ab, 0.0, 1.0)
    proj = v1_2d + t[:, np.newaxis] * ab
    dist_sq = np.sum((p - proj)**2, axis=1)

    closest_edge_idx = np.argmin(dist_sq)
    closest_edge = tuple(edges[closest_edge_idx])
    faces_to_modify = edge_to_faces[closest_edge]

    v1, v2 = closest_edge
    new_v = (vertices[v1] + vertices[v2]) / 2.0
    new_uv = (uvs[v1] + uvs[v2]) / 2.0

    new_idx = len(vertices)
    new_vertices = np.vstack((vertices, new_v))
    new_uvs = np.vstack((uvs, new_uv))

    new_faces = list(faces)
    new_tri_types = list(tri_types)
    new_tri_levels = list(tri_levels)

    # --- SÉCURITÉ TOPOLOGIQUE : Mise à jour de tri_levels incluse ---
    # On remplace les triangles en partant de la fin pour ne pas décaler les index
    for f_idx in sorted(faces_to_modify, reverse=True):
        f = new_faces.pop(f_idx)
        t_type = new_tri_types.pop(f_idx)
        t_lvl = new_tri_levels.pop(f_idx)

        if f[0] in (v1, v2) and f[1] in (v1, v2): A, B, C = f[0], f[1], f[2]
        elif f[1] in (v1, v2) and f[2] in (v1, v2): A, B, C = f[1], f[2], f[0]
        else: A, B, C = f[2], f[0], f[1]

        new_faces.append([A, new_idx, C])
        new_faces.append([new_idx, B, C])
        new_tri_types.extend([t_type, t_type])
        new_tri_levels.extend([t_lvl, t_lvl]) # Les nouveaux triangles héritent du niveau de l'ancien

    return new_vertices, new_uvs, np.array(new_faces), np.array(new_tri_types), np.array(new_tri_levels), new_idx

def get_airport_centers(vertices, faces, tri_types, runway_type=16):
    """
    Identifie les aéroports en regroupant les triangles de type "piste" adjacents.
    Retourne une liste de coordonnées 3D (centres surélevés).
    """
    runway_indices = np.where(tri_types == runway_type)[0]

    if len(runway_indices) == 0:
        return []

    vertex_to_faces = defaultdict(list)
    for idx in runway_indices:
        for v in faces[idx]:
            vertex_to_faces[v].append(idx)

    visited = set()
    airport_centers = []

    for idx in runway_indices:
        if idx not in visited:
            cluster = []
            queue = [idx]
            visited.add(idx)

            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for v in faces[curr]:
                    for neighbor in vertex_to_faces[v]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

            cluster_faces = faces[cluster]
            unique_verts = np.unique(cluster_faces.flatten())
            center = np.mean(vertices[unique_verts, :3], axis=0)
            center[2] += 150.0  # Surélever le marqueur
            airport_centers.append(center)

    return airport_centers

def get_faces_in_cylinder(vertices, faces, pivot_x, pivot_y, radius, base_alt):
    """
    Trouve les triangles dont au moins un sommet est dans le rayon et au-dessus de l'altitude base.
    Retourne la liste des index.
    """
    faces_verts = vertices[faces]
    dist_2d_sq = (faces_verts[:, :, 0] - pivot_x)**2 + (faces_verts[:, :, 1] - pivot_y)**2
    mask_radius = np.any(dist_2d_sq <= radius**2, axis=1)
    mask_alt = np.max(faces_verts[:, :, 2], axis=1) >= base_alt
    return np.where(mask_radius & mask_alt)[0].tolist()

def get_selection_boundary_2d(vertices, faces, selected_faces_indices):
    """
    Extrait les segments 2D (X, Y) formant le périmètre exact de la sélection.
    """
    sel_faces = faces[selected_faces_indices]
    edge_counts = {}

    # Comptage des arêtes pour trouver la frontière (apparue 1 seule fois)
    for face in sel_faces:
        edges = [tuple(sorted((face[0], face[1]))),
                 tuple(sorted((face[1], face[2]))),
                 tuple(sorted((face[2], face[0])))]
        for e in edges:
            edge_counts[e] = edge_counts.get(e, 0) + 1

    boundary_edges = [e for e, count in edge_counts.items() if count == 1]

    # Extraction des coordonnées 2D
    segments_2d = []
    for e in boundary_edges:
        p1 = vertices[e[0], :2]
        p2 = vertices[e[1], :2]
        segments_2d.append((p1, p2))

    return segments_2d

def points_to_segments_dist(points_2d, segments_2d):
    """
    Calcule la distance minimale de chaque point (N, 2) à un ensemble de segments 2D.
    Hautement optimisé avec NumPy pour traiter des milliers de points instantanément.
    """
    if not segments_2d:
        return np.full(len(points_2d), np.inf)

    min_dist_sq = np.full(len(points_2d), np.inf)

    for a, b in segments_2d:
        ab = b - a
        ab_ab = np.dot(ab, ab)

        if ab_ab == 0:
            d_sq = np.sum((points_2d - a)**2, axis=1)
        else:
            ap = points_2d - a
            t = np.clip(np.dot(ap, ab) / ab_ab, 0.0, 1.0)
            proj = a + np.outer(t, ab)
            d_sq = np.sum((points_2d - proj)**2, axis=1)

        min_dist_sq = np.minimum(min_dist_sq, d_sq)

    return np.sqrt(min_dist_sq)

def points_in_polygons_concave(points_2d, list_of_polygons):
    """
    Vérifie si des points 2D sont à l'intérieur d'une liste de polygones.
    Version Robuste : Utilise un pré-filtrage strict par Bounding Box (Boîte Englobante)
    pour empêcher les bugs de virgule flottante sur les lignes quasi-horizontales.
    """
    if len(points_2d) == 0 or not list_of_polygons:
        return np.zeros(len(points_2d), dtype=bool)

    global_inside = np.zeros(len(points_2d), dtype=bool)
    x = points_2d[:, 0]
    y = points_2d[:, 1]

    for poly in list_of_polygons:
        poly_pts = np.array(poly)
        if len(poly_pts) < 3: continue

        # --- CORRECTION : Filtrage strict par Boîte Englobante ---
        min_x, max_x = np.min(poly_pts[:, 0]), np.max(poly_pts[:, 0])
        min_y, max_y = np.min(poly_pts[:, 1]), np.max(poly_pts[:, 1])

        # Marge de sécurité de 2 mètres
        bb_mask = (x >= min_x - 2.0) & (x <= max_x + 2.0) & (y >= min_y - 2.0) & (y <= max_y + 2.0)

        # Si aucun point du mesh n'est physiquement proche, on ignore le calcul lourd
        if not np.any(bb_mask):
            continue

        poly_inside = np.zeros(len(points_2d), dtype=bool)
        p1x, p1y = poly_pts[-1]

        for p2x, p2y in poly_pts:
            # On applique le calcul du rayon UNIQUEMENT sur les points présélectionnés (bb_mask)
            edge_mask = bb_mask & (y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x))

            if p1y != p2y:
                # Si (p2y - p1y) est minuscule, xinters explose, mais edge_mask nous protège
                xinters = (y[edge_mask] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                intersect_mask = x[edge_mask] <= xinters

                # Inversion de l'état (dedans/dehors)
                idx_to_toggle = np.where(edge_mask)[0][intersect_mask]
                poly_inside[idx_to_toggle] = ~poly_inside[idx_to_toggle]

            p1x, p1y = p2x, p2y

        global_inside |= poly_inside

    return global_inside

def get_cosine_blend_factor(distance, trans_width):
    """
    Retourne un facteur de 1.0 (Altitude Cible) à 0.0 (Altitude Naturelle)
    selon la distance dans le talus.
    """
    if trans_width <= 0:
        return np.where(distance <= 0, 1.0, 0.0)

    ratio = np.clip(distance / trans_width, 0.0, 1.0)
    return (np.cos(ratio * np.pi) + 1.0) / 2.0

def calculate_earthwork_blend(current_z, target_z, distance, trans_width):
    """Noyau mathématique unifié de calcul du talus (Flatten)."""
    factor = get_cosine_blend_factor(distance, trans_width)
    return current_z * (1.0 - factor) + target_z * factor

def split_polygon_by_axis(polygon, axis_idx, line_val, epsilon=1e-4):
    """
    Coupe un polygone convexe 2D par une ligne alignée sur un axe (X=0 ou Y=1).
    Algorithme de Sutherland-Hodgman avec Snapping (Évite les slivers).
    Retourne une liste contenant 1 ou 2 sous-polygones.
    """
    poly_less = []
    poly_greater = []

    if len(polygon) < 3:
        return [polygon]

    for i in range(len(polygon)):
        p_curr = polygon[i]
        p_next = polygon[(i + 1) % len(polygon)]

        val_curr = p_curr[axis_idx]
        val_next = p_next[axis_idx]

        # 1. Tri du point courant avec tolérance (Snapping)
        if val_curr < line_val - epsilon:
            poly_less.append(p_curr)
        elif val_curr > line_val + epsilon:
            poly_greater.append(p_curr)
        else:
            # Le point est quasiment sur la ligne, on le force mathématiquement dessus
            p_snap = p_curr.copy()
            p_snap[axis_idx] = line_val
            poly_less.append(p_snap)
            poly_greater.append(p_snap)

        # 2. Vérification de l'intersection sur l'arête (Franchissement VRAI)
        if (val_curr < line_val - epsilon and val_next > line_val + epsilon) or \
           (val_next < line_val - epsilon and val_curr > line_val + epsilon):

            # Interpolation linéaire exacte
            t = (line_val - val_curr) / (val_next - val_curr)
            intersect = p_curr + t * (p_next - p_curr)
            intersect[axis_idx] = line_val # On force la précision

            poly_less.append(intersect)
            poly_greater.append(intersect)

    res = []

    # NOUVEAU : Nettoyage des doublons consécutifs avant de valider le polygone
    def clean_poly(poly_pts):
        if not poly_pts: return []
        clean = [poly_pts[0]]
        for pt in poly_pts[1:]:
            if np.linalg.norm(pt - clean[-1]) > epsilon:
                clean.append(pt)
        # Vérification entre le premier et le dernier point (fermeture)
        if len(clean) > 1 and np.linalg.norm(clean[0] - clean[-1]) <= epsilon:
            clean.pop()
        return clean

    clean_less = clean_poly(poly_less)
    clean_greater = clean_poly(poly_greater)

    # On ne conserve que les polygones valides (au moins 3 points)
    if len(clean_less) >= 3: res.append(np.array(clean_less))
    if len(clean_greater) >= 3: res.append(np.array(clean_greater))

    return res

def generate_sliced_runway_mesh(p1_2d, p2_2d, width, spline_func, x_lines, y_lines, step_size=20.0):
    """
    Génère les sommets et triangles d'un ruban de piste parfaitement découpé
    par le quadrillage ZL des textures.
    """
    # 1. Vecteurs directeurs 2D
    vec = p2_2d - p1_2d
    length = np.linalg.norm(vec)
    if length == 0: return np.array([]), np.array([])
    dir_vec = vec / length
    normal_vec = np.array([-dir_vec[1], dir_vec[0]])
    hw = width / 2.0

    # 2. Découpage longitudinal (Résolution Z du profil de la piste)
    num_steps = max(1, int(np.ceil(length / step_size)))
    t_vals = np.linspace(0, 1.0, num_steps + 1)

    polygons_2d = []

    # Génération des quads de base
    for i in range(num_steps):
        c_start = p1_2d + t_vals[i] * vec
        c_end = p1_2d + t_vals[i+1] * vec

        # Points projetés latéralement (Ordre de tracé anti-horaire : Bas-Gauche, Bas-Droite, Haut-Droite, Haut-Gauche)
        p_br = c_start - normal_vec * hw
        p_tr = c_end - normal_vec * hw
        p_tl = c_end + normal_vec * hw
        p_bl = c_start + normal_vec * hw

        polygons_2d.append(np.array([p_br, p_tr, p_tl, p_bl]))

    # 3. Tranchage par le quadrillage X (Frontières Nord-Sud)
    sliced_by_x = []
    for poly in polygons_2d:
        current_pieces = [poly]
        for lx in x_lines:
            new_pieces = []
            for piece in current_pieces:
                # Bounding box rapide pour éviter de calculer pour rien
                if np.min(piece[:, 0]) < lx < np.max(piece[:, 0]):
                    new_pieces.extend(split_polygon_by_axis(piece, 0, lx))
                else:
                    new_pieces.append(piece)
            current_pieces = new_pieces
        sliced_by_x.extend(current_pieces)

    # 4. Tranchage par le quadrillage Y (Frontières Est-Ouest)
    final_polygons = []
    for poly in sliced_by_x:
        current_pieces = [poly]
        for ly in y_lines:
            new_pieces = []
            for piece in current_pieces:
                if np.min(piece[:, 1]) < ly < np.max(piece[:, 1]):
                    new_pieces.extend(split_polygon_by_axis(piece, 1, ly))
                else:
                    new_pieces.append(piece)
            current_pieces = new_pieces
        final_polygons.extend(current_pieces)

    # 5. Injection de l'altitude (Z) et Triangulation
    vertices = []
    faces = []
    v_idx = 0

    for poly in final_polygons:
        poly_3d = []
        for pt in poly:
            # Pour assigner le bon Z, on projette le point sur l'axe central pour trouver son paramètre 't' (0 à 1)
            ap = pt - p1_2d
            t_pt = np.dot(ap, dir_vec) / length
            t_pt = np.clip(t_pt, 0.0, 1.0)
            z_val = spline_func(t_pt)
            poly_3d.append([pt[0], pt[1], z_val])

        # Triangulation en éventail (Triangle Fan) à partir du sommet 0
        start_idx = v_idx
        vertices.extend(poly_3d)
        num_pts = len(poly_3d)

        for i in range(1, num_pts - 1):
            faces.append([start_idx, start_idx + i, start_idx + i + 1])

        v_idx += num_pts

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.uint32)

def get_ordered_boundary_loop(faces, vertices=None, reference_2d=None):
    """
    Trouve les arêtes uniques d'un maillage et les chaîne pour former une boucle continue.
    Si vertices et reference_2d sont fournis, retourne la boucle physiquement la plus proche
    du point de référence (Évite les fausses bordures issues de patchs éloignés).
    """
    edge_counts = {}
    for f in faces:
        edges = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
        for e in edges:
            sorted_e = tuple(sorted(e))
            edge_counts[sorted_e] = edge_counts.get(sorted_e, 0) + 1

    adjacency = {}
    for f in faces:
        edges = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
        for e in edges:
            if edge_counts[tuple(sorted(e))] == 1:
                adjacency[e[0]] = e[1]

    if not adjacency:
        return []

    loops = []
    unvisited = set(adjacency.keys())

    while unvisited:
        start_node = next(iter(unvisited))
        current_loop = [start_node]
        current_node = adjacency[start_node]
        unvisited.remove(start_node)

        max_iters = len(adjacency)
        iters = 0

        while current_node != start_node and current_node is not None and iters < max_iters:
            current_loop.append(current_node)
            if current_node in unvisited:
                unvisited.remove(current_node)
            current_node = adjacency.get(current_node)
            iters += 1

        loops.append(current_loop)

    if not loops:
        return []

    # --- CORRECTION : Filtrage par Proximité Spatiale ---
    if vertices is not None and reference_2d is not None:
        best_loop = None
        min_dist = float('inf')
        for loop in loops:
            # On calcule le barycentre 2D de la boucle
            loop_pts = vertices[loop, :2]
            centroid = np.mean(loop_pts, axis=0)
            dist = np.linalg.norm(centroid - reference_2d[:2])

            if dist < min_dist:
                min_dist = dist
                best_loop = loop
        return best_loop

    # Fallback historique : on prend la boucle avec le plus de sommets
    return max(loops, key=len)

def analyze_mesh_anomalies(vertices, faces, center_2d, radius=100.0):
    """
    Analyse la topologie locale autour d'un point pour détecter les anomalies
    (Slivers, triangles géants, géométrie dégénérée).
    """
    # 1. Isoler les sommets dans le rayon
    dists = np.linalg.norm(vertices[:, :2] - center_2d, axis=1)
    verts_in_radius = np.where(dists <= radius)[0]

    if len(verts_in_radius) == 0:
        return f"DIAGNOSTIC: Aucun sommet trouvé dans un rayon de {radius}m."

    # 2. Trouver toutes les faces qui touchent au moins un de ces sommets
    mask = np.isin(faces, verts_in_radius).any(axis=1)
    local_faces = faces[mask]

    slivers = 0
    giants = 0
    degenerates = 0

    anomalies_details = []

    for idx, f in enumerate(local_faces):
        pts = vertices[f]

        # Calcul des longueurs des 3 arêtes
        e1 = np.linalg.norm(pts[1] - pts[0])
        e2 = np.linalg.norm(pts[2] - pts[1])
        e3 = np.linalg.norm(pts[0] - pts[2])
        lengths = sorted([e1, e2, e3])

        if lengths[0] < 0.01: # Arête de moins de 1 cm
            degenerates += 1
            continue

        aspect_ratio = lengths[2] / lengths[0]
        is_anomalous = False

        if lengths[2] > 2000.0:  # Arête de plus de 2 km
            giants += 1
            is_anomalous = True

        if aspect_ratio > 100.0: # Triangle 100 fois plus long que large
            slivers += 1
            is_anomalous = True

        if is_anomalous and len(anomalies_details) < 10:
            anomalies_details.append(
                f"  -> Face avec arêtes (m): [{lengths[0]:.2f}, {lengths[1]:.2f}, {lengths[2]:.2f}] | Ratio: {aspect_ratio:.1f}"
            )

    # 3. Formatage du rapport
    report = [
        f"--- DIAGNOSTIC MESH (Rayon {radius}m) ---",
        f"Analyse de {len(local_faces)} triangles impactant la zone :",
        f"- Triangles 'Slivers' (Ratio > 100) : {slivers}",
        f"- Triangles Géants (Bord > 2km) : {giants}",
        f"- Triangles Dégénérés (< 1cm) : {degenerates}"
    ]

    if anomalies_details:
        report.append("Exemples d'anomalies détectées :")
        report.extend(anomalies_details)

    report.append("--------------------------------------")

    return "\n".join(report)

def heal_local_topology(vertices, uvs, faces, tri_types, tri_levels, center_2d, radius=500.0, merge_threshold=0.5):
    """
    Répare le maillage en fusionnant les sommets trop proches (Slivers/Dégénérés)
    et en supprimant les triangles effondrés.
    """
    # 1. Isoler les sommets dans le rayon d'action
    dists = np.linalg.norm(vertices[:, :2] - center_2d, axis=1)
    in_radius_idx = np.where(dists <= radius)[0]

    if len(in_radius_idx) < 2:
        return vertices, uvs, faces, tri_types, tri_levels, 0

    # 2. Calcul des distances entre tous les sommets locaux
    local_v2d = vertices[in_radius_idx, :2]
    diff = local_v2d[:, np.newaxis, :] - local_v2d[np.newaxis, :, :]
    sq_dist_matrix = np.sum(diff**2, axis=-1)

    # --- CORRECTION : On enlève (sq_dist_matrix > 0) pour attraper les clones parfaits ---
    threshold_sq = merge_threshold**2
    close_pairs = np.argwhere(sq_dist_matrix < threshold_sq)

    if len(close_pairs) == 0:
        return vertices, uvs, faces, tri_types, tri_levels, 0

    # 3. Création de la carte de fusion (Union-Find)
    v_map = np.arange(len(vertices))
    merged_count = 0

    for i, j in close_pairs:
        if i >= j: continue # Ignore l'auto-comparaison (i==j) et les doublons (i>j)

        global_i = in_radius_idx[i]
        global_j = in_radius_idx[j]

        # Recherche de la racine pour éviter les chaînes de fusion
        root_i = global_i
        while v_map[root_i] != root_i: root_i = v_map[root_i]
        root_j = global_j
        while v_map[root_j] != root_j: root_j = v_map[root_j]

        if root_i != root_j:
            v_map[root_j] = root_i # Le sommet J devient le sommet I
            merged_count += 1

    if merged_count == 0:
        return vertices, uvs, faces, tri_types, tri_levels, 0

    # 4. Application de la fusion sur les triangles
    new_faces = v_map[faces]

    # 5. Destruction des triangles effondrés (ceux qui ont maintenant 2 ou 3 sommets identiques)
    valid_faces_mask = (new_faces[:, 0] != new_faces[:, 1]) & \
                       (new_faces[:, 1] != new_faces[:, 2]) & \
                       (new_faces[:, 0] != new_faces[:, 2])

    final_faces = new_faces[valid_faces_mask]
    final_types = tri_types[valid_faces_mask]

    if tri_levels is None or len(tri_levels) != len(faces):
        final_levels = np.zeros(len(final_faces), dtype=np.uint8)
    else:
        final_levels = tri_levels[valid_faces_mask]

    faces_removed = len(faces) - len(final_faces)

    return vertices, uvs, final_faces, final_types, final_levels, faces_removed

def ccw_vectorized(A, B, C):
    """Détermine si 3 points forment un virage anti-horaire (True) ou horaire (False)."""
    return (C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0]) > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])

def segments_intersect_vectorized(A, B, C, D):
    """Vérifie si une liste de segments AB croise un segment fixe CD."""
    return (ccw_vectorized(A, C, D) != ccw_vectorized(B, C, D)) & (ccw_vectorized(A, B, C) != ccw_vectorized(A, B, D))

def faces_intersecting_polygon(vertices_2d, faces, polygon):
    """
    Retourne un masque booléen identifiant les triangles dont au moins
    une arête traverse physiquement les bords du polygone donné.
    VERSION OPTIMISÉE : Calcule automatiquement sa propre Bounding Box interne !
    """
    intersect_mask = np.zeros(len(faces), dtype=bool)
    poly_pts = np.array(polygon)
    num_edges = len(poly_pts)

    if num_edges < 3:
        return intersect_mask

    # 1. Calcul de la Bounding Box du polygone
    min_x, max_x = np.min(poly_pts[:, 0]), np.max(poly_pts[:, 0])
    min_y, max_y = np.min(poly_pts[:, 1]), np.max(poly_pts[:, 1])

    # 2. Filtrage spatial (Boîte englobante + marge)
    margin = 5.0
    v2d = vertices_2d[:, :2] # Assure qu'on est en 2D

    v_in_bb = (v2d[:, 0] >= min_x - margin) & (v2d[:, 0] <= max_x + margin) & \
              (v2d[:, 1] >= min_y - margin) & (v2d[:, 1] <= max_y + margin)

    # 3. Indexation booléenne fulgurante pour cibler les triangles locaux
    f_in_bb_mask = v_in_bb[faces[:, 0]] | v_in_bb[faces[:, 1]] | v_in_bb[faces[:, 2]]
    cand_f_idx = np.where(f_in_bb_mask)[0]

    # Si aucun triangle n'est dans la zone, on renvoie False pour tout le monde instantanément
    if len(cand_f_idx) == 0:
        return intersect_mask

    # 4. Extraction de la mini-carte locale
    cand_faces = faces[cand_f_idx]

    # Allocation de la mémoire UNIQUEMENT pour les quelques triangles concernés !
    t_A = v2d[cand_faces[:, 0]]
    t_B = v2d[cand_faces[:, 1]]
    t_C = v2d[cand_faces[:, 2]]

    local_intersect_mask = np.zeros(len(cand_faces), dtype=bool)

    # 5. Calcul géométrique vectoriel
    for i in range(num_edges):
        P1 = poly_pts[i]
        P2 = poly_pts[(i + 1) % num_edges]

        P1_arr = np.tile(P1, (len(cand_faces), 1))
        P2_arr = np.tile(P2, (len(cand_faces), 1))

        int1 = segments_intersect_vectorized(t_A, t_B, P1_arr, P2_arr)
        int2 = segments_intersect_vectorized(t_B, t_C, P1_arr, P2_arr)
        int3 = segments_intersect_vectorized(t_C, t_A, P1_arr, P2_arr)

        local_intersect_mask |= (int1 | int2 | int3)

    # 6. Réinjection des résultats locaux dans le masque global
    intersect_mask[cand_f_idx] = local_intersect_mask

    return intersect_mask

def get_polygon_axis_intersections(polygon_3d, axis, value):
    """
    Trouve toutes les intersections entre un polygone 3D et un plan orthogonal (X=value ou Y=value).
    axis=0 pour X, axis=1 pour Y.
    Retourne une liste de points [X, Y, Z].
    """
    intersections = []
    n = len(polygon_3d)

    for i in range(n):
        p1 = polygon_3d[i]
        p2 = polygon_3d[(i + 1) % n]

        v1, v2 = p1[axis], p2[axis]

        # Vérifie si le plan 'value' coupe le segment
        if (v1 < value < v2) or (v2 < value < v1):
            # Interpolation linéaire
            t = (value - v1) / (v2 - v1)

            intersect_pt = np.zeros(3)
            intersect_pt[axis] = value
            intersect_pt[1 - axis] = p1[1 - axis] + t * (p2[1 - axis] - p1[1 - axis])
            intersect_pt[2] = p1[2] + t * (p2[2] - p1[2]) # Interpolation du Z

            intersections.append(intersect_pt)

        # Cas limite (Sommet touchant exactement la ligne)
        elif v1 == value:
            intersections.append(p1.copy())

    # Supprimer les doublons potentiels (liés aux sommets exacts partagés par 2 arêtes)
    if intersections:
        intersections = np.unique(np.array(intersections), axis=0).tolist()

    return intersections

def min_dist_to_polygon_edges(pt_2d, polygon_2d):
    """Calcule la distance minimum entre un point 2D et toutes les arêtes d'un polygone."""
    # A = points de départ des segments, B = points d'arrivée
    A = polygon_2d
    B = np.roll(polygon_2d, -1, axis=0)

    AB = B - A
    AP = pt_2d - A

    # Produit scalaire pour trouver la projection du point sur la ligne
    dot_AP_AB = np.sum(AP * AB, axis=1)
    dot_AB_AB = np.sum(AB * AB, axis=1)

    # Sécurité contre la division par zéro (points superposés)
    dot_AB_AB[dot_AB_AB == 0] = 1e-8

    # Position relative de la projection sur le segment (clippée entre 0 et 1)
    t = np.clip(dot_AP_AB / dot_AB_AB, 0.0, 1.0)

    # Coordonnées des points projetés
    projs = A + t[:, np.newaxis] * AB

    # Distances euclidiennes
    dists = np.linalg.norm(projs - pt_2d, axis=1)

    return np.min(dists)

def compute_anchor_segments(hole_poly, runway_poly, grid_x_lines, grid_y_lines):
    """
    Calcule les segments d'ancrage stricts dans l'espace du talus (entre H et R).
    Gère les lignes qui coupent H et R, ainsi que celles qui coupent uniquement H.
    Retourne une liste de paires de points 3D: [(P1, P2), (P3, P4), ...]
    """
    segments = []

    def process_axis(axis, grid_lines):
        other_axis = 1 - axis
        for val in grid_lines:
            ints_H = get_polygon_axis_intersections(hole_poly, axis, val)
            ints_R = get_polygon_axis_intersections(runway_poly, axis, val)

            all_ints = ints_H + ints_R
            if len(all_ints) < 2:
                continue

            all_ints.sort(key=lambda pt: pt[other_axis])

            for i in range(len(all_ints) - 1):
                p1 = np.array(all_ints[i])
                p2 = np.array(all_ints[i+1])

                if np.linalg.norm(p2 - p1) < 0.1:
                    continue

                # 4. Le test de vérité : Le point milieu
                mid_pt = (p1 + p2) / 2.0
                mid_pt_2d = mid_pt[:2]

                is_in_H = points_in_polygons_concave(mid_pt_2d.reshape(1, 2), [hole_poly[:, :2]])[0]
                is_in_R = points_in_polygons_concave(mid_pt_2d.reshape(1, 2), [runway_poly[:, :2]])[0]

                if is_in_H and not is_in_R:
                    # 5. NOUVEAU TEST : Rejet des segments confondus avec les frontières
                    dist_to_H = min_dist_to_polygon_edges(mid_pt_2d, hole_poly[:, :2])
                    dist_to_R = min_dist_to_polygon_edges(mid_pt_2d, runway_poly[:, :2])

                    # Marge de tolérance de 10 cm
                    if dist_to_H > 0.1 and dist_to_R > 0.1:
                        segments.append((p1, p2))

    process_axis(0, grid_x_lines)
    process_axis(1, grid_y_lines)

    return segments

def build_cdt_input(hole_poly, runway_poly, anchor_segments):
    """
    Assemble les polygones et les ancrages en une topologie valide pour 'triangle'.
    Puisque les points de grille sont déjà intégrés dans H et R, on se contente
    de retrouver leurs index et de lier les segments.
    """
    # 1. Empilement pur et simple des points existants
    vertices_3d = np.vstack((hole_poly, runway_poly))
    vertices_2d = vertices_3d[:, :2]

    num_H = len(hole_poly)
    num_R = len(runway_poly)

    segments = []

    # 2. Chaînage du contour du trou (H)
    for i in range(num_H):
        segments.append([i, (i + 1) % num_H])

    # 3. Chaînage du contour de la piste (R)
    for i in range(num_R):
        # Les index de R sont décalés de num_H
        segments.append([num_H + i, num_H + ((i + 1) % num_R)])

    # 4. Intégration des segments d'ancrage (lignes oranges)
    for p1, p2 in anchor_segments:
        # Trouver l'index exact de ces points dans notre tableau global
        # On utilise une tolérance très faible puisque les points existent déjà
        dists1 = np.linalg.norm(vertices_3d - p1, axis=1)
        dists2 = np.linalg.norm(vertices_3d - p2, axis=1)

        idx1 = int(np.argmin(dists1))
        idx2 = int(np.argmin(dists2))

        # Sécurité : on s'assure qu'on a bien matché un point existant (tolérance 5 cm)
        if dists1[idx1] < 0.05 and dists2[idx2] < 0.05:
            if idx1 != idx2:  # <-- SÉCURITÉ CRITIQUE ANTI-CRASH CDT
                segments.append([idx1, idx2])
        else:
            logging.warning("Ancrage ignoré : point non trouvé dans les contours natifs.")

    segments_np = np.array(segments, dtype=np.int32)

    # 5. Calcul d'un point 'trou' GARANTI à l'intérieur (Même pour un polygone concave en "U")
    # On trouve le sommet avec le X minimum (C'est mathématiquement un coin convexe)
    idx_min_x = np.argmin(runway_poly[:, 0])
    n_pts = len(runway_poly)
    p_prev = runway_poly[(idx_min_x - 1) % n_pts, :2]
    p_curr = runway_poly[idx_min_x, :2]
    p_next = runway_poly[(idx_min_x + 1) % n_pts, :2]

    # On prend le milieu du segment virtuel qui ferme ce coin
    mid_base = (p_prev + p_next) / 2.0
    # On se décale légèrement (10%) vers ce milieu depuis la pointe
    safe_inside_pt = p_curr + 0.1 * (mid_base - p_curr)
    runway_center_2d = safe_inside_pt.reshape(1, 2)

    cdt_input = {
        'vertices': vertices_2d,
        'segments': segments_np,
        'holes': runway_center_2d
    }

    return cdt_input, vertices_3d

def perform_stitching_cdt(hole_poly, runway_poly, anchor_segments):
    """
    Exécute la Triangulation de Delaunay Contrainte pour relier la piste au terrain.
    Retourne un tableau NumPy de triangles (N, 3) avec des index relatifs au talus,
    et les sommets locaux 3D (qui peuvent inclure de nouveaux points générés par le mailleur).
    """
    # 1. Préparation des données
    cdt_input, local_vertices_3d = build_cdt_input(hole_poly, runway_poly, anchor_segments)

    # 2. Appel au moteur C (triangle)
    try:
        cdt_output = tr.triangulate(cdt_input, 'pzY')
    except Exception as e:
        print(f"Erreur fatale du mailleur CDT : {e}")
        return None, None

    new_faces_local = cdt_output.get('triangles')

    if new_faces_local is None or len(new_faces_local) == 0:
        print("Erreur: Le mailleur n'a généré aucun triangle.")
        return None, None

    # 3. GESTION DES NOUVEAUX SOMMETS (Intersections de grille)
    out_vertices_2d = cdt_output.get('vertices')

    if out_vertices_2d is not None and len(out_vertices_2d) > len(local_vertices_3d):
        num_original = len(local_vertices_3d)
        new_verts_2d = out_vertices_2d[num_original:]

        # CORRECTION : Cast strict en float32 dès la création
        new_verts_3d = np.zeros((len(new_verts_2d), 3), dtype=np.float32)
        new_verts_3d[:, :2] = new_verts_2d

        # Calcul du Z (Altitude) par Inverse Distance Weighting (IDW)
        orig_2d = local_vertices_3d[:, :2]
        orig_z = local_vertices_3d[:, 2]

        for i, pt in enumerate(new_verts_2d):
            dists = np.linalg.norm(orig_2d - pt, axis=1)
            dists[dists < 1e-6] = 1e-6
            weights = 1.0 / (dists ** 2)
            new_verts_3d[i, 2] = np.sum(weights * orig_z) / np.sum(weights)

        local_vertices_3d = np.vstack((local_vertices_3d, new_verts_3d))

        logging.info(f"CDT: {len(new_verts_2d)} point(s) d'intersection de grille résolu(s) et interpolé(s) en 3D.")

    return new_faces_local, local_vertices_3d

# --- AJOUT POUR LE RELIEF FRACTAL (FBM) ---

def _hash2d(x, y):
    """Fonction de hachage pseudo-aléatoire vectorisée pour NumPy."""
    # Combinaisons de constantes pour créer un chaos déterministe
    val = np.sin(x * 12.9898 + y * 78.233) * 43758.5453123
    return val - np.floor(val)

def _value_noise(x, y):
    """Bruit de valeur 2D basique interpolé (Smoothstep)."""
    i = np.floor(x)
    j = np.floor(y)
    f_x = x - i
    f_y = y - j

    # Interpolation hermite (smoothstep) pour adoucir les transitions
    u = f_x * f_x * (3.0 - 2.0 * f_x)
    v = f_y * f_y * (3.0 - 2.0 * f_y)

    # Les 4 coins de la cellule locale
    a = _hash2d(i, j)
    b = _hash2d(i + 1.0, j)
    c = _hash2d(i, j + 1.0)
    d = _hash2d(i + 1.0, j + 1.0)

    # Interpolation bilinéaire
    res = a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
    return res * 2.0 - 1.0  # Ramener entre -1.0 et 1.0

def fbm_noise_2d(x, y, octaves=4, persistence=0.5, lacunarity=2.0, scale=150.0):
    """Génère un Bruit Fractal (Fractal Brownian Motion)."""
    total = np.zeros_like(x)
    amplitude = 1.0
    frequency = 1.0 / scale
    max_value = 0.0

    # Offsets arbitraires pour décaler la grille à chaque octave et éviter les symétries
    offset_x, offset_y = 1000.0, 1000.0

    for _ in range(octaves):
        # Génération du bruit normalisé
        noise_val = _value_noise((x + offset_x) * frequency, (y + offset_y) * frequency)

        total += noise_val * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

        # Décalage de la matrice d'échantillonnage pour la prochaine octave
        offset_x += 100.0
        offset_y += 100.0

    return total / max_value

def apply_fbm_noise_to_selection(vertices, faces, selected_faces_indices, tri_types, amplitude=25.0, octaves=4, scale=150.0, feather_radius=2):
    """
    Applique le bruit fractal FBM sur l'axe Z des sommets sélectionnés avec fondu aux frontières.
    Protège strictement les sommets appartenant à des triangles de type > 0.
    """
    if not selected_faces_indices:
        return vertices

    local_faces = faces[selected_faces_indices]
    global_vert_indices = np.unique(local_faces)
    N = len(global_vert_indices)

    # 1. Mapping local-global
    global_to_local_map = np.full(len(vertices), -1, dtype=np.int32)
    global_to_local_map[global_vert_indices] = np.arange(N)
    local_faces_mapped = global_to_local_map[local_faces]

    # 2. Feathering par BFS (Masque de fusion aux bords de la sélection)
    edges = np.vstack((
        local_faces_mapped[:, [0, 1]],
        local_faces_mapped[:, [1, 2]],
        local_faces_mapped[:, [2, 0]]
    ))
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_vertices = np.unique(unique_edges[counts == 1])

    adj = [[] for _ in range(N)]
    for v1, v2 in unique_edges:
        adj[v1].append(v2)
        adj[v2].append(v1)

    distances = np.full(N, -1, dtype=np.int32)
    queue = deque(boundary_vertices)
    for bv in boundary_vertices:
        distances[bv] = 0

    while queue:
        curr = queue.popleft()
        d = distances[curr]
        for neighbor in adj[curr]:
            if distances[neighbor] == -1:
                distances[neighbor] = d + 1
                queue.append(neighbor)

    alphas = np.zeros(N, dtype=np.float32)
    for i in range(N):
        d = distances[i]
        if d <= 0:
            alphas[i] = 0.0 # Bords de sélection préservés
        elif d >= feather_radius:
            alphas[i] = 1.0 # Centre altéré à 100%
        else:
            alphas[i] = (1.0 - np.cos((d / feather_radius) * np.pi)) / 2.0

    # --- 2.5 PROTECTION DES SOMMETS SPÉCIAUX (TYPE > 0) ---
    # Identifier tous les sommets globaux touchant au moins un triangle de type > 0
    protected_global_verts = np.unique(faces[tri_types > 0])

    # Trouver leurs indices locaux dans notre sélection actuelle
    protected_local_indices = global_to_local_map[protected_global_verts]
    # Ne garder que ceux qui font effectivement partie de la sélection (valeur >= 0)
    protected_local_indices = protected_local_indices[protected_local_indices >= 0]

    # Forcer l'alpha à 0.0 pour annuler tout déplacement sur ces sommets
    alphas[protected_local_indices] = 0.0
    # ------------------------------------------------------

    # 3. Application du Bruit FBM
    local_vertices = vertices[global_vert_indices]
    x_coords = local_vertices[:, 0]
    y_coords = local_vertices[:, 1]

    # Génération du bruit (entre -1.0 et 1.0)
    noise_values = fbm_noise_2d(x_coords, y_coords, octaves=octaves, scale=scale)

    # Modification de l'axe Z (Bruit * Amplitude * Masque_Alpha_Corrigé)
    vertices[global_vert_indices, 2] += noise_values * amplitude * alphas

    return vertices