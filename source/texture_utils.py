import math
import os
import re

def meters_to_latlon(x_m, y_m, lon_to_m, lat_to_m, x_center, y_center):
    """Convertit les coordonnées locales X/Y (mètres) en Longitude/Latitude."""
    lon = (x_m / lon_to_m) + x_center
    lat = (y_m / lat_to_m) + y_center
    return lat, lon

def get_tile_coords_for_zl(lat, lon, zl):
    """Calcule les coordonnées Ortho4XP (blocs de 16) pour un Lat/Lon et un ZL donnés."""
    n = 2.0 ** zl
    lat_rad = math.radians(lat)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - (math.asinh(math.tan(lat_rad)) / math.pi)) / 2.0 * n)

    til_x_left = (x // 16) * 16
    til_y_top = (y // 16) * 16

    return til_x_left, til_y_top

def find_best_texture_match(lat, lon, dds_dir):
    """
    Scanne le dossier DDS, extrait les ZL et identifie la meilleure texture
    couvrant les coordonnées actuelles (Priorité: ZL max, puis Fichier le plus récent).
    """
    pattern = re.compile(r"^(\d+)_(\d+)_([a-zA-Z0-9_-]+?)(\d{2})\.(?:dds|ter)$", re.IGNORECASE)
    matches = []

    if not os.path.exists(dds_dir):
        return None

    for filename in os.listdir(dds_dir):
        match = pattern.match(filename)
        if match:
            filepath = os.path.join(dds_dir, filename)
            matches.append({
                'filename': filename,
                'til_y': int(match.group(1)),
                'til_x': int(match.group(2)),
                'provider': match.group(3),
                'zl': int(match.group(4)),
                'timestamp': os.path.getmtime(filepath)
            })

    expected_coords = {}
    for m in matches:
        zl = m['zl']
        if zl not in expected_coords:
            expected_coords[zl] = get_tile_coords_for_zl(lat, lon, zl)

    valid_matches = []
    for m in matches:
        zl = m['zl']
        if zl in expected_coords:
            exp_x, exp_y = expected_coords[zl]
            if m['til_x'] == exp_x and m['til_y'] == exp_y:
                valid_matches.append(m)

    if not valid_matches:
        return None

    valid_matches.sort(key=lambda x: (x['zl'], x['timestamp']), reverse=True)
    return valid_matches[0]

def latlon_to_meters(lat, lon, lon_to_m, lat_to_m, x_center, y_center):
    """Convertit les coordonnées Longitude/Latitude en coordonnées locales X/Y (mètres)."""
    x_m = (lon - x_center) * lon_to_m
    y_m = (lat - y_center) * lat_to_m
    return x_m, y_m

def get_texture_grid_lines(min_xm, max_xm, min_ym, max_ym, zl, lon_to_m, lat_to_m, x_center, y_center):
    """
    Calcule les coordonnées X et Y (en mètres locaux) des lignes de découpe
    des textures DDS d'Ortho4XP pour une zone délimitée (bounding box).
    """
    # 1. Convertir la bounding box locale en Lat/Lon
    min_lat, min_lon = meters_to_latlon(min_xm, min_ym, lon_to_m, lat_to_m, x_center, y_center)
    max_lat, max_lon = meters_to_latlon(max_xm, max_ym, lon_to_m, lat_to_m, x_center, y_center)

    # S'assurer du bon ordre (min/max)
    lat_start, lat_end = min(min_lat, max_lat), max(min_lat, max_lat)
    lon_start, lon_end = min(min_lon, max_lon), max(min_lon, max_lon)

    n = 2.0 ** zl

    # 2. Convertir les Lat/Lon extrêmes en coordonnées Slippy (flottants)
    x_slippy_start = (lon_start + 180.0) / 360.0 * n
    x_slippy_end = (lon_end + 180.0) / 360.0 * n

    # Attention: L'axe Y Slippy est inversé par rapport à la latitude (Y augmente quand Lat diminue)
    y_slippy_start = (1.0 - (math.asinh(math.tan(math.radians(lat_end))) / math.pi)) / 2.0 * n
    y_slippy_end = (1.0 - (math.asinh(math.tan(math.radians(lat_start))) / math.pi)) / 2.0 * n

    # 3. Trouver tous les multiples de 16 stricts dans ces intervalles (Limites des DDS)
    x_grid_indices = range(math.ceil(x_slippy_start / 16.0) * 16, math.floor(x_slippy_end) + 1, 16)
    y_grid_indices = range(math.ceil(y_slippy_start / 16.0) * 16, math.floor(y_slippy_end) + 1, 16)

    x_lines_m = []
    y_lines_m = []

    # 4. Reconvertir ces indices stricts en coordonnées locales (mètres)
    for x_idx in x_grid_indices:
        lon = (x_idx / n) * 360.0 - 180.0
        x_m, _ = latlon_to_meters(y_center, lon, lon_to_m, lat_to_m, x_center, y_center)
        x_lines_m.append(x_m)

    for y_idx in y_grid_indices:
        # Formule inverse de Slippy Y vers Latitude
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_idx / n))))
        _, y_m = latlon_to_meters(lat, x_center, lon_to_m, lat_to_m, x_center, y_center)
        y_lines_m.append(y_m)

    return sorted(x_lines_m), sorted(y_lines_m)