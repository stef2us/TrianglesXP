import json
import sys
import math
import os
import re
import configparser
import numpy as np
import cv2
import logging
import shutil
import gc
import psutil

from PIL import Image

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLineEdit, QFileDialog,
                             QLabel, QMessageBox, QCheckBox, QDoubleSpinBox,
                             QSlider, QSplitter, QTabWidget, QFrame, QGroupBox,
                             QRubberBand, QGridLayout, QShortcut, QTableWidget,
                             QTableWidgetItem, QHeaderView, QInputDialog,
                             QComboBox, QRadioButton, QButtonGroup, QSpinBox,
                             QDialog, QProgressBar, QAbstractItemView,
                             QToolButton, QSizePolicy)

from PyQt5.QtGui import QPalette, QColor, QKeySequence
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, QTimer

from functools import wraps

from vispy import scene
from vispy.visuals.filters import WireframeFilter, TextureFilter
from vispy.visuals.transforms import STTransform

from logging.handlers import RotatingFileHandler

from concurrent.futures import ThreadPoolExecutor

from i18n import load_language, _
from texture_utils import meters_to_latlon, find_best_texture_match, get_texture_grid_lines
from mesh_io import parse_ortho_mesh, write_ortho_mesh
from mesh_ops import (subdivide_mesh_selection, apply_cotangent_smooth,
                      find_triangles_in_rect, split_closest_edge,
                      get_airport_centers, get_faces_in_cylinder,
                      get_selection_boundary_2d, points_to_segments_dist,
                      calculate_earthwork_blend, get_cosine_blend_factor,
                      generate_sliced_runway_mesh, points_in_polygons_concave,
                      get_ordered_boundary_loop, faces_intersecting_polygon,
                      compute_anchor_segments, perform_stitching_cdt,
                      apply_fbm_noise_to_selection)

def wait_cursor(func):
    """Décorateur pour afficher un sablier pendant l'exécution d'une fonction."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            return func(*args, **kwargs)
        finally:
            QApplication.restoreOverrideCursor()
    return wrapper

class OrthoMeshViewer(QMainWindow):

    # =========================================================================
    #
    # 1. INITIALISATION & CONFIGURATION
    #
    # =========================================================================

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrianglesXP 0.28.2")
        self.resize(1280, 800)

        # 1. Initialisation de toutes les variables d'état
        self.init_state_variables()

        # 2. Création du conteneur principal et du Splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- ZONE GAUCHE : VISPY ---
        self.vispy_container = QWidget()
        self.vispy_layout = QVBoxLayout(self.vispy_container)
        self.vispy_layout.setContentsMargins(0, 0, 0, 0)
        self.setup_vispy_ui(self.vispy_layout)
        self.splitter.addWidget(self.vispy_container)

        # --- ZONE DROITE : PANNEAU DE CONTRÔLE ---
        self.control_panel = QWidget()
        # On fixe une largeur minimum pour éviter que les boutons soient écrasés
        self.control_panel.setMinimumWidth(390)
        self.control_layout = QVBoxLayout(self.control_panel)
        self.splitter.addWidget(self.control_panel)

        # Configuration des proportions initiales du Splitter (ex: 75% gauche, 25% droite)
        self.splitter.setSizes([890, 390])

        # 2.5. Chargement des paramètres sauvegardés (.ini)
        self.load_pre_settings()

        # 3. Construction de l'interface dans le panneau droit
        self.build_side_panel()

        # 4. Chargement des paramètres sauvegardés (.ini)
        self.load_settings()

        # 5. Configuration des raccourcis globaux
        self.setup_global_shortcuts()

        # 6. Initialisation du préréglage de relief par défaut
        self.load_relief_presets()
        self.apply_fbm_preset("Smooth")

    def init_state_variables(self):
        """Déclare toutes les variables globales utilisées par l'application."""
        # Variables issues de settings.ini au cas où fichier inexistant
        self.FONT_SIZE = 10
        self.MAX_HISTORY = 10
        self.MAX_TEXTURES = 4

        self.MAX_SUBDIV_LEVEL = 4
        self.FLATTEN_BANK = 100
        self.ZONE_SUBDIV = 2
        self.RUNWAY_MAX_PTS = 7
        self.RUNWAY_WIDTH = 40
        self.RUNWAY_MAX_WIDTH = 500
        self.RUNWAY_BANK_SIDE = 100
        self.RUNWAY_SUBDIV = 2
        self.RUNWAY_RECT_SIZE = 10
        self.RUNWAY_TYPE = 8

        self.CYL_RADIUS = 1000
        self.CYL_MAX_RADIUS = 3000
        self.CYL_BASE = 200
        self.CYL_MAX_BASE = 2000
        self.SMOOTH_ITER = 5
        self.SMOOTH_ALPHA = 0.25
        self.SMOOTH_FEATH_RAD = 2
        self.SMOOTH_LOOPS = 2
        self.SMOOTH_LOOPS_MIN = 0
        self.SMOOTH_LOOPS_MAX = 3
        self.FBM_PRECISION_MIN = 0
        self.FBM_PRECISION_MAX = 2
        self.FBM_OCTAVES_MIN = 1
        self.FBM_OCTAVES_MAX = 4
        self.FBM_AMPLITUDE_MIN = 10
        self.FBM_AMPLITUDE_MAX = 1000
        self.FBM_SCALE_MIN = 10
        self.FBM_SCALE_MAX = 1000
        self.RETOUCH_THRESH_B = 0
        self.RETOUCH_THRESH_W = 220

        self.POINT_RANGE_MIN = -1000
        self.POINT_RANGE_MAX = 1000
        self.BRUSH_RADIUS_MIN = 1
        self.BRUSH_RADIUS_MAX = 30
        self.BRUSH_RADIUS_DEF = 15

        self.SMOOTH_LOOPS_LOW = 1
        self.SMOOTH_LOOPS_MID = 1
        self.SMOOTH_LOOPS_HI = 1
        self.FBM_PRECISION_LOW = 0
        self.FBM_PRECISION_MID = 0
        self.FBM_PRECISION_HI = 0
        self.FBM_OCTAVES_LOW = 4
        self.FBM_OCTAVES_MID = 3
        self.FBM_OCTAVES_HI = 2
        self.FBM_AMPLITUDE_LOW = 25
        self.FBM_AMPLITUDE_MID = 35
        self.FBM_AMPLITUDE_HI = 45
        self.FBM_SCALE_LOW = 150
        self.FBM_SCALE_MID = 100
        self.FBM_SCALE_HI = 65
        self.FBM_SCALE_HI2 = 40

        # Etats
        self.is_modified = False
        self.show_airports = False
        self.show_lakes = False
        self.show_roads = False
        self.show_taxi = False
        self.show_infra = False
        self.show_cross = False
        self.mesh_visual = None
        self.selected_faces_indices = []

        # HISTORIQUE UNDO/REDO ===
        self.undo_stack = []
        self.redo_stack = []
        self.new_selection = True # pour bouton Compare

        # Textures
        self.active_texture_meshes = []
        self.active_texture_names = []
        self.texture_mesh_visual = None

        # Point edition
        self.is_editing_point = False
        self.active_new_vertex_idx = None

        # Flatten tool
        self.flatten_plane = None
        self.flatten_active = False
        self._slider_center_z = 0.0

        # Zone Tool
        self.zone_draw_mode = False
        self.zone_polygon_points = []   # Stocke les points 3D [x, y, z] du tracé
        self.zone_polygon_redo_stack = [] # Pile pour le CTRL+Y du lasso 3D
        self.zone_polygon_visual = None
        self.zone_markers_visual = None
        self.current_zone_name = ""
        self.current_runway_name = ""

        # Cylinder Tool
        self.cylinder_active = False
        self.cylinder_visual = None
        self._cyl_slider_center_z = 0.0
        self.cylinder_center = (0.0, 0.0)
        self.last_smooth_max_z = None

        # --- Outil Piste (Runway) ---
        self.runway_active = False
        self.runway_pts_2d = []
        self.runway_visuals = []

        # Mode Retouche 2D
        self.image_2d_visual = None
        self.camera_3d_state = {}
        self.is_2d_mode = False
        self.grid_2d_visual = None
        self._global_grid_rects = []
        self.camera_2d_state = {}
        self.selection_mask = None        # Masque binaire (0 ou 255) de taille HxW
        self.lasso_mode = "add" # Modes : "new", "add", "sub"
        self.active_selection_polygons = [] # Garde en mémoire les tracés bruts du masque
        self.selection_visual = None      # Objet VisPy pour afficher le rouge
        self.overlay_color = [255, 0, 0, 100] # Rouge semi-transparent (R, G, B, Alpha)
        self.polygon_points = []
        self.polygon_visual = None
        self.polygon_redo_stack = []
        self.polygon_markers_visual = None
        self.current_sel_name = ""          # Nom interne de la sélection
        self.is_selection_modified = False  # Flag "Masque altéré"
        self.is_color_preset_modified = False
        self.batch_2d_state = None # États possibles : None, "show_masks"

        # --- Mode Retouche Couleur ---
        self.retouch_preview_visual = None
        self.retouch_roi_data = None
        self.is_retouching = False
        self._color_apply_needs_pulse = False

        # --- Timer pour le recalcul du contour au zoom ---
        self._zoom_timer = QTimer()
        self._zoom_timer.setSingleShot(True) # Ne s'exécute qu'une seule fois après le délai
        self._zoom_timer.timeout.connect(self.update_selection_visual)

        # --- TIMER POUR L'ANIMATION PULSE ---
        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self.animate_pulse)
        self.pulse_state = False

        # Mean Shift
        self.is_blur_mode = False
        self.is_brush_drawing = False
        self.brush_points = []
        self.brush_visual = None
        self.brush_radius = 5
        self.brush_cursor_visual = None
        self.heal_history = []
        self.heal_redo_stack = []
        self.has_healing_edits = False

        # Seamless Clone
        self.is_seamless_mode = False
        self.is_seamless_dragging = False
        self.seamless_start_pos = None
        self.seamless_ghost_visual = None
        self.seamless_contour_pts = None

        # Coordonnées géospatiales
        self.mean_lat = 0.0
        self.lat_to_m = 111120.0
        self.lon_to_m = 111120.0
        self.x_center = 0.0
        self.y_center = 0.0
        self.z_min = 0.0
        self.z_max = 0.0

    def load_pre_settings(self):
        self.config_file = "settings.ini"
        self.config = configparser.ConfigParser()

        if os.path.exists(self.config_file):
            self.config.read(self.config_file, encoding='utf-8')

            try:
                # --- Section [Window] ---
                if 'Window' in self.config:
                    w_cfg = self.config['Window']

                    # 1. Taille de la fenêtre
                    width = int(w_cfg.get('width', '1280'))
                    height = int(w_cfg.get('height', '800'))
                    self.resize(width, height)

                    # 2. État maximisé
                    if w_cfg.getboolean('maximized', False):
                        self.showMaximized()

                    # 3. État du Splitter (Proportions gauche/droite)
                    splitter_data = w_cfg.get('splitter_sizes', '')
                    if splitter_data:
                        try:
                            sizes = [int(s) for s in splitter_data.split(',')]
                            self.splitter.setSizes(sizes)
                        except ValueError:
                            pass # Format de données invalide, on ignore

                    # 4. Taille de la police globale ===
                    self.FONT_SIZE = int(w_cfg.get('font_size', '10'))
                    app_font = QApplication.instance().font()
                    app_font.setPointSize(self.FONT_SIZE)
                    QApplication.instance().setFont(app_font)

                # --- Section [System] ---
                if 'System' in self.config:
                    s = self.config['System']
                    self.MAX_HISTORY = int(s.get('max_history', '10'))
                    self.MAX_TEXTURES = int(s.get('max_textures', '4'))

                # --- Section [Tools] ---
                if 'Tools' in self.config:
                    t = self.config['Tools']
                    self.MAX_SUBDIV_LEVEL = int(t.get('max_subdiv_level', '4'))
                    self.FLATTEN_BANK = int(t.get('flatten_bank', '100'))
                    self.ZONE_SUBDIV = int(t.get('zone_subdiv', '2'))

                    self.RUNWAY_MAX_PTS = int(t.get('runway_max_pts', '7'))
                    self.RUNWAY_WIDTH = int(t.get('runway_width', '40'))
                    self.RUNWAY_MAX_WIDTH = int(t.get('runway_max_width', '500'))
                    self.RUNWAY_BANK_SIDE = int(t.get('runway_bank_side', '100'))
                    self.RUNWAY_SUBDIV = int(t.get('runway_subdiv', '2'))
                    self.RUNWAY_RECT_SIZE = int(t.get('runway_rect_size', '10'))
                    self.RUNWAY_TYPE = int(t.get('runway_type', '8'))

                    self.CYL_RADIUS = int(t.get('cyl_radius', '1000'))
                    self.CYL_MAX_RADIUS = int(t.get('cyl_max_radius', '3000'))
                    self.CYL_BASE = int(t.get('cyl_base', '200'))
                    self.CYL_MAX_BASE = int(t.get('cyl_max_base', '2000'))
                    self.SMOOTH_ITER = int(t.get('smooth_iter', '5'))
                    self.SMOOTH_ALPHA = float(t.get('smooth_alpha', '0.25'))
                    self.SMOOTH_FEATH_RAD = int(t.get('smooth_feath_rad', '2'))

                    self.SMOOTH_LOOPS = int(t.get('smooth_loops', '2'))
                    self.SMOOTH_LOOPS_MIN = int(t.get('smooth_loops_min', '0'))
                    self.SMOOTH_LOOPS_MAX = int(t.get('smooth_loops_max', '3'))
                    self.FBM_PRECISION_MIN = int(t.get('fbm_precision_min', '0'))
                    self.FBM_PRECISION_MAX = int(t.get('fbm_precision_max', '2'))
                    self.FBM_OCTAVES_MIN = int(t.get('fbm_octaves_min', '1'))
                    self.FBM_OCTAVES_MAX = int(t.get('fbm_octaves_max', '4'))
                    self.FBM_AMPLITUDE_MIN = int(t.get('fbm_amplitude_min', '10'))
                    self.FBM_AMPLITUDE_MAX = int(t.get('fbm_amplitude_max', '1000'))
                    self.FBM_SCALE_MIN = int(t.get('fbm_scale_min', '10'))
                    self.FBM_SCALE_MAX = int(t.get('fbm_scale_max', '1000'))

                    self.RETOUCH_THRESH_B = int(t.get('retouch_thresh_b', '0'))
                    self.RETOUCH_THRESH_W = int(t.get('retouch_thresh_w', '220'))

                    self.POINT_RANGE_MIN = int(t.get('point_range_min', '-1000'))
                    self.POINT_RANGE_MAX = int(t.get('point_range_max', '1000'))
                    self.BRUSH_RADIUS_MIN = int(t.get('brush_radius_min', '1'))
                    self.BRUSH_RADIUS_MAX = int(t.get('brush_radius_max', '30'))
                    self.BRUSH_RADIUS_DEF = int(t.get('brush_radius_def', '15'))
                    self.brush_radius = self.BRUSH_RADIUS_DEF

            except ValueError:
                QMessageBox.warning(self, _("msg_error_title"), _("msg_invalid_parms"))
                logging.error("Failed to parse settings.ini. Corrupted numerical value detected.", exc_info=True)
                return

    def load_settings(self):
        self.config_file = "settings.ini"
        self.config = configparser.ConfigParser()

        if os.path.exists(self.config_file):
            self.config.read(self.config_file, encoding='utf-8')

            try:
                # --- Section [Lang] ---
                if 'Lang' in self.config:
                    lang = self.config['Lang'].get('language', 'en')
                    index = self.combo_lang.findData(lang)
                    if index >= 0: self.combo_lang.setCurrentIndex(index)

                # --- Section [Paths] ---
                if 'Paths' in self.config:
                    tex_dir = self.config['Paths'].get('texture_dir', '')
                    if tex_dir: self.tex_dir_input.setText(os.path.normpath(tex_dir))
                    last_mesh = self.config['Paths'].get('last_mesh_dir', '')
                    if last_mesh: self.last_mesh_dir = os.path.normpath(last_mesh)
                    dds_dir = self.config['Paths'].get('dds_dir', '')
                    if dds_dir: self.dds_dir_input.setText(os.path.normpath(dds_dir))

            except ValueError:
                QMessageBox.warning(self, _("msg_error_title"), _("msg_invalid_parms"))
                logging.error("Failed to parse settings.ini. Corrupted numerical value detected.", exc_info=True)
                return

    def save_settings(self):
        if 'Window' not in self.config: self.config['Window'] = {}
        if 'Lang' not in self.config: self.config['Lang'] = {}
        if 'Paths' not in self.config: self.config['Paths'] = {}
        if 'System' not in self.config: self.config['System'] = {}
        if 'Tools' not in self.config: self.config['Tools'] = {}

        # Window
        self.config['Window']['maximized'] = str(self.isMaximized())
        if not self.isMaximized():
            self.config['Window']['width'] = str(self.width())
            self.config['Window']['height'] = str(self.height())
        self.config['Window']['splitter_sizes'] = ",".join(map(str, self.splitter.sizes()))
        self.config['Window']['font_size'] = str(getattr(self, 'FONT_SIZE', 10))

        # Lang
        self.config['Lang']['language'] = self.combo_lang.currentData()

        # Paths
        self.config['Paths']['texture_dir'] = self.tex_dir_input.text()
        self.config['Paths']['last_mesh_dir'] = getattr(self, 'last_mesh_dir', '')
        self.config['Paths']['dds_dir'] = self.dds_dir_input.text()

        # System
        self.config['System']['max_history'] = str(self.MAX_HISTORY)
        self.config['System']['max_textures'] = str(self.MAX_TEXTURES)

        # Tools
        t = self.config['Tools']
        t['max_subdiv_level'] = str(self.MAX_SUBDIV_LEVEL)
        t['flatten_bank'] = str(self.FLATTEN_BANK)
        t['zone_subdiv'] = str(self.ZONE_SUBDIV)

        t['runway_max_pts'] = str(self.RUNWAY_MAX_PTS)
        t['runway_width'] = str(self.RUNWAY_WIDTH)
        t['runway_max_width'] = str(self.RUNWAY_MAX_WIDTH)
        t['runway_bank_side'] = str(self.RUNWAY_BANK_SIDE)
        t['runway_subdiv'] = str(self.RUNWAY_SUBDIV)
        t['runway_rect_size'] = str(self.RUNWAY_RECT_SIZE)
        t['runway_type'] = str(self.RUNWAY_TYPE)

        t['cyl_radius'] = str(self.CYL_RADIUS)
        t['cyl_max_radius'] = str(self.CYL_MAX_RADIUS)
        t['cyl_base'] = str(self.CYL_BASE)
        t['cyl_max_base'] = str(self.CYL_MAX_BASE)
        t['smooth_iter'] = str(self.SMOOTH_ITER)
        t['smooth_alpha'] = str(self.SMOOTH_ALPHA)
        t['smooth_feath_rad'] = str(self.SMOOTH_FEATH_RAD)
        t['smooth_loops'] = str(self.SMOOTH_LOOPS)
        t['smooth_loops_min'] = str(self.SMOOTH_LOOPS_MIN)
        t['smooth_loops_max'] = str(self.SMOOTH_LOOPS_MAX)
        t['fbm_precision_min'] = str(self.FBM_PRECISION_MIN)
        t['fbm_precision_max'] = str(self.FBM_PRECISION_MAX)
        t['fbm_octaves_min'] = str(self.FBM_OCTAVES_MIN)
        t['fbm_octaves_max'] = str(self.FBM_OCTAVES_MAX)
        t['fbm_amplitude_min'] = str(self.FBM_AMPLITUDE_MIN)
        t['fbm_amplitude_max'] = str(self.FBM_AMPLITUDE_MAX)
        t['fbm_scale_min'] = str(self.FBM_SCALE_MIN)
        t['fbm_scale_max'] = str(self.FBM_SCALE_MAX)
        t['retouch_thresh_b'] = str(self.RETOUCH_THRESH_B)
        t['retouch_thresh_w'] = str(self.RETOUCH_THRESH_W)
        t['point_range_min'] = str(self.POINT_RANGE_MIN)
        t['point_range_max'] = str(self.POINT_RANGE_MAX)
        t['brush_radius_min'] = str(self.BRUSH_RADIUS_MIN)
        t['brush_radius_max'] = str(self.BRUSH_RADIUS_MAX)
        t['brush_radius_def'] = str(self.BRUSH_RADIUS_DEF)

        with open(self.config_file, 'w', encoding='utf-8') as configfile:
            self.config.write(configfile)

    def setup_global_shortcuts(self):
        """Configure les raccourcis clavier globaux de l'application."""

        # Fichier
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.export_mesh)

        # Touche Echap pour annuler la sélection
        self.shortcut_escape = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.shortcut_escape.activated.connect(self.cancel_selection)

        # UNDO / REDO ===
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo_action)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.redo_action)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self).activated.connect(self.redo_action)

        # Raccourcis pour les calques visuels
        QShortcut(QKeySequence("A"), self).activated.connect(
            lambda: self.cb_show_airports.setChecked(not self.cb_show_airports.isChecked()) if not getattr(self, 'is_2d_mode', False) else None)

        QShortcut(QKeySequence("W"), self).activated.connect(
            lambda: self.cb_show_lakes.setChecked(not self.cb_show_lakes.isChecked()) if not getattr(self, 'is_2d_mode', False) else None)

        QShortcut(QKeySequence("R"), self).activated.connect(
            lambda: self.cb_show_roads.setChecked(not self.cb_show_roads.isChecked()) if not getattr(self, 'is_2d_mode', False) else None)

        # Raccourcis pour la texture satellite
        QShortcut(QKeySequence("T"), self).activated.connect(self.toggle_texture)
        QShortcut(QKeySequence("D"), self).activated.connect(self.clear_textures)

        # Raccourci pour la texture globale
        QShortcut(QKeySequence("G"), self).activated.connect(self._shortcut_toggle_global)

    def _shortcut_toggle_global(self):
        """Contourne le verrouillage des onglets pour forcer l'affichage de la texture."""
        if getattr(self, 'is_2d_mode', False):
            return

        # On inverse l'état du bouton manuellement et on appelle la fonction
        current_state = self.btn_toggle_global_tex.isChecked()
        self.btn_toggle_global_tex.setChecked(not current_state)
        self.toggle_global_texture_display()

    def build_side_panel(self):
        # --- LES ONGLETS ---
        self.tabs = QTabWidget()
        # On force la police en gras directement sur l'objet Qt
        # pour que le moteur géométrique calcule la bonne largeur des onglets.
        tab_font = self.tabs.font()
        tab_font.setBold(True)
        self.tabs.setFont(tab_font)
        self.control_layout.addWidget(self.tabs)

        # 1. Mesh
        tab_proj = QWidget(); lay_proj = QVBoxLayout(tab_proj)
        self.setup_file_ui(lay_proj)
        self.setup_texture_ui(lay_proj)
        self.setup_language_ui(lay_proj)
        self.setup_cleanup_ui(lay_proj)
        lay_proj.addStretch(); self.tabs.addTab(tab_proj, _("tab_mesh"))

        # 2. Affichage
        tab_vues = QWidget(); lay_vues = QVBoxLayout(tab_vues)
        self.setup_texture_actions_ui(lay_vues)
        self.setup_filter_ui(lay_vues)
        lay_vues.addStretch(); self.tabs.addTab(tab_vues, _("tab_vue"))

        # 3. Aplanissement
        tab_sel = QWidget(); lay_sel = QVBoxLayout(tab_sel)
        self.setup_zone_ui(lay_sel)
        self.setup_point_ui(lay_sel)
        self.setup_type_modifier_ui(lay_sel)
        lay_sel.addStretch(); self.tabs.addTab(tab_sel, _("tab_airport"))

        # 4. Piste
        tab_piste = QWidget(); lay_piste = QVBoxLayout(tab_piste)
        self.setup_runway_ui(lay_piste)
        lay_piste.addStretch(); self.tabs.addTab(tab_piste, _("tab_altiport"))

        # 5. Relief
        tab_ret = QWidget(); lay_ret = QVBoxLayout(tab_ret)
        self.setup_topo_ui(lay_ret)
        lay_ret.addStretch(); self.tabs.addTab(tab_ret, _("tab_relief"))

        # 6. Couleurs
        tab_ed = QWidget(); lay_ed = QVBoxLayout(tab_ed)
        self.setup_retouch_ui(lay_ed)
        lay_ed.addStretch(); self.tabs.addTab(tab_ed, _("tab_edit"))

        # Verrouillage initial des onglets sauf Mesh
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, False)

        # --- PANNEAU PERMANENT (BAS) ---
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(5, 5, 5, 5)

        # === Compteur de triangles ===
        self.lbl_mesh_stats = QLabel("Triangles : 0")
        self.lbl_mesh_stats.setAlignment(Qt.AlignCenter)
        # Style discret mais lisible pour s'intégrer au thème sombre
        self.lbl_mesh_stats.setStyleSheet("color: #bdc3c7; font-weight: bold; margin-bottom: 2px;")
        bottom_layout.addWidget(self.lbl_mesh_stats)

        # === Boutons Historique ===
        history_layout = QHBoxLayout()
        self.btn_undo = QPushButton(_("btn_undo"))
        self.btn_undo.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        self.btn_undo.clicked.connect(self.undo_action)
        self.btn_undo.setEnabled(False)
        history_layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton(_("btn_redo"))
        self.btn_redo.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        self.btn_redo.clicked.connect(self.redo_action)
        self.btn_redo.setEnabled(False)
        history_layout.addWidget(self.btn_redo)

        # Bouton global d'annulation de sélection
        self.btn_cancel_selection = QPushButton(_("btn_cancel_sel"))
        self.btn_cancel_selection.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 4px;
            }
            #QPushButton:hover { background-color: #e74c3c; }
        """)
        self.btn_cancel_selection.clicked.connect(self.cancel_selection)
        history_layout.addWidget(self.btn_cancel_selection)
        bottom_layout.addLayout(history_layout)

        # Bouton Quitter tout en bas
        self.btn_quit = QPushButton(_("btn_quit"))
        self.btn_quit.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; padding: 10px;")
        self.btn_quit.clicked.connect(self.close)
        bottom_layout.addWidget(self.btn_quit)

        self.control_layout.addWidget(bottom_container)

    def closeEvent(self, event):
        """Intercepte la fermeture de la fenêtre pour vérifier les modifications non sauvegardées."""
        # 1. On sauvegarde les chemins dans le .ini quoi qu'il arrive
        self.save_settings()

        # 2. Vérification des modifications du Mesh
        mesh_modified = getattr(self, 'is_modified', False)

        # 3. Vérification des modifications de Retouches (Masque actif ET non exporté)
        # 3. Vérification des modifications de Retouches (Masque actif ou pinceau)
        has_color_edits = getattr(self, 'cumulative_mask', None) is not None and np.any(self.cumulative_mask)
        has_retouches = has_color_edits or getattr(self, 'has_healing_edits', False)
        retouch_modified = has_retouches and not getattr(self, 'is_retouch_exported', False)

        if mesh_modified or retouch_modified:
            # Construction du message d'alerte
            warning_msg = ""
            if mesh_modified:
                warning_msg += _("msg_mesh_modified")

            if retouch_modified:
                if warning_msg:
                    warning_msg += "\n\n" + "-"*40 + "\n\n"
                # On réutilise ta traduction existante pour les retouches non exportées
                warning_msg += _("msg_retouch_not_saved")

            reply = QMessageBox.question(self, _("msg_quit_editor"), warning_msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                logging.info("Application closed confirmed (Changes lost).")
                event.accept() # Autorise la fermeture de la fenêtre
            else:
                event.ignore() # Annule la fermeture, l'utilisateur reste sur l'app
        else:
            # S'il n'y a pas de modifications, on ferme direct
            event.accept()

    # =========================================================================
    #
    # 2. CONSTRUCTIONS DES COMPOSANTS UI (PyQt5)
    #
    # =========================================================================

    def setup_vispy_ui(self, main_layout):
        self.canvas = scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45, distance=150000)
        main_layout.addWidget(self.canvas.native, stretch=1)

        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.canvas.native)
        self.drag_origin = QPoint()
        self.drag_button = None

        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)

        self.canvas.events.key_press.connect(self.on_key_press)

        # Écoute du zoom
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)

    def setup_file_ui(self, parent_layout):
        """Configure la section de gestion du fichier Mesh (Chargement / Exportation)."""

        group_file = QGroupBox(_("group_file"))
        layout_file = QVBoxLayout()

        # 1. Champ de texte affichant le chemin du fichier
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText(_("ph_no_file"))
        self.file_input.setReadOnly(True)
        # On peut ajouter un petit style pour bien montrer qu'il est en lecture seule
        self.file_input.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa; padding: 4px; border: 1px solid #555;")
        layout_file.addWidget(self.file_input)

        # 2. Bouton Charger
        layout_file_io = QHBoxLayout()
        btn_load = QPushButton(_("btn_load"))
        btn_load.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; padding: 6px;")
        btn_load.clicked.connect(self.load_mesh)
        layout_file_io.addWidget(btn_load)

        # 3. Bouton Exporter
        btn_export = QPushButton(_("btn_export"))
        btn_export.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 6px;")
        btn_export.clicked.connect(self.export_mesh)
        layout_file_io.addWidget(btn_export)

        # 4. Bouton Reset Mesh
        self.btn_reset_mesh = QPushButton(_("btn_reset_mesh"))
        self.btn_reset_mesh.setStyleSheet("color: white; font-weight: bold; padding: 6px;")
        self.btn_reset_mesh.clicked.connect(self.reset_mesh)
        layout_file_io.addWidget(self.btn_reset_mesh)
        layout_file.addLayout(layout_file_io)

        # Application du layout au groupe, puis ajout au panneau parent
        group_file.setLayout(layout_file)
        parent_layout.addWidget(group_file)

    def setup_texture_ui(self, parent_layout):
        """Configure uniquement la sélection des dossiers de textures (JPEGs et DDS)."""
        group_tex = QGroupBox(_("group_tex_dir"))
        layout_tex = QVBoxLayout()

        # Dossier des DDS
        layout_tex.addWidget(QLabel(_("lbl_dir_dds")))
        layout_dds_input = QHBoxLayout()
        self.dds_dir_input = QLineEdit()
        self.dds_dir_input.setPlaceholderText(_("txt_path_to_the_dds"))
        self.dds_dir_input.setReadOnly(True)
        self.dds_dir_input.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa; padding: 4px; border: 1px solid #555;")
        btn_browse_dds = QPushButton(_("btn_browse"))
        btn_browse_dds.clicked.connect(self.browse_dds_dir)
        layout_dds_input.addWidget(self.dds_dir_input)
        layout_dds_input.addWidget(btn_browse_dds)
        layout_tex.addLayout(layout_dds_input)

        # Dossier des JPEGs
        layout_tex.addWidget(QLabel(_("lbl_dir_jpg")))
        layout_jpeg_input = QHBoxLayout()
        self.tex_dir_input = QLineEdit()
        self.tex_dir_input.setPlaceholderText(_("ph_photos_dir"))
        self.tex_dir_input.setReadOnly(True)
        self.tex_dir_input.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa; padding: 4px; border: 1px solid #555;")
        btn_browse_tex = QPushButton(_("btn_browse"))
        btn_browse_tex.clicked.connect(self.browse_texture_dir)
        layout_jpeg_input.addWidget(self.tex_dir_input)
        layout_jpeg_input.addWidget(btn_browse_tex)
        layout_tex.addLayout(layout_jpeg_input)

        group_tex.setLayout(layout_tex)
        parent_layout.addWidget(group_tex)

    def setup_texture_actions_ui(self, parent_layout):
        """Configure les boutons d'affichage des textures."""
        group_global = QGroupBox(_("group_global_tex"))
        group_local = QGroupBox(_("group_local_tex"))
        layout_global = QHBoxLayout()
        layout_local = QHBoxLayout()

        # Boutons texture globale
        self.btn_toggle_global_tex = QPushButton(_("btn_show_overall_texture"))
        self.btn_toggle_global_tex.setCheckable(True)
        self.btn_toggle_global_tex.setEnabled(True)
        self.btn_toggle_global_tex.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; padding: 6px;")
        self.btn_toggle_global_tex.clicked.connect(self.toggle_global_texture_display)
        self.btn_generate_mosaic = QPushButton(_("btn_generate_global_mosaic"))
        self.btn_generate_mosaic.setStyleSheet("color: white; font-weight: bold; padding: 6px;")
        self.btn_generate_mosaic.clicked.connect(self.generate_global_texture)
        layout_global.addWidget(self.btn_toggle_global_tex)
        layout_global.addWidget(self.btn_generate_mosaic)
        group_global.setLayout(layout_global)

        # Boutons texture HD
        self.btn_add_texture = QPushButton(_("btn_add_tex"))
        self.btn_add_texture.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; padding: 6px;")
        self.btn_add_texture.clicked.connect(self.toggle_texture)
        layout_local.addWidget(self.btn_add_texture)
        self.btn_clear_textures = QPushButton(_("btn_clear_tex"))
        self.btn_clear_textures.setStyleSheet("color: white; font-weight: bold; padding: 6px;")
        self.btn_clear_textures.clicked.connect(self.clear_textures)
        layout_local.addWidget(self.btn_clear_textures)
        group_local.setLayout(layout_local)

        parent_layout.addWidget(group_global)
        parent_layout.addWidget(group_local)

    def setup_filter_ui(self, parent_layout):
        """Configure les filtres d'affichage (Altitude et Aéroports)."""

        group_filter = QGroupBox(_("group_filters"))
        layout_filter = QVBoxLayout()

        # ==========================================
        # POINTS D'INTÉRÊT
        # ==========================================
        # Case à cocher pour les aéroports
        self.cb_show_airports = QCheckBox(_("chkbox_airports"))
        self.cb_show_airports.setStyleSheet("font-weight: bold; color: #b39ddb; margin-top: 8px; margin-bottom: 8px;")
        self.cb_show_airports.stateChanged.connect(self.toggle_airports)
        layout_filter.addWidget(self.cb_show_airports)

        # Case à cocher pour les lacs
        self.cb_show_lakes = QCheckBox(_("chkbox_water"))
        self.cb_show_lakes.setStyleSheet("font-weight: bold; color: #3498db; margin-bottom: 8px;")
        self.cb_show_lakes.stateChanged.connect(self.toggle_lakes)
        layout_filter.addWidget(self.cb_show_lakes)

        # Case à cocher pour les routes
        self.cb_show_roads = QCheckBox(_("chkbox_roads"))
        self.cb_show_roads.setStyleSheet("font-weight: bold; color: #ff69b4; margin-bottom: 8px;")
        self.cb_show_roads.stateChanged.connect(self.toggle_roads)
        layout_filter.addWidget(self.cb_show_roads)

        # Case à cocher pour les taxiways
        self.cb_show_taxi = QCheckBox(_("chkbox_taxi"))
        self.cb_show_taxi.setStyleSheet("font-weight: bold; color: #ffff00; margin-bottom: 8px;")
        self.cb_show_taxi.stateChanged.connect(self.toggle_taxi)
        layout_filter.addWidget(self.cb_show_taxi)

        # Case à cocher pour les autres infrastructures
        self.cb_show_infra = QCheckBox(_("chkbox_infra"))
        self.cb_show_infra.setStyleSheet("font-weight: bold; color: #00ff00; margin-bottom: 8px;")
        self.cb_show_infra.stateChanged.connect(self.toggle_infra)
        layout_filter.addWidget(self.cb_show_infra)

        # Case à cocher pour les croisements de type
        self.cb_show_cross = QCheckBox(_("chkbox_cross"))
        self.cb_show_cross.setStyleSheet("font-weight: bold; color: #ff6969; margin-bottom: 8px;")
        self.cb_show_cross.stateChanged.connect(self.toggle_cross)
        layout_filter.addWidget(self.cb_show_cross)

        # Application du layout au groupe, puis ajout au panneau parent
        group_filter.setLayout(layout_filter)
        parent_layout.addWidget(group_filter)

    def setup_topo_ui(self, parent_layout):
        """Outils de modification de la structure : Subdivision et Lissage."""
        # 1 : SÉLECTION CYLINDRIQUE ---
        self.group_cylinder = CollapsibleBox(_("group_vert_sel"))
        layout_cylinder = QVBoxLayout()

        # Bouton principal (Toggle)
        self.btn_toggle_cylinder = QPushButton(_("btn_activate_sel"))
        self.btn_toggle_cylinder.setCheckable(True)
        self.btn_toggle_cylinder.setStyleSheet("background-color: #188034; color: white; font-weight: bold; padding: 6px;")
        self.btn_toggle_cylinder.clicked.connect(self.toggle_cylinder_tool)
        layout_cylinder.addWidget(self.btn_toggle_cylinder)

        self.pivot_alt_input = QLineEdit()
        self.pivot_alt_input.setReadOnly(True)
        self.pivot_alt_input.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa; padding: 4px; border: 1px solid #555;")

        self.radius_input = QLineEdit(str(self.CYL_RADIUS))

        self.cylinder_radius_slider = QSlider(Qt.Horizontal)
        self.cylinder_radius_slider.setRange(0, self.CYL_MAX_RADIUS)
        self.cylinder_radius_slider.setValue(self.CYL_RADIUS)
        self.cylinder_radius_slider.setEnabled(False)
        self.cylinder_radius_slider.valueChanged.connect(self.on_cylinder_radius_slider_changed)

        self.base_alt_input = QLineEdit()

        self.cylinder_z_slider = QSlider(Qt.Horizontal)
        self.cylinder_z_slider.setRange(0, self.CYL_MAX_BASE)
        self.cylinder_z_slider.setValue(self.CYL_BASE)
        self.cylinder_z_slider.setEnabled(False)
        self.cylinder_z_slider.valueChanged.connect(self.on_cylinder_z_slider_changed)

        grid_cyl = QGridLayout()
        grid_cyl.addWidget(QLabel(_("lbl_piv_alt")), 0, 0)
        grid_cyl.addWidget(self.pivot_alt_input, 0, 1)
        grid_cyl.addWidget(QLabel(_("lbl_ray")), 1, 0)
        grid_cyl.addWidget(self.radius_input, 1, 1)
        grid_cyl.addWidget(self.cylinder_radius_slider, 2, 0, 1, 2)
        grid_cyl.addWidget(QLabel(_("lbl_base_alt")), 3, 0)
        grid_cyl.addWidget(self.base_alt_input, 3, 1)
        grid_cyl.addWidget(self.cylinder_z_slider, 4, 0, 1, 2)
        layout_cylinder.addLayout(grid_cyl)

        btn_layout_cyl = QHBoxLayout()
        self.btn_apply_sel = QPushButton(_("btn_apply"))
        self.btn_apply_sel.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_apply_sel.setEnabled(False)
        self.btn_apply_sel.clicked.connect(self.apply_selection)

        self.btn_cancel_cyl = QPushButton(_("btn_cancel"))
        self.btn_cancel_cyl.setStyleSheet("color: white; font-weight: bold;")
        self.btn_cancel_cyl.setEnabled(False)
        self.btn_cancel_cyl.clicked.connect(self.cancel_cylinder)

        btn_layout_cyl.addWidget(self.btn_apply_sel)
        btn_layout_cyl.addWidget(self.btn_cancel_cyl)
        layout_cylinder.addLayout(btn_layout_cyl)

        self.radius_input.textEdited.connect(self.on_cylinder_text_edited)
        self.base_alt_input.textEdited.connect(self.on_cylinder_text_edited)

        self.group_cylinder.setContentLayout(layout_cylinder)
        parent_layout.addWidget(self.group_cylinder)

        # 2 : MODIFICATIONS TOPOLOGIQUES & GENERATION DE RELIEF ---
        group_topo = QGroupBox(_("group_topo_mod"))
        layout_topo = QVBoxLayout()

        # --- NOUVEAU MENU DES PRESETS ET SLIDERS ---
        # Liste déroulante des Préréglages
        layout_preset = QHBoxLayout()
        lbl_preset = QLabel(_("lbl_preset"))
        lbl_preset.setFixedWidth(100)
        self.combo_presets = QComboBox()
        self.combo_presets.currentTextChanged.connect(self.on_preset_changed)
        layout_preset.addWidget(lbl_preset)
        layout_preset.addWidget(self.combo_presets)
        layout_topo.addLayout(layout_preset)

        # Paramètre de lissage global
        layout_smooth = QHBoxLayout()
        lbl_smooth = QLabel(_("lbl_smooth"))
        lbl_smooth.setFixedWidth(100)
        self.slider_smooth = QSlider(Qt.Horizontal)
        self.slider_smooth.setRange(self.SMOOTH_LOOPS_MIN, self.SMOOTH_LOOPS_MAX)
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(self.SMOOTH_LOOPS_MIN, self.SMOOTH_LOOPS_MAX)
        self.spin_smooth.setFixedWidth(50)
        self.slider_smooth.valueChanged.connect(self.spin_smooth.setValue)
        self.spin_smooth.valueChanged.connect(self.slider_smooth.setValue)
        self.slider_smooth.valueChanged.connect(self.on_user_changed_slider)
        layout_smooth.addWidget(lbl_smooth)
        layout_smooth.addWidget(self.slider_smooth)
        layout_smooth.addWidget(self.spin_smooth)
        layout_topo.addLayout(layout_smooth)

        # Groupe fractal (Chaos) cochable (Active/Désactive automatiquement ses enfants)
        self.group_chaos_params = QGroupBox(_("group_chaos_params"))
        # self.group_chaos_params.setTitle(_("group_chaos_params"))
        self.group_chaos_params.setCheckable(True)
        self.group_chaos_params.toggled.connect(self.on_chaos_group_toggled)
        layout_chaos_v = QVBoxLayout()

        # Slider Précision (Subdivisions)
        layout_prec = QHBoxLayout()
        lbl_prec = QLabel(_("lbl_precision"))
        lbl_prec.setFixedWidth(90)
        self.slider_precision = QSlider(Qt.Horizontal)
        self.slider_precision.setRange(self.FBM_PRECISION_MIN, self.FBM_PRECISION_MAX)
        self.spin_precision = QSpinBox()
        self.spin_precision.setRange(self.FBM_PRECISION_MIN, self.FBM_PRECISION_MAX)
        self.spin_precision.setFixedWidth(50)
        self.slider_precision.valueChanged.connect(self.spin_precision.setValue)
        self.spin_precision.valueChanged.connect(self.slider_precision.setValue)
        self.slider_precision.valueChanged.connect(self.on_user_changed_slider)
        layout_prec.addWidget(lbl_prec)
        layout_prec.addWidget(self.slider_precision)
        layout_prec.addWidget(self.spin_precision)
        layout_chaos_v.addLayout(layout_prec)

        # Slider Octaves
        layout_oct = QHBoxLayout()
        lbl_oct = QLabel(_("lbl_oct"))
        lbl_oct.setFixedWidth(90)
        self.slider_octaves = QSlider(Qt.Horizontal)
        self.slider_octaves.setRange(self.FBM_OCTAVES_MIN, self.FBM_OCTAVES_MAX)
        self.spin_octaves = QSpinBox()
        self.spin_octaves.setRange(self.FBM_OCTAVES_MIN, self.FBM_OCTAVES_MAX)
        self.spin_octaves.setFixedWidth(50)
        self.slider_octaves.valueChanged.connect(self.spin_octaves.setValue)
        self.spin_octaves.valueChanged.connect(self.slider_octaves.setValue)
        self.slider_octaves.valueChanged.connect(self.on_user_changed_slider)
        layout_oct.addWidget(lbl_oct)
        layout_oct.addWidget(self.slider_octaves)
        layout_oct.addWidget(self.spin_octaves)
        layout_chaos_v.addLayout(layout_oct)

        # Slider Amplitude
        layout_amp = QHBoxLayout()
        lbl_amp = QLabel(_("lbl_amp"))
        lbl_amp.setFixedWidth(90)
        self.slider_amplitude = QSlider(Qt.Horizontal)
        self.slider_amplitude.setRange(self.FBM_AMPLITUDE_MIN, self.FBM_AMPLITUDE_MAX)
        self.spin_amplitude = QSpinBox()
        self.spin_amplitude.setRange(self.FBM_AMPLITUDE_MIN, self.FBM_AMPLITUDE_MAX)
        self.spin_amplitude.setFixedWidth(50)
        self.slider_amplitude.valueChanged.connect(self.spin_amplitude.setValue)
        self.spin_amplitude.valueChanged.connect(self.slider_amplitude.setValue)
        self.slider_amplitude.valueChanged.connect(self.on_user_changed_slider)
        layout_amp.addWidget(lbl_amp)
        layout_amp.addWidget(self.slider_amplitude)
        layout_amp.addWidget(self.spin_amplitude)
        layout_chaos_v.addLayout(layout_amp)

        # Slider Scale (Échelle)
        layout_scale = QHBoxLayout()
        lbl_scale = QLabel(_("lbl_scale"))
        lbl_scale.setFixedWidth(90)
        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setRange(self.FBM_SCALE_MIN, self.FBM_SCALE_MAX)
        self.spin_scale = QSpinBox()
        self.spin_scale.setRange(self.FBM_SCALE_MIN, self.FBM_SCALE_MAX)
        self.spin_scale.setFixedWidth(50)
        self.slider_scale.valueChanged.connect(self.spin_scale.setValue)
        self.spin_scale.valueChanged.connect(self.slider_scale.setValue)
        self.slider_scale.valueChanged.connect(self.on_user_changed_slider)
        layout_scale.addWidget(lbl_scale)
        layout_scale.addWidget(self.slider_scale)
        layout_scale.addWidget(self.spin_scale)
        layout_chaos_v.addLayout(layout_scale)

        self.group_chaos_params.setLayout(layout_chaos_v)
        layout_topo.addWidget(self.group_chaos_params)

        # Sauvegarder le Preset
        layout_topo_btns = QHBoxLayout()
        self.btn_save_preset = QPushButton(_("btn_save_runway"))
        self.btn_save_preset.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.btn_save_preset.setEnabled(False) # Grisé par défaut, s'active en mode Custom
        self.btn_save_preset.clicked.connect(self.save_custom_relief_preset)
        layout_topo_btns.addWidget(self.btn_save_preset)

        # Apply
        self.btn_all_in_one = QPushButton(_("btn_apply"))
        self.btn_all_in_one.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_all_in_one.clicked.connect(lambda checked=False: self.apply_all_in_one(undo_first=False))
        layout_topo_btns.addWidget(self.btn_all_in_one)

        # Undo + apply
        self.btn_all_in_one_2 = QPushButton(_("btn_all_in_one"))
        self.btn_all_in_one_2.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold;")
        self.btn_all_in_one_2.clicked.connect(lambda checked=False: self.apply_all_in_one(undo_first=True))
        layout_topo_btns.addWidget(self.btn_all_in_one_2)
        layout_topo.addLayout(layout_topo_btns)

        group_topo.setLayout(layout_topo)
        parent_layout.addWidget(group_topo)

    def setup_point_ui(self, parent_layout):
        """Configure les outils d'édition manuelle des sommets avec un Proxy visuel."""
        self.group_point_edit = CollapsibleBox(_("group_point_edit"))
        layout_point = QVBoxLayout()
        grid_point = QGridLayout()

        # Ligne 1 : Créer / Éditer
        self.btn_create_point = QPushButton(_("btn_create_pt"))
        self.btn_create_point.setStyleSheet("background-color: #188034; color: white; font-weight: bold;")
        self.btn_create_point.clicked.connect(self.create_point)

        self.btn_edit_point = QPushButton(_("btn_edit_pt"))
        self.btn_edit_point.setStyleSheet("background-color: #188034; color: white; font-weight: bold;")
        self.btn_edit_point.clicked.connect(self.edit_point)

        grid_point.addWidget(self.btn_create_point, 0, 0)
        grid_point.addWidget(self.btn_edit_point, 0, 1)

        # Ligne 2 : Contrôle de l'altitude (Slider + Spinbox)
        layout_z = QHBoxLayout()
        layout_z.addWidget(QLabel(_("lbl_target_z")))

        self.point_slider = QSlider(Qt.Horizontal)
        self.point_slider.setRange(self.POINT_RANGE_MIN, self.POINT_RANGE_MAX)
        self.point_slider.setEnabled(False)
        self.point_slider.valueChanged.connect(self.on_point_slider_changed)

        self.point_spinbox = QDoubleSpinBox()
        self.point_spinbox.setRange(-2000.0, 10000.0)
        self.point_spinbox.setDecimals(1)
        self.point_spinbox.setSuffix(" m")
        self.point_spinbox.setEnabled(False)
        self.point_spinbox.valueChanged.connect(self.on_point_spinbox_changed)

        layout_z.addWidget(self.point_slider)
        layout_z.addWidget(self.point_spinbox)

        # Ajout du layout Z sur toute la largeur de la grille (colspan=2)
        grid_point.addLayout(layout_z, 1, 0, 1, 2)

        # Ligne 3 : Validation / Annulation
        self.btn_validate_point = QPushButton(_("btn_apply"))
        self.btn_validate_point.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_validate_point.clicked.connect(self.validate_point)
        self.btn_validate_point.setEnabled(False)

        self.btn_cancel_point = QPushButton(_("btn_cancel"))
        self.btn_cancel_point.setStyleSheet("color: white; font-weight: bold;")
        self.btn_cancel_point.clicked.connect(self.cancel_point)
        self.btn_cancel_point.setEnabled(False)

        grid_point.addWidget(self.btn_validate_point, 2, 0)
        grid_point.addWidget(self.btn_cancel_point, 2, 1)

        layout_point.addLayout(grid_point)
        self.group_point_edit.setContentLayout(layout_point)
        parent_layout.addWidget(self.group_point_edit)

    def setup_zone_ui(self, parent_layout):
        """Configure l'outil unifié : Tracé polygonal (Étape 1) et Aplanissement (Étape 2)."""
        group_zone = QGroupBox(_("group_complex_area"))
        layout_zone = QVBoxLayout()

        # Nom polygone
        lay_name = QHBoxLayout()
        lay_name.addWidget(QLabel(_("lbl_current_name")))
        self.current_zone_name_input = QLineEdit()
        self.current_zone_name_input.setReadOnly(True)
        self.current_zone_name_input.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa; padding: 4px; border: 1px solid #555;")
        lay_name.addWidget(self.current_zone_name_input)
        layout_zone.addLayout(lay_name)

        # --- ÉTAPE 1 : TRACÉ ---
        self.btn_toggle_zone = QPushButton(_("btn_area_trace"))
        self.btn_toggle_zone.setCheckable(True)
        self.btn_toggle_zone.setStyleSheet("background-color: #188034; color: white; font-weight: bold; padding: 6px;")
        self.btn_toggle_zone.clicked.connect(self.toggle_zone_mode)
        layout_zone.addWidget(self.btn_toggle_zone)

        layout_talus = QHBoxLayout()
        layout_talus.addWidget(QLabel(_("lbl_bank_size")))
        self.flatten_trans_input = QLineEdit(str(self.FLATTEN_BANK))
        self.flatten_trans_input.setEnabled(False)
        layout_talus.addWidget(self.flatten_trans_input)
        layout_zone.addLayout(layout_talus)

        # Paramètre de lissage global
        layout_area_subdiv = QHBoxLayout()
        lbl_area_subdiv = QLabel(_("lbl_bank_precision"))
        lbl_area_subdiv.setFixedWidth(100)
        self.slider_area_subdiv = QSlider(Qt.Horizontal)
        self.slider_area_subdiv.setRange(1, self.MAX_SUBDIV_LEVEL)
        self.spin_area_subdiv = QSpinBox()
        self.spin_area_subdiv.setRange(1, self.MAX_SUBDIV_LEVEL)
        self.spin_area_subdiv.setFixedWidth(50)
        self.slider_area_subdiv.valueChanged.connect(self.spin_area_subdiv.setValue)
        self.spin_area_subdiv.valueChanged.connect(self.slider_area_subdiv.setValue)
        self.slider_area_subdiv.setValue(self.ZONE_SUBDIV)
        layout_area_subdiv.addWidget(lbl_area_subdiv)
        layout_area_subdiv.addWidget(self.slider_area_subdiv)
        layout_area_subdiv.addWidget(self.spin_area_subdiv)
        layout_zone.addLayout(layout_area_subdiv)

        # 0.28.0 --- SOUS-GROUPE : PLAN INCLINÉ ---
        self.group_tilted_plane = QGroupBox(_("group_tilted_plane"))
        self.group_tilted_plane.setCheckable(True)
        self.group_tilted_plane.setChecked(False)
        self.group_tilted_plane.toggled.connect(self.on_tilted_plane_toggled)
        layout_tilted = QVBoxLayout()

        self.btn_def_ab = QPushButton(_("btn_def_ab"))
        self.btn_def_ab.setCheckable(True)
        self.btn_def_ab.setStyleSheet("background-color: #188034; color: white; font-weight: bold;")
        self.btn_def_ab.clicked.connect(self.toggle_define_ab_mode)
        layout_tilted.addWidget(self.btn_def_ab)

        # Contrôles Point A
        grid_A = QGridLayout()
        grid_A.addWidget(QLabel(_("lbl_alt_A")), 0, 0)
        self.slider_z_a = QSlider(Qt.Horizontal)
        self.slider_z_a.setRange(-1000, 1000)
        self.spin_z_a = QDoubleSpinBox()
        self.spin_z_a.setRange(-2000.0, 10000.0)
        self.spin_z_a.setSuffix(" m")
        grid_A.addWidget(self.slider_z_a, 0, 1)
        grid_A.addWidget(self.spin_z_a, 0, 2)

        grid_A.addWidget(QLabel(_("lbl_rot_A")), 1, 0)
        self.slider_rot_a = QSlider(Qt.Horizontal)
        self.slider_rot_a.setRange(-450, 450)
        grid_A.addWidget(self.slider_rot_a, 1, 1, 1, 2)
        layout_tilted.addLayout(grid_A)

        # Contrôles Point B
        grid_B = QGridLayout()
        grid_B.addWidget(QLabel(_("lbl_alt_B")), 0, 0)
        self.slider_z_b = QSlider(Qt.Horizontal)
        self.slider_z_b.setRange(-1000, 1000)
        self.spin_z_b = QDoubleSpinBox()
        self.spin_z_b.setRange(-2000.0, 10000.0)
        self.spin_z_b.setSuffix(" m")
        grid_B.addWidget(self.slider_z_b, 0, 1)
        grid_B.addWidget(self.spin_z_b, 0, 2)

        grid_B.addWidget(QLabel(_("lbl_rot_B")), 1, 0)
        self.slider_rot_b = QSlider(Qt.Horizontal)
        self.slider_rot_b.setRange(-450, 450)
        grid_B.addWidget(self.slider_rot_b, 1, 1, 1, 2)
        layout_tilted.addLayout(grid_B)

        self.group_tilted_plane.setLayout(layout_tilted)
        layout_zone.addWidget(self.group_tilted_plane)
        self.group_tilted_plane.setEnabled(False)

        # Connexions des signaux
        self.slider_z_a.valueChanged.connect(self.on_tilted_z_changed)
        self.spin_z_a.valueChanged.connect(self.on_tilted_spin_changed)
        self.slider_z_b.valueChanged.connect(self.on_tilted_z_changed)
        self.spin_z_b.valueChanged.connect(self.on_tilted_spin_changed)

        self.slider_rot_a.sliderPressed.connect(self.on_tilted_rot_pressed)
        self.slider_rot_a.valueChanged.connect(lambda v: self.on_tilted_rot_changed(v, 'A'))
        self.slider_rot_a.sliderReleased.connect(self.on_tilted_rot_released)

        self.slider_rot_b.sliderPressed.connect(self.on_tilted_rot_pressed)
        self.slider_rot_b.valueChanged.connect(lambda v: self.on_tilted_rot_changed(v, 'B'))
        self.slider_rot_b.sliderReleased.connect(self.on_tilted_rot_released)

        self.tilted_pts_2d = []

        layout_io_btns = QHBoxLayout()
        self.btn_load_zone = QPushButton(_("btn_load_runway"))
        self.btn_load_zone.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_load_zone.clicked.connect(self.load_flat_project)
        self.btn_save_zone = QPushButton(_("btn_save_runway"))
        self.btn_save_zone.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.btn_save_zone.setEnabled(False)
        self.btn_save_zone.clicked.connect(self.save_flat_project)
        layout_io_btns.addWidget(self.btn_load_zone)
        layout_io_btns.addWidget(self.btn_save_zone)
        layout_zone.addLayout(layout_io_btns)

        btn_layout_step1 = QHBoxLayout()
        self.btn_apply_zone = QPushButton(_("btn_validate_area"))
        self.btn_apply_zone.setStyleSheet("background-color: #34495e; color: white; font-weight: bold")
        self.btn_apply_zone.setEnabled(False)
        self.btn_apply_zone.clicked.connect(self.apply_zone_cut_ui)

        self.btn_cancel_zone = QPushButton(_("btn_clear_polygon"))
        self.btn_cancel_zone.setStyleSheet("color: white; font-weight: bold")
        self.btn_cancel_zone.setEnabled(False)
        self.btn_cancel_zone.clicked.connect(self.cancel_zone)

        btn_layout_step1.addWidget(self.btn_apply_zone)
        btn_layout_step1.addWidget(self.btn_cancel_zone)
        layout_zone.addLayout(btn_layout_step1)

        # 0.28.1 - autoriser destruction tous types de triangles
        self.cb_allow_destroy_all_types = QCheckBox(_("chk_allow_destr"))
        layout_zone.addWidget(self.cb_allow_destroy_all_types)

        group_zone.setLayout(layout_zone)
        parent_layout.addWidget(group_zone)

        # --- ÉTAPE 2 : APLANISSEMENT ---
        group_flat = QGroupBox(_("group_flatten"))
        layout_flat = QVBoxLayout()

        layout_z = QHBoxLayout()
        layout_z.addWidget(QLabel(_("lbl_target_z")))

        self.flatten_slider = QSlider(Qt.Horizontal)
        self.flatten_slider.setRange(0, 1000)
        self.flatten_slider.setValue(500)
        self.flatten_slider.setEnabled(False)
        self.flatten_slider.valueChanged.connect(self.on_flatten_slider_changed)

        self.flatten_spinbox = QDoubleSpinBox()
        self.flatten_spinbox.setRange(-2000.0, 10000.0)
        self.flatten_spinbox.setDecimals(1)
        self.flatten_spinbox.setSuffix(" m")
        self.flatten_spinbox.setEnabled(False)
        self.flatten_spinbox.valueChanged.connect(self.on_flatten_spinbox_changed)

        layout_z.addWidget(self.flatten_slider)
        layout_z.addWidget(self.flatten_spinbox)
        layout_flat.addLayout(layout_z)

        btn_layout_step2 = QHBoxLayout()
        self.btn_apply_flatten = QPushButton(_("btn_apply_flattening"))
        self.btn_apply_flatten.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_apply_flatten.setEnabled(False)
        self.btn_apply_flatten.clicked.connect(self.apply_flatten)

        self.btn_cancel_flatten = QPushButton(_("btn_cancel_flattering"))
        self.btn_cancel_flatten.setStyleSheet("color: white; font-weight: bold;")
        self.btn_cancel_flatten.setEnabled(False)
        self.btn_cancel_flatten.clicked.connect(self.abort_flattening)

        btn_layout_step2.addWidget(self.btn_apply_flatten)
        btn_layout_step2.addWidget(self.btn_cancel_flatten)
        layout_flat.addLayout(btn_layout_step2)

        group_flat.setLayout(layout_flat)
        parent_layout.addWidget(group_flat)

    def setup_runway_ui(self, parent_layout):
        group_runway = QGroupBox(_("group_slope_runway"))
        layout = QVBoxLayout()

        # Nom piste
        lay_name = QHBoxLayout()
        lay_name.addWidget(QLabel(_("lbl_current_name")))
        self.current_runway_name_input = QLineEdit()
        self.current_runway_name_input.setReadOnly(True)
        self.current_runway_name_input.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa; padding: 4px; border: 1px solid #555;")
        lay_name.addWidget(self.current_runway_name_input)
        layout.addLayout(lay_name)

        # Activation
        self.btn_toggle_runway = QPushButton(_("btn_def_axis"))
        self.btn_toggle_runway.setCheckable(True)
        self.btn_toggle_runway.setStyleSheet("background-color: #188034; color: white; font-weight: bold; padding: 6px;")
        self.btn_toggle_runway.clicked.connect(self.toggle_runway_mode)
        layout.addWidget(self.btn_toggle_runway)

        # Bouton bascule de prévisualisation (Squelette vs Surface)
        self.btn_preview_mode = QPushButton(_("txt_line_mode"))
        self.btn_preview_mode.setCheckable(True)
        self.btn_preview_mode.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_preview_mode.clicked.connect(self.toggle_preview_mode)
        self.btn_preview_mode.setEnabled(False)
        layout.addWidget(self.btn_preview_mode)

        # Paramètres géométriques
        grid_params = QGridLayout()
        self.runway_width_input = QLineEdit(str(self.RUNWAY_WIDTH))
        self.runway_width_input.textEdited.connect(self.on_runway_width_text_edited)

        self.runway_width_slider = QSlider(Qt.Horizontal)
        self.runway_width_slider.setRange(10, self.RUNWAY_MAX_WIDTH)
        self.runway_width_slider.setValue(self.RUNWAY_WIDTH)
        self.runway_width_slider.setEnabled(False)
        self.runway_width_slider.valueChanged.connect(self.on_runway_width_changed)

        # Remplacement de l'ancien bloc talus
        lay_talus = QHBoxLayout()
        self.runway_trans_side_input = QLineEdit(str(self.RUNWAY_BANK_SIDE))

        grid_params.addWidget(QLabel(_("lbl_runway_width")), 0, 0)
        grid_params.addWidget(self.runway_width_input, 0, 1)
        grid_params.addWidget(self.runway_width_slider, 1, 0, 1, 2)

        # Organisation côte à côte
        grid_params.addWidget(QLabel(_("lbl_bank_size")), 2, 0)
        lay_talus.addWidget(self.runway_trans_side_input)
        grid_params.addLayout(lay_talus, 2, 1)

        layout.addLayout(grid_params)

        # Paramètre de lissage global
        layout_runway_subdiv = QHBoxLayout()
        lbl_runway_subdiv = QLabel(_("lbl_bank_precision"))
        lbl_runway_subdiv.setFixedWidth(100)
        self.slider_runway_subdiv = QSlider(Qt.Horizontal)
        self.slider_runway_subdiv.setRange(1, self.MAX_SUBDIV_LEVEL)
        self.spin_runway_subdiv = QSpinBox()
        self.spin_runway_subdiv.setRange(1, self.MAX_SUBDIV_LEVEL)
        self.spin_runway_subdiv.setFixedWidth(50)
        self.slider_runway_subdiv.valueChanged.connect(self.spin_runway_subdiv.setValue)
        self.spin_runway_subdiv.valueChanged.connect(self.slider_runway_subdiv.setValue)
        self.slider_runway_subdiv.setValue(self.RUNWAY_SUBDIV)
        layout_runway_subdiv.addWidget(lbl_runway_subdiv)
        layout_runway_subdiv.addWidget(self.slider_runway_subdiv)
        layout_runway_subdiv.addWidget(self.spin_runway_subdiv)
        layout.addLayout(layout_runway_subdiv)

        # Points de contrôle (Tableau)
        self.runway_table = QTableWidget(0, 2)
        self.runway_table.setHorizontalHeaderLabels(["Position (%)", "Altitude (m)"])
        self.runway_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.runway_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.runway_table.setSelectionMode(QTableWidget.SingleSelection)
        # Bloque la saisie manuelle au clavier
        self.runway_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.runway_table.itemSelectionChanged.connect(self.on_runway_table_selection)
        layout.addWidget(self.runway_table)

        # --- Remplacer btn_add_midpoint par deux boutons ---
        layout_add_pts = QHBoxLayout()

        self.btn_add_point_above = QPushButton(_("btn_add_above"))
        self.btn_add_point_above.setEnabled(False)
        self.btn_add_point_above.clicked.connect(self.add_runway_point_above)

        self.btn_add_point_below = QPushButton(_("btn_add_below"))
        self.btn_add_point_below.setEnabled(False)
        self.btn_add_point_below.clicked.connect(self.add_runway_point_below)

        self.btn_delete_point = QPushButton(_("btn_delete_pt"))
        self.btn_delete_point.setEnabled(False)
        self.btn_delete_point.clicked.connect(self.delete_runway_point)

        layout_add_pts.addWidget(self.btn_add_point_above)
        layout_add_pts.addWidget(self.btn_add_point_below)
        layout_add_pts.addWidget(self.btn_delete_point)

        layout.addLayout(layout_add_pts)

        # Sliders d'édition du point sélectionné ---
        group_edit = CollapsibleBox(_("group_edit_pt"))
        layout_edit = QVBoxLayout()

        # Slider Position / Échelle
        layout_pos = QHBoxLayout()
        self.lbl_runway_pos = QLabel(_("lbl_pos_pct")) # Devenu une variable pour changer le texte
        layout_pos.addWidget(self.lbl_runway_pos)
        self.runway_pos_slider = QSlider(Qt.Horizontal)
        self.runway_pos_slider.setRange(0, 100)
        self.runway_pos_slider.setEnabled(False)

        # Ajout des 3 signaux pour le mode Scale
        self.runway_pos_slider.sliderPressed.connect(self.on_runway_pos_pressed)
        self.runway_pos_slider.valueChanged.connect(self.on_runway_pos_slider_changed)
        self.runway_pos_slider.sliderReleased.connect(self.on_runway_pos_released)

        layout_pos.addWidget(self.runway_pos_slider)
        layout_edit.addLayout(layout_pos)

        # Slider Altitude (Z)
        layout_z = QHBoxLayout()
        layout_z.addWidget(QLabel(_("lbl_alt_z")))
        self.runway_z_slider = QSlider(Qt.Horizontal)
        self.runway_z_slider.setRange(-1000, 1000)
        self.runway_z_slider.setEnabled(False)
        self.runway_z_slider.valueChanged.connect(self.on_runway_z_slider_changed)
        layout_z.addWidget(self.runway_z_slider)
        layout_edit.addLayout(layout_z)

        # Slider Altitude Globale (Z) ---
        layout_global_z = QHBoxLayout()
        layout_global_z.addWidget(QLabel(_("lbl_global_z")))
        self.runway_global_z_slider = QSlider(Qt.Horizontal)
        self.runway_global_z_slider.setRange(-1000, 1000)
        self.runway_global_z_slider.setEnabled(False)

        # On utilise 3 signaux pour gérer le comportement "infini"
        self.runway_global_z_slider.sliderPressed.connect(self.on_global_z_pressed)
        self.runway_global_z_slider.valueChanged.connect(self.on_global_z_changed)
        self.runway_global_z_slider.sliderReleased.connect(self.on_global_z_released)

        layout_global_z.addWidget(self.runway_global_z_slider)
        layout_edit.addLayout(layout_global_z)

        # Slider Rotation (Azimut) ---
        layout_rot = QHBoxLayout()
        layout_rot.addWidget(QLabel(_("lbl_rotation")))
        self.runway_rot_slider = QSlider(Qt.Horizontal)
        self.runway_rot_slider.setRange(-450, 450) # Représente +/- 45.0 degrés
        self.runway_rot_slider.setEnabled(False)

        self.runway_rot_slider.sliderPressed.connect(self.on_runway_rot_pressed)
        self.runway_rot_slider.valueChanged.connect(self.on_runway_rot_changed)
        self.runway_rot_slider.sliderReleased.connect(self.on_runway_rot_released)

        layout_rot.addWidget(self.runway_rot_slider)
        layout_edit.addLayout(layout_rot)

        group_edit.setContentLayout(layout_edit)
        layout.addWidget(group_edit)

        # Bibliothèque de Pistes (Save/Load) ---
        layout_io = QHBoxLayout()
        self.btn_load_runway = QPushButton(_("btn_load_runway"))
        self.btn_load_runway.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_load_runway.clicked.connect(self.load_runway_project)
        self.btn_save_runway = QPushButton(_("btn_save_runway"))
        self.btn_save_runway.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.btn_save_runway.clicked.connect(self.save_runway_project)
        self.btn_save_runway.setEnabled(False)
        layout_io.addWidget(self.btn_load_runway)
        layout_io.addWidget(self.btn_save_runway)
        layout.addLayout(layout_io)

        # Actions
        btn_layout = QHBoxLayout()
        self.btn_apply_runway = QPushButton(_("btn_apply"))
        self.btn_apply_runway.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_apply_runway.setEnabled(False)
        self.btn_apply_runway.clicked.connect(self.apply_runway)
        self.btn_cancel_runway = QPushButton(_("btn_cancel"))
        self.btn_cancel_runway.setStyleSheet("color: white; font-weight: bold;")
        self.btn_cancel_runway.setEnabled(False)
        self.btn_cancel_runway.clicked.connect(self.abort_runway)
        btn_layout.addWidget(self.btn_apply_runway)
        btn_layout.addWidget(self.btn_cancel_runway)

        layout.addLayout(btn_layout)
        group_runway.setLayout(layout)
        parent_layout.addWidget(group_runway)

    def setup_retouch_ui(self, parent_layout):
        """Configure les outils de retouche colorimétrique 2D."""

        # ==========================================
        # BOUTONS GLOBAUX (En haut)
        # ==========================================
        self.btn_toggle_2d = QPushButton(_("btn_enable_2d_retouch_mode"))
        self.btn_toggle_2d.setStyleSheet("background-color: #188034; color: white; font-weight: bold; padding: 6px;")
        self.btn_toggle_2d.setCheckable(True)
        self.btn_toggle_2d.clicked.connect(self.toggle_2d_mode)
        parent_layout.addWidget(self.btn_toggle_2d)

        lay_2d_layers = QHBoxLayout()
        self.btn_toggle_grid_2d = QPushButton(_("btn_show_tile_grid"))
        self.btn_toggle_grid_2d.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_toggle_grid_2d.setCheckable(True)
        self.btn_toggle_grid_2d.setEnabled(False)
        self.btn_toggle_grid_2d.clicked.connect(self.toggle_2d_grid)
        lay_2d_layers.addWidget(self.btn_toggle_grid_2d)

        self.btn_toggle_mesh_mask = QPushButton(_("btn_toggle_mesh_mask"))
        self.btn_toggle_mesh_mask.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_toggle_mesh_mask.setCheckable(True)
        self.btn_toggle_mesh_mask.setEnabled(False)
        self.btn_toggle_mesh_mask.clicked.connect(self.toggle_mesh_mask)
        lay_2d_layers.addWidget(self.btn_toggle_mesh_mask)

        parent_layout.addLayout(lay_2d_layers)

        # ==========================================
        # GROUPE : SÉLECTION
        # ==========================================
        line_sel_1 = QFrame()
        line_sel_1.setFrameShape(QFrame.HLine)
        line_sel_1.setFrameShadow(QFrame.Sunken)
        parent_layout.addWidget(line_sel_1)

        self.group_selection = CollapsibleBox(_("chk_selection"))
        layout_selection = QVBoxLayout()

        # Nom de la sélection en cours
        lay_name = QHBoxLayout()
        lay_name.addWidget(QLabel(_("lbl_current_name")))
        self.current_sel_name_input = QLineEdit()
        self.current_sel_name_input.setReadOnly(True)
        self.current_sel_name_input.setPlaceholderText(_("txt_no_named_selection"))
        self.current_sel_name_input.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa; padding: 4px; border: 1px solid #555;")
        lay_name.addWidget(self.current_sel_name_input)
        layout_selection.addLayout(lay_name)

        # --- Bibliothèque des Sélections ---
        layout_io_btns = QHBoxLayout()
        self.btn_load_sel_2d = QPushButton(_("btn_load_runway"))
        self.btn_load_sel_2d.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_load_sel_2d.clicked.connect(self.load_selection_project)

        self.btn_save_sel_2d = QPushButton(_("btn_save_runway"))
        self.btn_save_sel_2d.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.btn_save_sel_2d.setEnabled(False)
        self.btn_save_sel_2d.clicked.connect(self.save_selection_project)

        self.btn_crop_sel_2d = QPushButton(_("btn_crop_by_saved_area"))
        self.btn_crop_sel_2d.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;") # e67e22
        self.btn_crop_sel_2d.setEnabled(False)
        self.btn_crop_sel_2d.clicked.connect(self.crop_by_saved_selection)

        layout_io_btns.addWidget(self.btn_load_sel_2d)
        layout_io_btns.addWidget(self.btn_save_sel_2d)
        layout_io_btns.addWidget(self.btn_crop_sel_2d)
        layout_selection.addLayout(layout_io_btns)

        # --- Mode d'affichage de la sélection ---
        self.chk_show_outline = QCheckBox(_("chk_hide_red_fill_retouch_mod"))
        self.chk_show_outline.stateChanged.connect(self.update_selection_visual)
        layout_selection.addWidget(self.chk_show_outline)

        # --- Options du Lasso (Opérations Booléennes) ---
        layout_lasso = QHBoxLayout()
        lbl_lasso = QLabel(_("chk_tracing_mode_lasso"))
        self.lasso_btn_group = QButtonGroup()
        self.rb_add = QRadioButton(_("btn_add"))
        self.rb_sub = QRadioButton(_("btn_subtract"))
        self.rb_add.setChecked(True) # Mode par défaut
        self.lasso_btn_group.addButton(self.rb_add, 1)
        self.lasso_btn_group.addButton(self.rb_sub, 2)
        self.rb_add.toggled.connect(lambda: self.set_lasso_mode("add"))
        self.rb_sub.toggled.connect(lambda: self.set_lasso_mode("sub"))
        layout_lasso.addWidget(lbl_lasso)
        layout_lasso.addWidget(self.rb_add)
        layout_lasso.addWidget(self.rb_sub)
        layout_selection.addLayout(layout_lasso)

        self.group_selection.setContentLayout(layout_selection)
        parent_layout.addWidget(self.group_selection)

        # ==========================================
        # PANNEAU DE COULEURS
        # ==========================================
        self.group_color = CollapsibleBox(_("chk_color_retouching"))
        layout_color = QVBoxLayout()
        layout_color.setContentsMargins(10, 10, 10, 10)

        self.sliders_color = {}
        self.spinboxes_color = {}
        self.retouch_defaults = {}

        sliders_config = {
            "brightness": (_("lbl_brightness"), -100, 100, 0),
            "contrast": (_("lbl_contrast"), -100, 100, 0),
            "temp": (_("lbl_temp"), -100, 100, 0),
            "tint": (_("lbl_tint"), -100, 100, 0),
            "saturation": (_("lbl_saturation"), -100, 100, 0),
            "thresh_b": (_("lbl_thresh_b"), 0, 255, self.RETOUCH_THRESH_B),
            "thresh_w": (_("lbl_thresh_w"), 0, 255, self.RETOUCH_THRESH_W)
        }

        for key, (label_text, v_min, v_max, v_def) in sliders_config.items():
            lay = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(120)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(v_min, v_max)
            slider.setValue(v_def)
            slider.setEnabled(False)

            spinbox = QSpinBox()
            spinbox.setRange(v_min, v_max)
            spinbox.setValue(v_def)
            spinbox.setEnabled(False)
            spinbox.setFixedWidth(60)

            slider.valueChanged.connect(spinbox.setValue)
            spinbox.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(self.update_retouch_preview)

            self.sliders_color[key] = slider
            self.spinboxes_color[key] = spinbox
            self.retouch_defaults[key] = v_def

            lay.addWidget(lbl)
            lay.addWidget(slider)
            lay.addWidget(spinbox)
            layout_color.addLayout(lay)

        # Sauvegarde / Chargement des réglages couleurs
        layout_color_io = QHBoxLayout()
        self.btn_load_colors = QPushButton(_("btn_load_settings"))
        self.btn_load_colors.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_load_colors.setEnabled(False)
        self.btn_load_colors.clicked.connect(self.load_color_preset)

        self.btn_save_colors = QPushButton(_("btn_save_settings"))
        self.btn_save_colors.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.btn_save_colors.setEnabled(False)
        self.btn_save_colors.clicked.connect(self.save_color_preset)

        layout_color_io.addWidget(self.btn_load_colors)
        layout_color_io.addWidget(self.btn_save_colors)
        layout_color.addLayout(layout_color_io)

        # Apply / Cancel des réglages couleurs
        lay_col_btns = QHBoxLayout()
        self.btn_apply_color = QPushButton(_("btn_apply"))
        self.btn_apply_color.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_apply_color.setEnabled(False)
        self.btn_apply_color.clicked.connect(self.apply_color_retouch)

        self.btn_cancel_color = QPushButton(_("btn_cancel"))
        self.btn_cancel_color.setStyleSheet("color: white; font-weight: bold;")
        self.btn_cancel_color.setEnabled(False)
        self.btn_cancel_color.clicked.connect(self.cancel_color_retouch)

        self.btn_reset_local_retouch = QPushButton(_("btn_reset_color"))
        self.btn_reset_local_retouch.setStyleSheet("color: white; font-weight: bold;")
        self.btn_reset_local_retouch.setEnabled(False)
        self.btn_reset_local_retouch.clicked.connect(self.reset_selection_retouch)

        lay_col_btns.addWidget(self.btn_apply_color)
        lay_col_btns.addWidget(self.btn_cancel_color)
        lay_col_btns.addWidget(self.btn_reset_local_retouch)
        layout_color.addLayout(lay_col_btns)

        # self.group_color.setLayout(layout_color)
        self.group_color.setContentLayout(layout_color)
        parent_layout.addWidget(self.group_color)

        # ==========================================
        # BATCH
        # ==========================================
        line_batch = QFrame()
        line_batch.setFrameShape(QFrame.HLine)
        line_batch.setFrameShadow(QFrame.Sunken)
        parent_layout.addWidget(line_batch)

        self.group_batch = CollapsibleBox(_("chk_batch_retouching"))
        layout_batch = QHBoxLayout()

        # Boutons show all - apply all
        self.btn_show_all_sel_2d = QPushButton(_("btn_show_all_sel_2d"))
        self.btn_show_all_sel_2d.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_show_all_sel_2d.clicked.connect(self.show_all_selections_2d)
        layout_batch.addWidget(self.btn_show_all_sel_2d)

        self.btn_apply_all_sel_2d = QPushButton(_("btn_apply_all_sel_2d"))
        self.btn_apply_all_sel_2d.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_apply_all_sel_2d.setEnabled(False)
        self.btn_apply_all_sel_2d.clicked.connect(self.apply_all_color_presets)
        layout_batch.addWidget(self.btn_apply_all_sel_2d)

        self.group_batch.setContentLayout(layout_batch)
        parent_layout.addWidget(self.group_batch)

        # ==========================================
        # GROUPE : Texture globale
        # ==========================================
        line_tex = QFrame()
        line_tex.setFrameShape(QFrame.HLine)
        line_tex.setFrameShadow(QFrame.Sunken)
        parent_layout.addWidget(line_tex)
        # parent_layout.addWidget(QLabel(_("lbl_layer_mode")))

        self.group_global_tex = CollapsibleBox(_("chk_global_tex"))
        layout_global_tex = QVBoxLayout()
        lay_brush_btns = QHBoxLayout()

        # SEAMLESS CLONE
        self.btn_toggle_seamless = QPushButton(_("btn_toggle_seamless"))
        self.btn_toggle_seamless.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_toggle_seamless.setCheckable(True)
        self.btn_toggle_seamless.clicked.connect(self.toggle_seamless_mode)
        lay_brush_btns.addWidget(self.btn_toggle_seamless)

        # Mélangeur de couleurs
        self.btn_toggle_meansh = QPushButton(_("btn_toggle_meansh"))
        self.btn_toggle_meansh.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_toggle_meansh.setCheckable(True)
        self.btn_toggle_meansh.clicked.connect(self.toggle_meansh_mode)
        lay_brush_btns.addWidget(self.btn_toggle_meansh)

        # Slider de taille
        lbl_size = QLabel("O")
        lay_brush_btns.addWidget(lbl_size)
        self.slider_brush_rad = QSlider(Qt.Horizontal)
        self.slider_brush_rad.setRange(self.BRUSH_RADIUS_MIN, self.BRUSH_RADIUS_MAX)
        self.slider_brush_rad.setValue(self.BRUSH_RADIUS_DEF)
        self.slider_brush_rad.valueChanged.connect(self.on_brush_radius_changed)
        lay_brush_btns.addWidget(self.slider_brush_rad)

        layout_global_tex.addLayout(lay_brush_btns)

        # Boutons load / save global texture
        lay_tex_io_btns = QHBoxLayout()

        self.btn_load_global_tex_2d = QPushButton(_("btn_load_runway"))
        self.btn_load_global_tex_2d.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_load_global_tex_2d.clicked.connect(self.load_global_tex_2d_from_imgfile)
        lay_tex_io_btns.addWidget(self.btn_load_global_tex_2d)

        self.btn_save_global_tex_2d = QPushButton(_("btn_save_runway"))
        self.btn_save_global_tex_2d.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.btn_save_global_tex_2d.clicked.connect(self.save_global_tex_2d_to_imgfile)
        lay_tex_io_btns.addWidget(self.btn_save_global_tex_2d)

        layout_global_tex.addLayout(lay_tex_io_btns)
        self.group_global_tex.setContentLayout(layout_global_tex)
        parent_layout.addWidget(self.group_global_tex)

        # ==========================================
        # Boutons du bas - export ortho + undo all
        # ==========================================
        line_end = QFrame()
        line_end.setFrameShape(QFrame.HLine)
        line_end.setFrameShadow(QFrame.Sunken)
        parent_layout.addWidget(line_end)

        lay_end_btns = QHBoxLayout()
        self.btn_batch_export = QPushButton(_("btn_update_orthophotos"))
        self.btn_batch_export.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold; margin-top: 5px; padding: 6px;")
        self.btn_batch_export.clicked.connect(self.open_batch_export_dialog)
        self.btn_reset_retouches = QPushButton(_("btn_undo_all_edits"))
        self.btn_reset_retouches.setStyleSheet("color: white; font-weight: bold; margin-top: 5px; padding: 6px;")
        self.btn_reset_retouches.clicked.connect(self.reset_all_retouch)

        lay_end_btns.addWidget(self.btn_batch_export)
        lay_end_btns.addWidget(self.btn_reset_retouches)
        parent_layout.addLayout(lay_end_btns)

        # ==========================================
        # VERROUILLAGE INITIAL (On démarre en 3D)
        # ==========================================
        self.group_selection.setEnabled(False)
        self.group_color.setEnabled(False)
        self.group_batch.setEnabled(False)
        self.group_global_tex.setEnabled(False)
        self.btn_reset_retouches.setEnabled(False)
        self.btn_batch_export.setEnabled(False)

    def setup_language_ui(self, parent_layout):
        """Configure le sélecteur de langue en bas de l'onglet Mesh."""
        group_lang = QGroupBox(_("chk_language"))
        layout_lang = QVBoxLayout()
        self.combo_lang = QComboBox()
        self.combo_lang.setStyleSheet("padding: 4px; background-color: #1e1e1e; color: white; border: 1px solid #555;")

        # Scan dynamique du dossier 'locales'
        # --- Chemin sécurisé pour le scan ---
        base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        locales_dir = os.path.join(base_dir, "locales")

        # Petit dictionnaire pour afficher de jolis noms dans la liste
        # (Si un code n'y est pas, le programme affichera juste le code)
        lang_names = {
            "en": "English",
            "fr": "Français",
            "es": "Español",
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português"
        }

        # On vérifie que le dossier existe bien
        if os.path.exists(locales_dir) and os.path.isdir(locales_dir):
            for filename in os.listdir(locales_dir):
                if filename.endswith(".json"):
                    # On retire les 5 derniers caractères (".json") pour isoler le code (ex: "fr")
                    lang_code = filename[:-5]

                    # On cherche le joli nom, ou on utilise le code en majuscule par défaut (ex: "RU")
                    display_name = lang_names.get(lang_code, lang_code.upper())

                    # On ajoute à la liste (Nom affiché, Data cachée)
                    self.combo_lang.addItem(display_name, lang_code)
        else:
            # Fallback de sécurité au cas où le dossier est introuvable
            self.combo_lang.addItem("English", "en")
            self.combo_lang.addItem("Français", "fr")

        layout_lang.addWidget(self.combo_lang)
        group_lang.setLayout(layout_lang)
        parent_layout.addWidget(group_lang)

    def setup_cleanup_ui(self, parent_layout):
        """Configure le gestionnaire de fichiers personnalisés pour la tuile en cours."""
        self.group_cleanup = CollapsibleBox(_("group_cleanup"))
        layout_cleanup = QVBoxLayout()

        # 1. Liste déroulante des fichiers Custom
        layout_cleanup.addWidget(QLabel(_("lbl_custom_files")))
        self.combo_cleanup_files = QComboBox()
        self.combo_cleanup_files.setStyleSheet("background-color: #1e1e1e; color: white; padding: 4px; border: 1px solid #555;")
        self.combo_cleanup_files.currentIndexChanged.connect(self.on_cleanup_file_changed)
        layout_cleanup.addWidget(self.combo_cleanup_files)

        # 2. Liste déroulante des entrées (Projets)
        layout_cleanup.addWidget(QLabel(_("lbl_custom_entries")))
        self.combo_cleanup_entries = QComboBox()
        self.combo_cleanup_entries.setStyleSheet("background-color: #1e1e1e; color: white; padding: 4px; border: 1px solid #555;")
        layout_cleanup.addWidget(self.combo_cleanup_entries)

        # 3. Boutons d'action
        lay_btns = QHBoxLayout()

        self.btn_refresh_list = QPushButton(_("btn_refresh"))
        self.btn_refresh_list.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_refresh_list.clicked.connect(self.update_cleanup_files_list)

        self.btn_rename_entry = QPushButton(_("btn_rename"))
        self.btn_rename_entry.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.btn_rename_entry.clicked.connect(self.rename_custom_entry)

        self.btn_delete_entry = QPushButton(_("btn_delete"))
        self.btn_delete_entry.setStyleSheet("color: white; font-weight: bold;")
        self.btn_delete_entry.clicked.connect(self.delete_custom_entry)

        lay_btns.addWidget(self.btn_refresh_list)
        lay_btns.addWidget(self.btn_rename_entry)
        lay_btns.addWidget(self.btn_delete_entry)
        layout_cleanup.addLayout(lay_btns)

        self.group_cleanup.setContentLayout(layout_cleanup)
        parent_layout.addWidget(self.group_cleanup)

        # Désactivé par défaut jusqu'au chargement d'un Mesh
        self.group_cleanup.setEnabled(False)

    def setup_type_modifier_ui(self, parent_layout):
        """UI pour l'altération forcée des attributs des triangles (Bypass sécurités)."""
        self.group_type_mod = CollapsibleBox(_("group_type_mod"))
        layout = QVBoxLayout()

        self.btn_convert_type = QPushButton(_("btn_convert_type"))
        # Un style rouge/orangé pour signifier une action destructrice/radicale
        self.btn_convert_type.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 6px;")
        self.btn_convert_type.clicked.connect(self.convert_selected_triangles_type)

        layout.addWidget(self.btn_convert_type)
        self.group_type_mod.setContentLayout(layout)
        parent_layout.addWidget(self.group_type_mod)

    # =========================================================================
    #
    # 3. INTERACTION MATÉRIELLE & INTERCEPTION (VisPy / Mouse / Key)
    #
    # =========================================================================

    def on_mouse_press(self, event):
        # --- MODE 2D (Retouche) ---
        if getattr(self, 'is_2d_mode', False):

            # Mean shift (CTRL + Clic Gauche)
            if getattr(self, 'is_blur_mode', False):
                if 'Control' in [m.name for m in event.modifiers] and event.button == 1:
                    transform = self.image_2d_visual.get_transform('canvas', 'visual')
                    pos_image = transform.map(event.pos)
                    x, y = int(pos_image[0]), int(pos_image[1])
                    self.brush_points = [[x, y]]
                    self.is_brush_drawing = True
                    self.update_brush_visual()
                    event.handled = True
                return

            # SEAMLESS CLONE (CTRL + Clic Gauche)
            if getattr(self, 'is_seamless_mode', False) and getattr(self, 'selection_mask', None) is not None:
                if 'Control' in [m.name for m in event.modifiers] and event.button == 1:
                    transform = self.image_2d_visual.get_transform('canvas', 'visual')
                    pos_image = transform.map(event.pos)
                    x, y = int(pos_image[0]), int(pos_image[1])

                    # On vérifie qu'on clique bien DANS le masque rouge
                    if self.selection_mask[y, x] > 0:
                        self.is_seamless_dragging = True
                        self.seamless_start_pos = np.array([x, y])

                        # Extraction du contour pour le "Ghost" visuel
                        contours, _dummy = cv2.findContours(self.selection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            self.seamless_contour_pts = contours[0].reshape(-1, 2)
                            # Fermer la boucle du contour pour VisPy
                            self.seamless_contour_pts = np.vstack((self.seamless_contour_pts, self.seamless_contour_pts[0]))

                        event.handled = True
                    return

            # --- Outil Polygone (MAJ/Shift + Clics) ---
            if 'Shift' in [m.name for m in event.modifiers]:
                transform = self.image_2d_visual.get_transform('canvas', 'visual')
                pos_image = transform.map(event.pos)
                x, y = int(pos_image[0]), int(pos_image[1])

                if event.button == 1: # Clic Gauche : Ajouter un point
                    self.polygon_points.append([x, y])
                    self.polygon_redo_stack.clear()
                    self.update_polygon_visual()
                elif event.button == 2: # Clic Droit : Fermer et valider
                    self.apply_polygon_selection()
                return

            # --- SÉCURITÉ ANTI-CRASH ---
            return

        # --- MODE 3D : DÉCOUPE DE ZONE (Polygone avec Control) ---
        if getattr(self, 'zone_draw_mode', False) and 'Control' in [m.name for m in event.modifiers]:
            # Vérification de l'état du polygone
            is_closed = len(self.zone_polygon_points) >= 4 and (self.zone_polygon_points[0] == self.zone_polygon_points[-1])
            if is_closed:
                return

            if abs(abs(self.view.camera.elevation) - 90) > 5:
                QMessageBox.warning(self, _("msg_warning_title"), _("msg_cam_90"))
                return

            # On fige la caméra
            self.view.camera.interactive = False

            if event.button == 1: # Clic Gauche
                # Itération 1 : Raycast vers l'altitude du centre de l'écran
                target_z = self.view.camera.center[2]
                wx, wy = self._get_world_xy_from_canvas(event.pos[0], event.pos[1], target_z)
                ground_z = self.get_z_at_xy(wx, wy)

                # Itération 2 : Raycast CORRIGÉ vers l'altitude réelle du sol (Annule la parallaxe)
                wx_exact, wy_exact = self._get_world_xy_from_canvas(event.pos[0], event.pos[1], ground_z)
                # --- SÉCURITÉ ---
                if not self.is_xy_strictly_inside_mesh(wx_exact, wy_exact):
                    logging.warning("Outside mesh limits")
                    return
                ground_z_exact = self.get_z_at_xy(wx_exact, wy_exact)

                self.zone_polygon_points.append([wx_exact, wy_exact, ground_z_exact])
                self.zone_polygon_redo_stack.clear()
                self.update_zone_polygon_visual()

            elif event.button == 2: # Clic Droit : Fermer le tracé visuellement
                if len(self.zone_polygon_points) >= 3:
                    first_point = self.zone_polygon_points[0]
                    self.zone_polygon_points.append(first_point)
                    self.zone_polygon_redo_stack.clear()
                    self.update_zone_polygon_visual()

            event.handled = True
            return

        # 0.28.0 - MODE : DÉFINITION AXE PLAN INCLINÉ
        if getattr(self, 'tilted_define_active', False) and event.button == 1 and 'Control' in [m.name for m in event.modifiers]:
            self.view.camera.interactive = False

            target_z = self.view.camera.center[2]
            wx, wy = self._get_world_xy_from_canvas(event.pos[0], event.pos[1], target_z)
            ground_z = self.get_z_at_xy(wx, wy)
            wx_exact, wy_exact = self._get_world_xy_from_canvas(event.pos[0], event.pos[1], ground_z)

            self.tilted_pts_2d.append(np.array([wx_exact, wy_exact, ground_z]))

            if len(self.tilted_pts_2d) == 1:
                self.draw_temp_tilted_marker(wx_exact, wy_exact, ground_z)
            elif len(self.tilted_pts_2d) == 2:
                self.clear_temp_tilted_marker()
                self.btn_def_ab.setChecked(False)
                self.tilted_define_active = False
                self.btn_def_ab.setText(_("lbl_AB_axis"))
                self.init_tilted_plane_logic()

            event.handled = True
            return

        # MODE : PISTE
        if getattr(self, 'runway_active', False) and event.button == 1 and 'Control' in [m.name for m in event.modifiers]:
            if abs(abs(self.view.camera.elevation) - 90) > 5:
                QMessageBox.warning(self, _("msg_warning_title"), _("msg_cam_90"))
                return

            # On fige la caméra
            self.view.camera.interactive = False

            # --- CORRECTION PARALLAXE (Double Raycast) ---
            # Itération 1 : Raycast vers l'altitude du centre de l'écran (Approximation)
            target_z = self.view.camera.center[2]
            wx, wy = self._get_world_xy_from_canvas(event.pos[0], event.pos[1], target_z)
            ground_z = self.get_z_at_xy(wx, wy)

            # Itération 2 : Raycast CORRIGÉ vers l'altitude réelle du sol (Précision géométrique)
            wx_exact, wy_exact = self._get_world_xy_from_canvas(event.pos[0], event.pos[1], ground_z)
            # --- SÉCURITÉ ---
            if not self.is_xy_strictly_inside_mesh(wx_exact, wy_exact):
                logging.warning("Outside mesh limits")
                return
            self.runway_pts_2d.append(np.array([wx_exact, wy_exact]))

            # --- FEEDBACK VISUEL : PREMIER POINT ---
            if len(self.runway_pts_2d) == 1:
                self.draw_temp_runway_marker(wx_exact, wy_exact, ground_z)

            # --- DEUXIÈME POINT : VALIDATION DE L'AXE ---
            elif len(self.runway_pts_2d) == 2:
                self.clear_temp_runway_marker() # On nettoie la boule temporaire

                self.btn_toggle_runway.setChecked(False)
                self.runway_active = False
                self.btn_toggle_runway.setText(_("txt_axis_defined"))
                self.btn_toggle_runway.setEnabled(False)

                self.init_runway_table()
                self.update_runway_preview()

            event.handled = True
            return

        # 1 = CTRL+Clic Gauche (Sélection), 2 = CTRL+Clic Droit (Désélection)
        if 'Control' not in [m.name for m in event.modifiers]:
            return

        # Bloque la sélection rectangulaire si l'outil Zone ou Piste est actif
        if getattr(self, 'zone_draw_mode', False) or getattr(self, 'runway_active', False):
            return

        if abs(abs(self.view.camera.elevation) - 90) > 5:
            QMessageBox.warning(self, _("msg_warning_title"), _("msg_cam_90"))
            return

        if event.button in (1, 2):
            self.new_selection = True
            self.drag_button = event.button
            self.drag_origin = QPoint(int(event.pos[0]), int(event.pos[1]))
            self.rubber_band.setGeometry(QRect(self.drag_origin, QSize()))
            self.rubber_band.show()
            event.handled = True

    def on_mouse_move(self, event):
        if getattr(self, 'is_2d_mode', False):
            # Prise en charge du Zoom au Clic Droit
            if 2 in event.buttons and hasattr(self, 'chk_show_outline') and self.chk_show_outline.isChecked():
                self._zoom_timer.start(150)

            # Logique du curseur et du dessin
            if getattr(self, 'is_blur_mode', False):
                transform = self.image_2d_visual.get_transform('canvas', 'visual')
                pos_image = transform.map(event.pos)
                x, y = int(pos_image[0]), int(pos_image[1])

                # Affichage / Mise à jour du cercle sous le curseur
                if self.brush_cursor_visual is None:
                    # Création du cercle (transparent avec bordure blanche)
                    from vispy.scene.visuals import Ellipse
                    self.brush_cursor_visual = Ellipse(
                        center=(x, y),
                        radius=(self.brush_radius, self.brush_radius),
                        color=(1, 1, 1, 0.1),
                        border_color=(1, 1, 1, 0.8),
                        border_width=1,
                        parent=self.view.scene
                    )
                    self.brush_cursor_visual.order = 20 # Au-dessus de l'image
                else:
                    self.brush_cursor_visual.center = (x, y)
                    self.brush_cursor_visual.visible = True

                # Enregistrement des points si on est en train de cliquer/dessiner
                if getattr(self, 'is_brush_drawing', False):
                    self.brush_points.append([x, y])
                    self.update_brush_visual()

                event.handled = True
                return

            # --- SEAMLESS CLONE : GHOST VISUEL ---
            if getattr(self, 'is_seamless_dragging', False):
                transform = self.image_2d_visual.get_transform('canvas', 'visual')
                pos_image = transform.map(event.pos)
                current_pos = np.array([int(pos_image[0]), int(pos_image[1])])

                delta = current_pos - self.seamless_start_pos

                if self.seamless_contour_pts is not None:
                    shifted_pts = self.seamless_contour_pts + delta

                    if self.seamless_ghost_visual is None:
                        self.seamless_ghost_visual = scene.visuals.Line(
                            pos=shifted_pts, color='red', width=2, parent=self.view.scene
                        )
                        self.seamless_ghost_visual.set_gl_state('translucent', depth_test=False)
                        self.seamless_ghost_visual.order = 20
                    else:
                        self.seamless_ghost_visual.set_data(pos=shifted_pts)
                        self.seamless_ghost_visual.visible = True

                event.handled = True
                return

            return

        if hasattr(self, 'drag_button') and self.drag_button is not None:
            current_pos = QPoint(int(event.pos[0]), int(event.pos[1]))
            self.rubber_band.setGeometry(QRect(self.drag_origin, current_pos).normalized())
            event.handled = True

    def on_mouse_release(self, event):
        if getattr(self, 'is_2d_mode', False):
            # Application du pinceau au relâchement
            if getattr(self, 'is_blur_mode', False) and getattr(self, 'is_brush_drawing', False):
                self.is_brush_drawing = False
                self.apply_mean_shift()
                event.handled = True
                return

            # --- SEAMLESS CLONE : APPLICATION ---
            if getattr(self, 'is_seamless_dragging', False):
                self.is_seamless_dragging = False

                if getattr(self, 'seamless_ghost_visual', None):
                    self.seamless_ghost_visual.visible = False

                transform = self.image_2d_visual.get_transform('canvas', 'visual')
                pos_image = transform.map(event.pos)
                end_pos = np.array([int(pos_image[0]), int(pos_image[1])])
                self.apply_seamless_clone(end_pos)

                event.handled = True
                return
            return

        # On libère la caméra
        if hasattr(self.view, 'camera'):
            self.view.camera.interactive = True

        if hasattr(self, 'drag_button') and self.drag_button == event.button:
            self.rubber_band.hide()
            rect = self.rubber_band.geometry()

            if rect.width() < 3 and rect.height() < 3:
                rect.adjust(-2, -2, 2, 2)

            if self.drag_button == 1:
                # MODE ZONE
                if getattr(self, 'zone_draw_mode', False):
                    logging.info("Rectangle selection is disabled in Area Drawing Mode. Use SHIFT+Click to draw.")
                # --- MODE CLASSIQUE ---
                else:
                    self.select_triangles_in_rect(rect)
            elif self.drag_button == 2:
                self.unselect_triangles_in_rect(rect)

            self.drag_button = None
            event.handled = True

        # SYNCHRONISATION GLOBALE DU PIVOT
        if not getattr(self, 'is_editing_point', False) and hasattr(self, 'update_pivot_z'):
            self.update_pivot_z()

    def on_mouse_wheel(self, event):
        """Intercepte le zoom pour rafraîchir l'épaisseur du contour avec un délai (Debounce)."""
        # On ne le déclenche que si on est en 2D et que le contour (pas le remplissage) est actif
        if getattr(self, 'is_2d_mode', False) and hasattr(self, 'chk_show_outline') and self.chk_show_outline.isChecked():
            self._zoom_timer.start(150)

    def on_key_press(self, event):
        """Gère les raccourcis clavier locaux quand la vue 3D a le focus."""
        # Sécurité : on ignore les touches non mappées par Vispy (ex: Caps Lock)
        if event.key is None:
            return

        if getattr(self, 'is_2d_mode', False):
            return
        key = event.key.name

        # --- Navigation Relative (Caméra) ---
        if key == 'Up':
            self.move_camera_relative('forward')
        elif key == 'Down':
            self.move_camera_relative('backward')
        elif key == 'Left':
            self.move_camera_relative('left')
        elif key == 'Right':
            self.move_camera_relative('right')

        # --- Vues rapides ---
        elif key and key.upper() == 'V':
            self.set_top_down_view()

    def set_top_down_view(self):
        """Place la caméra exactement à la verticale pour permettre la sélection."""
        self.view.camera.elevation = 90
        self.view.camera.azimuth = 0
        logging.info("Camera positioned at 90°. Use CTRL+Click to start selection.")

    def toggle_airports(self, state):
        """Active ou désactive l'affichage des aéroports (Triangles + Marqueurs POI)."""
        self.show_airports = (state == Qt.Checked)

        # 1. On met à jour la couleur des triangles (comme avant)
        self.update_selection_colors()

        # 2. On affiche ou on masque les marqueurs flottants
        if hasattr(self, 'airport_markers') and self.airport_markers is not None:
            self.airport_markers.visible = self.show_airports

    def toggle_lakes(self, state):
        """Active ou désactive l'affichage de l'eau (Type 1)."""
        self.show_lakes = (state == Qt.Checked)
        self.update_selection_colors()

    def toggle_roads(self, state):
        """Active ou désactive l'affichage des routes (Type 8)."""
        self.show_roads = (state == Qt.Checked)
        self.update_selection_colors()

    def toggle_taxi(self, state):
        """Active ou désactive l'affichage des taxiways (Type 32)."""
        self.show_taxi = (state == Qt.Checked)
        self.update_selection_colors()

    def toggle_infra(self, state):
        """Active ou désactive l'affichage des autres infrastructures (Type >= 64)."""
        self.show_infra = (state == Qt.Checked)
        self.update_selection_colors()

    def toggle_cross(self, state):
        """Active ou désactive l'affichage des croisements de type (Types <> 0,1,2,4,8,16,32,62,128)."""
        self.show_cross = (state == Qt.Checked)
        self.update_selection_colors()

    # =========================================================================
    #
    # 4. GESTION DU MAILLAGE GLOBAL & DES I/O
    #
    # =========================================================================

    def load_mesh(self):
        # 1. SÉCURITÉ : Vérification des modifications non sauvegardées
        if getattr(self, 'is_modified', False):
            reply = QMessageBox.question(self, _("msg_warning"), _("msg_reload"),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # 2. OUVERTURE : Boîte de dialogue de sélection de fichier
        file_path, _filter = QFileDialog.getOpenFileName(self, _("dialog_select_mesh"), self.last_mesh_dir, _("dialog_filter"))

        if not file_path:
            return # L'utilisateur a cliqué sur "Annuler"

        # On met à jour le dernier dossier utilisé
        mesh_dir = os.path.normpath(os.path.dirname(file_path))
        self.last_mesh_dir = mesh_dir

        # Détection automatique du dossier textures
        potential_dds_dir = os.path.normpath(os.path.join(mesh_dir, "textures"))

        # On vérifie si le sous-dossier "textures" existe
        if os.path.isdir(potential_dds_dir):
            # On récupère le chemin actuel de l'UI et on le normalise aussi s'il n'est pas vide
            raw_current_path = self.dds_dir_input.text()
            current_dds_path = os.path.normpath(raw_current_path) if raw_current_path else ""

            # On ne met à jour que si le chemin (nettoyé) est fondamentalement différent
            if potential_dds_dir != current_dds_path:
                self.dds_dir_input.setText(potential_dds_dir)
                logging.info(f"DDS folder automatically updated : {potential_dds_dir}")

        # 3. RÉINITIALISATION COMPLÈTE de l'état pour le nouveau mesh
        self.reset_state_for_new_mesh()

        self.file_input.setText(os.path.normpath(file_path))
        logging.info(f"Loading of {file_path}...")

        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            # ---------------------------------------------------------
            # LECTURE DELEGUÉE AU MODULE MESH_IO
            # ---------------------------------------------------------
            vertices, uvs, faces, tri_types, nbr_nodes, nbr_tris = parse_ortho_mesh(file_path)

            logging.info("Parsing done. Preparing 3D data...")

            # --- NORMALISATION SPATIALE (Degrés vers Mètres) ---
            self.mean_lat = np.mean(vertices[:, 1])
            self.lat_to_m = 111120.0
            self.lon_to_m = 111120.0 * np.cos(np.radians(self.mean_lat))

            self.x_center = (np.min(vertices[:, 0]) + np.max(vertices[:, 0])) / 2.0
            self.y_center = (np.min(vertices[:, 1]) + np.max(vertices[:, 1])) / 2.0

            vertices[:, 0] = (vertices[:, 0] - self.x_center) * self.lon_to_m
            vertices[:, 1] = (vertices[:, 1] - self.y_center) * self.lat_to_m

            self.original_vertices = vertices
            self.original_uvs = uvs
            self.original_faces = faces
            self.original_tri_types = tri_types
            self.original_tri_levels = np.zeros(len(faces), dtype=np.uint8)

            self.backup_vertices = vertices.copy()
            self.backup_uvs = uvs.copy()
            self.backup_faces = faces.copy()
            self.backup_tri_types = tri_types.copy()
            self.backup_tri_levels = self.original_tri_levels.copy()

            # --- AFFICHAGE ---
            self.mesh_visual = scene.visuals.Mesh(vertices=self.original_vertices,
                                                  faces=self.original_faces,
                                                  color=(0.4, 0.4, 0.4, 1.0),
                                                  shading='flat')

            wireframe_filter = WireframeFilter(color='white', width=0.5)
            self.mesh_visual.attach(wireframe_filter)

            self.view.add(self.mesh_visual)

            # --- CORRECTION DE LA CAMÉRA ---
            # 1. Calcul de l'altitude au centre géographique
            dist_2d_sq_center = (vertices[:, 0] - 0)**2 + (vertices[:, 1] - 0)**2
            closest_idx_center = np.argmin(dist_2d_sq_center)
            z_center = vertices[closest_idx_center, 2]

            # 2. On laisse VisPy calculer les limites d'affichage (clipping planes) pour la nouvelle géométrie
            self.view.camera.set_range()

            # 3. ÉCRASEMENT TOTAL : On force le retour aux paramètres d'usine de la vue
            self.view.camera.azimuth = 0          # Rotation horizontale à 0 (Face au Nord)
            self.view.camera.elevation = 30       # Inclinaison par défaut
            self.view.camera.distance = 150000    # Niveau de zoom (Perspective)
            self.view.camera.scale_factor = 150000 # Niveau de zoom (Orthographique/Hybride)
            self.view.camera.center = (0, 0, z_center) # Centrage géographique strict

            # --- REPÈRES VISUELS (PIVOT) ---
            self.z_min = np.min(vertices[:, 2])
            self.z_max = np.max(vertices[:, 2])

            line_coords = np.array([[0, 0, self.z_min - 2000], [0, 0, self.z_max + 2000]])
            self.pivot_line = scene.visuals.Line(pos=line_coords, color='cyan', width=4, parent=self.view.scene)

            self.pivot_marker = scene.visuals.Markers(parent=self.view.scene)
            self.pivot_marker.set_data(pos=np.array([[0, 0, z_center]]), face_color='red', edge_color='white', size=12)

            # --- CRÉATION DES MARQUEURS D'AÉROPORTS ---
            airport_centers = get_airport_centers(self.original_vertices, self.original_faces, self.original_tri_types)

            if len(airport_centers) > 0:
                self.airport_markers = scene.visuals.Markers(parent=self.view.scene)
                self.airport_markers.set_data(pos=np.array(airport_centers),
                                              face_color=(0.8, 0.2, 1.0, 1.0),
                                              edge_color='white', symbol='+', size=15)
                self.airport_markers.visible = self.show_airports
                logging.info(f"{len(airport_centers)} airports found.")
            else:
                self.airport_markers = None
                logging.info("No runway detected in this tile.")

            # Le nouveau mesh est vierge de modification !
            self.is_modified = False

            # --- STATISTIQUES DU MESH ---
            # np.unique retourne les valeurs uniques triées et leurs fréquences respectives
            unique_types, counts = np.unique(self.original_tri_types, return_counts=True)

            # Formatage élégant : "Type 0: 150240, Type 1: 450, Type 16: 12"
            stats_str = ", ".join([f"Type {t}: {c}" for t, c in zip(unique_types, counts)])
            logging.info(f"Mesh geometry distribution -> {stats_str}")
            # ----------------------------

            # Calcul des bornes du mesh (attention le mesh est un trapèze)
            self._global_min_x = np.min(self.original_vertices[:, 0])
            self._global_max_x = np.max(self.original_vertices[:, 0])
            self._global_min_y = np.min(self.original_vertices[:, 1])
            self._global_max_y = np.max(self.original_vertices[:, 1])

            # Déverrouillage des onglets
            for i in range(1, self.tabs.count()):
                self.tabs.setTabEnabled(i, True)

            self.update_mesh_stats()
            self.update_cleanup_files_list()
            self.canvas.native.setFocus()

            # QMessageBox.information(self, _("msg_success_title"), _("msg_mesh_loaded", nodes=nbr_nodes, tris=nbr_tris))
            QApplication.restoreOverrideCursor()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, _("msg_fatal_error_title"), _("msg_parse_error", error=str(e)))

    def export_mesh(self):
        """Demande le chemin du fichier, affiche le sablier et lance l'exportation."""
        if getattr(self, 'is_2d_mode', False):
            return

        if self.mesh_visual is None:
            QMessageBox.warning(self, _("msg_error_title"), _("msg_no_mesh"))
            return

        if not getattr(self, 'is_modified', False):
            QMessageBox.information(self, _("msg_info_title"), _("msg_not_modified"))
            return

        # Utilisation de self.last_mesh_dir
        file_path, _filter = QFileDialog.getSaveFileName(self, _("dialog_save_mesh"), getattr(self, 'last_mesh_dir', ''), _("dialog_filter_mesh"))
        if not file_path:
            return

        # On met à jour le dernier dossier utilisé
        self.last_mesh_dir = os.path.dirname(file_path)

        logging.info(f"Preparing export to {file_path}...")

        # --- 1. ACTIVATION DU SABLIER ---
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # --- 2. PAUSE STRATÉGIQUE (On rend la main à l'interface pendant 50ms) ---
        # Cela laisse le temps au système de dessiner le sablier avant de bloquer le processeur.
        QTimer.singleShot(50, lambda: self._do_export_mesh(file_path))

    def _do_export_mesh(self, file_path):
        """Partie calculatoire de l'exportation (protégée par le sablier)."""
        try:
            # --- 2. CALCULS LOURDS ---
            # Inversion de la projection (Mètres -> Degrés Lat/Lon)
            export_vertices = self.original_vertices.copy()
            export_vertices[:, 0] = (export_vertices[:, 0] / self.lon_to_m) + self.x_center
            export_vertices[:, 1] = (export_vertices[:, 1] / self.lat_to_m) + self.y_center

            # --- 3. ECRITURE DELEGUÉE AU MODULE MESH_IO ---
            write_ortho_mesh(
                file_path,
                export_vertices,
                self.original_uvs,
                self.original_faces,
                self.original_tri_types
            )

            nbr_vert = len(export_vertices)
            nbr_tri = len(self.original_faces)

            # --- 4. DÉSACTIVATION DU SABLIER (Toujours avant la popup !) ---
            QApplication.restoreOverrideCursor()

            QMessageBox.information(self, _("msg_success_title"), _("msg_export_success", nodes=nbr_vert, tris=nbr_tri))
            self.is_modified = False
            logging.info("Export done.")

        except Exception as e:
            # En cas de crash, on s'assure que le sablier ne reste pas bloqué
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, _("msg_fatal_error_title"), _("msg_write_error", error=str(e)))

    def reset_mesh(self):
        """Restaure le mesh dans son état d'origine en préservant la caméra et les repères."""
        if self.mesh_visual is None or not hasattr(self, 'backup_vertices'):
            return

        if getattr(self, 'is_modified', False):
            reply = QMessageBox.question(self, _("msg_warning"), _("msg_undo_reset_mesh"),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
        else:
            return

        logging.info("Restoring original mesh...")
        self.save_state_to_history()

        # 1. Restauration des données depuis la copie de sauvegarde
        self.original_vertices = self.backup_vertices.copy()
        self.original_uvs = self.backup_uvs.copy()
        self.original_faces = self.backup_faces.copy()
        self.original_tri_types = self.backup_tri_types.copy()
        self.original_tri_levels = self.backup_tri_levels.copy()

        # 2. On vide la sélection pour éviter tout crash avec des index obsolètes
        self.selected_faces_indices = []

        # 3. Mise à jour visuelle (appel de notre fonction centralisée)
        self.update_selection_colors()
        self.canvas.native.setFocus()

        self.is_modified = False
        logging.info("Mesh successfully restored and selection cleared.")

    def reset_state_for_new_mesh(self):
        """Réinitialise de manière exhaustive l'état et l'interface pour un nouveau fichier."""

        # 1. FORCER LE RETOUR EN 3D (Si on était en mode retouche 2D)
        if getattr(self, 'is_2d_mode', False):
            self.btn_toggle_2d.setChecked(False)
            self.toggle_2d_mode() # Cette méthode s'occupe de masquer les éléments 2D et de déverrouiller l'UI

        # 2. FERMETURE PROPRE DES OUTILS ACTIFS (Délégation)
        # On utilise les méthodes dédiées qui gèrent déjà le nettoyage des variables ET de l'UI
        if getattr(self, 'zone_draw_mode', False): self.cancel_zone()
        if getattr(self, 'runway_active', False): self.cancel_runway()
        if getattr(self, 'cylinder_active', False): self.cancel_cylinder()
        if getattr(self, 'is_editing_point', False): self.cancel_point()
        if getattr(self, 'flatten_active', False): self.abort_flattening()

        self.cancel_color_retouch()
        self.clear_selection_mask()
        self.cancel_selection()
        self.clear_textures() # Indispensable pour vider le cache des images HD

        # 3. NETTOYAGE DES STRUCTURES DE DONNÉES GLOBALES
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_undo_redo_buttons()
        self.is_modified = False
        self.new_selection = True

        self.x_center = 0.0
        self.y_center = 0.0
        self.z_min = 0.0
        self.z_max = 0.0

        if hasattr(self, 'heal_history'): self.heal_history.clear()
        if hasattr(self, 'heal_redo_stack'): self.heal_redo_stack.clear()
        self.has_healing_edits = False

        # Masques et matrices globales
        self.cumulative_mask = None
        self._global_canvas_data = None
        self._original_canvas_data = None
        self._global_bounds = None
        self.batch_2d_state = None

        # 4. DESTRUCTION DES OBJETS VISPY RÉSIDUELS (Liés à la scène globale)
        for attr in ['mesh_visual', 'airport_markers', 'pivot_marker', 'pivot_line',
                     'global_texture_mesh', 'image_2d_visual', 'grid_2d_visual', 'mesh_mask_visual']:
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.parent = None
                setattr(self, attr, None)

        # 5. RÉINITIALISATION STRICTE DE LA CAMÉRA
        self.view.camera.azimuth = 0
        self.view.camera.elevation = 30
        self.view.camera.center = (0, 0, 0)

        # 6. RÉINITIALISATION DE L'INTERFACE UTILISATEUR GLOBALE
        self.btn_toggle_global_tex.setChecked(False)
        self.btn_toggle_global_tex.setEnabled(False)
        if hasattr(self, 'btn_apply_all_sel_2d'):
            self.btn_apply_all_sel_2d.setEnabled(False)

        # Réinitialisation des filtres d'affichage ---
        checkboxes_to_reset = ['cb_show_airports', 'cb_show_lakes', 'cb_show_roads',
                               'cb_show_taxi', 'cb_show_infra', 'cb_show_cross']
        for cb_name in checkboxes_to_reset:
            if hasattr(self, cb_name):
                cb_widget = getattr(self, cb_name)
                # On bloque les signaux pour ne pas lancer les méthodes toggle_* dans le vide
                cb_widget.blockSignals(True)
                cb_widget.setChecked(False)
                cb_widget.blockSignals(False)

        self.show_airports = False
        self.show_lakes = False
        self.show_roads = False
        self.show_taxi = False
        self.show_infra = False
        self.show_cross = False

        if hasattr(self, 'lbl_mesh_stats'):
            self.lbl_mesh_stats.setText("Triangles : 0")

    def move_camera_relative(self, direction):
        """Déplace le centre d'observation relativement à l'angle de vue de la caméra (Azimut)."""
        if self.mesh_visual is None or not hasattr(self, 'original_vertices'):
            return

        step = self.view.camera.scale_factor * 0.05
        center = list(self.view.camera.center)

        # 1. Récupération de l'azimut en radians
        # VisPy utilise un azimut en degrés. 0 = face au Nord (+Y).
        azimuth_rad = math.radians(self.view.camera.azimuth)

        # 2. Calcul du vecteur "Avant" (Forward) basé sur la trigonométrie standard
        # Selon le sens de rotation (horaire/anti-horaire) de VisPy, il est possible
        # que le signe du Sinus doive être inversé. En général, c'est -sin(x) pour X et cos(x) pour Y.
        dir_x_forward = -math.sin(azimuth_rad)
        dir_y_forward = math.cos(azimuth_rad)

        # Le vecteur "Droite" (Right) est décalé de 90 degrés (-pi/2)
        dir_x_right = math.cos(azimuth_rad)
        dir_y_right = math.sin(azimuth_rad)

        # 3. Application du mouvement vectoriel
        if direction == 'forward':
            center[0] += dir_x_forward * step
            center[1] += dir_y_forward * step
        elif direction == 'backward':
            center[0] -= dir_x_forward * step
            center[1] -= dir_y_forward * step
        elif direction == 'right':
            center[0] += dir_x_right * step
            center[1] += dir_y_right * step
        elif direction == 'left':
            center[0] -= dir_x_right * step
            center[1] -= dir_y_right * step

        # 4. Applique le nouveau centre à la caméra (X et Y ont changé)
        self.view.camera.center = center

        # 5. Recalcule le Z, ajuste la caméra et les marqueurs
        self.update_pivot_z()

    def update_pivot_z(self):
        """Recalcule et met à jour l'altitude du pivot selon la géométrie actuelle sous la caméra."""
        if getattr(self, 'mesh_visual', None) is None or not hasattr(self, 'original_vertices'):
            return

        center = list(self.view.camera.center)

        # 1. Calcul de l'altitude (Z) exacte sous la caméra
        ground_z = self.get_z_at_xy(center[0], center[1])

        # 2. Mise à jour de l'altitude du centre
        center[2] = ground_z
        self.view.camera.center = center

        # 3. Mise à jour des UI (Champs texte)
        self.pivot_alt_input.setText(f"{ground_z:.1f}")
        if not getattr(self, 'cylinder_active', False):
            self.base_alt_input.setText(f"{ground_z-200:.1f}")

        # 4. Mise à jour des repères visuels (Point et Ligne)
        if hasattr(self, 'pivot_marker') and hasattr(self, 'pivot_line'):
            self.pivot_marker.set_data(pos=np.array([center]), face_color='red', edge_color='white', size=12)

            line_coords = np.array([
                [center[0], center[1], getattr(self, 'z_min', -2000) - 2000],
                [center[0], center[1], getattr(self, 'z_max', 2000) + 2000]
            ])
            self.pivot_line.set_data(pos=line_coords)

    def update_selection_colors(self):
        """Centralise la mise à jour des couleurs avec gestion des calques (Aéroports puis Sélection)."""
        if self.mesh_visual is None or not hasattr(self, 'original_tri_types'): return

        # 1. Base : Tout en gris
        colors = np.full((len(self.original_faces), 4), (0.4, 0.4, 0.4, 1.0), dtype=np.float32)
        T = self.original_tri_types

        # 2. Aéroports : En violet si l'option est cochée
        if self.show_airports:
            airport_mask = T == 16
            colors[airport_mask] = (0.6, 0.2, 0.8, 1.0) # Violet

        # 3. Lacs / Eau : En bleu si coché (Type 1 et 2)
        if self.show_lakes:
            lakes_mask = T == 1
            colors[lakes_mask] = (0.2, 0.6, 1.0, 1.0) # Bleu ciel (DodgerBlue)
            sea_mask = T == 2
            colors[sea_mask] = (0.2, 0.2, 1.0, 1.0)

        # 4. Routes : En rose si coché (Type 8)
        if getattr(self, 'show_roads', False):
            roads_mask = T == 8
            colors[roads_mask] = (1.0, 0.4, 0.7, 1.0) # HotPink

        # 5. Taxiways : En jaune si coché (Type 32)
        if getattr(self, 'show_taxi', False):
            taxi_mask = T == 32
            colors[taxi_mask] = (1.0, 0.85, 0.1, 1.0) # Yellow

        # 6. Autres infrastructures : En vert si coché (Type >=64)
        if getattr(self, 'show_infra', False):
            infra_mask = T >= 64
            colors[infra_mask] = (0.1, 1.0, 0.1, 1.0) # Green

        # 7. Croisements : En rouge si coché (Type 32)
        if getattr(self, 'show_cross', False):
            cross_mask = (T >= 3) & (T <= 255) & ((T & (T - 1)) != 0)
            colors[cross_mask] = (1.0, 0.1, 0.1, 1.0) # Red

        # 8. Sélection : En orange par-dessus tout (priorité absolue)
        if len(self.selected_faces_indices) > 0:
            # Sécurité: conversion en tableau d'entiers
            sel_idx = np.array(list(self.selected_faces_indices), dtype=int)
            colors[sel_idx] = (1.0, 0.5, 0.0, 1.0)

        self.mesh_visual.set_data(vertices=self.original_vertices,
            faces=self.original_faces,
            face_colors=colors)

        # Vérification pour le bouton d'aplanissement
        has_selection = len(self.selected_faces_indices) > 0
        if not has_selection and getattr(self, 'flatten_active', False):
            self.abort_flattening()

        # === Mise à jour du compteur ===
        self.update_mesh_stats()

    def update_mesh_stats(self):
        """Met à jour le compteur de triangles dans l'interface (ignoré en mode 2D)."""
        if getattr(self, 'is_2d_mode', False):
            return

        if hasattr(self, 'original_faces') and self.original_faces is not None:
            nb_triangles = len(self.original_faces)
            # Formatage avec espace pour les milliers (ex: 1 542 300)
            formatted_count = f"{nb_triangles:,}".replace(',', ' ')
            self.lbl_mesh_stats.setText(f"Triangles : {formatted_count}")
        else:
            self.lbl_mesh_stats.setText("Triangles : 0")

    def update_cleanup_files_list(self):
        """Rafraîchit la liste des fichiers existants pour la tuile en cours."""

        # --- 1. Mémorisation de l'état actuel ---
        current_file_data = self.combo_cleanup_files.currentData()
        current_entry_text = self.combo_cleanup_entries.currentText()

        self.combo_cleanup_files.blockSignals(True)
        self.combo_cleanup_files.clear()
        self.combo_cleanup_entries.clear()

        tile_id = self.get_current_tile_id()
        if tile_id == "unknown_tile":
            self.group_cleanup.setEnabled(False)
            self.combo_cleanup_files.blockSignals(False)
            return

        self.group_cleanup.setEnabled(True)

        # Liste des fichiers cibles gérés par l'application
        known_files = ["custom_flat.json", "custom_runways.json", "custom_2D_selections.json", "custom_colors.json"]

        for fname in known_files:
            fpath = self.get_custom_file_path(fname)
            if os.path.exists(fpath):
                # On utilise la data de l'item pour stocker le chemin complet
                self.combo_cleanup_files.addItem(fname, fpath)

        # --- 2. Restauration du fichier précédemment sélectionné ---
        if current_file_data:
            idx = self.combo_cleanup_files.findData(current_file_data)
            if idx >= 0:
                self.combo_cleanup_files.setCurrentIndex(idx)

        self.combo_cleanup_files.blockSignals(False)
        self.on_cleanup_file_changed()

        # --- 3. Restauration de l'entrée précédemment sélectionnée ---
        if current_entry_text:
            entry_idx = self.combo_cleanup_entries.findText(current_entry_text)
            if entry_idx >= 0:
                self.combo_cleanup_entries.setCurrentIndex(entry_idx)

    def on_cleanup_file_changed(self):
        """Remplit la seconde combobox en fonction du fichier sélectionné."""
        self.combo_cleanup_entries.clear()
        fpath = self.combo_cleanup_files.currentData()

        if not fpath or not os.path.exists(fpath):
            self.btn_rename_entry.setEnabled(False)
            self.btn_delete_entry.setEnabled(False)
            return

        tile_id = self.get_current_tile_id()
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if tile_id in data:
                entries = data[tile_id]
                # Différence structurelle : listes (tracés) vs dictionnaires (couleurs)
                if isinstance(entries, list):
                    for item in entries:
                        if "name" in item:
                            self.combo_cleanup_entries.addItem(item["name"])
                elif isinstance(entries, dict):
                    for key in entries.keys():
                        self.combo_cleanup_entries.addItem(key)
        except Exception as e:
            logging.error(f"Cleanup: Error reading {fpath} - {e}")

        has_entries = self.combo_cleanup_entries.count() > 0
        self.btn_rename_entry.setEnabled(has_entries)
        self.btn_delete_entry.setEnabled(has_entries)

    def rename_custom_entry(self):
        """Renomme l'entrée sélectionnée dans le JSON correspondant."""
        current_name = self.combo_cleanup_entries.currentText()
        if not current_name: return

        new_name, ok = QInputDialog.getText(self, _("msg_rename_title"), _("msg_rename_prompt"), text=current_name)
        if not ok or not new_name.strip() or new_name.strip() == current_name:
            return

        new_name = new_name.strip()
        fpath = self.combo_cleanup_files.currentData()
        tile_id = self.get_current_tile_id()

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            entries = data.get(tile_id)

            if isinstance(entries, list):
                if any(e.get("name") == new_name for e in entries):
                    QMessageBox.warning(self, _("msg_error_title"), _("msg_name_exists"))
                    return
                for e in entries:
                    if e.get("name") == current_name:
                        e["name"] = new_name
                        break
            elif isinstance(entries, dict):
                if new_name in entries:
                    QMessageBox.warning(self, _("msg_error_title"), _("msg_name_exists"))
                    return
                entries[new_name] = entries.pop(current_name)

            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            self.on_cleanup_file_changed()

            # --- CORRECTION : Pointer sur le nouvel élément renommé ---
            new_index = self.combo_cleanup_entries.findText(new_name)
            if new_index >= 0:
                self.combo_cleanup_entries.setCurrentIndex(new_index)

            # QMessageBox.information(self, _("msg_success_title"), _("msg_rename_success"))

        except Exception as e:
            QMessageBox.critical(self, _("msg_error_title"), str(e))

    def delete_custom_entry(self):
        """Supprime l'entrée sélectionnée avec confirmation."""
        current_name = self.combo_cleanup_entries.currentText()
        if not current_name: return

        # --- CORRECTION : On mémorise l'index avant suppression ---
        current_entry_idx = self.combo_cleanup_entries.currentIndex()

        reply = QMessageBox.warning(self, _("msg_warning_title"), _("msg_confirm_delete").replace("{name}", current_name),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.No: return

        fpath = self.combo_cleanup_files.currentData()
        tile_id = self.get_current_tile_id()

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            entries = data.get(tile_id)
            if isinstance(entries, list):
                data[tile_id] = [e for e in entries if e.get("name") != current_name]
            elif isinstance(entries, dict):
                data[tile_id].pop(current_name, None)

            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            self.update_cleanup_files_list() # On rafraîchit tout au cas où le fichier se vide

            # --- CORRECTION : Repli intelligent sur l'élément voisin ---
            new_count = self.combo_cleanup_entries.count()
            if new_count > 0:
                # min() permet de s'assurer que si on supprime le dernier élément (ex: index 3 sur 4),
                # on se replie sur le nouveau dernier (index 2).
                new_idx = min(current_entry_idx, new_count - 1)
                self.combo_cleanup_entries.setCurrentIndex(new_idx)

            # QMessageBox.information(self, _("msg_success_title"), _("msg_delete_success"))

        except Exception as e:
            QMessageBox.critical(self, _("msg_error_title"), str(e))

    # =========================================================================
    #
    # 5. HISTORIQUE & OUTILS DE SÉLECTION GÉNÉRIQUES (Undo / Redo)
    #
    # =========================================================================

    def get_current_state(self):
        """Capture l'état actuel du maillage et de la sélection."""
        return {
            'vertices': self.original_vertices.copy(),
            'uvs': self.original_uvs.copy(),
            'faces': self.original_faces.copy(),
            'tri_types': self.original_tri_types.copy(),
            'tri_levels': getattr(self, 'original_tri_levels', np.zeros(len(self.original_faces), dtype=np.uint8)).copy(),
            'selection': list(self.selected_faces_indices)
        }

    def save_state_to_history(self, action_name="Action standard", custom_state=None):
        """
        Enregistre un état dans la pile Undo avec des métadonnées.
        """
        state_data = custom_state if custom_state else self.get_current_state()

        # L'encapsulation !
        history_entry = {
            'name': action_name,
            'data': state_data
        }

        self.undo_stack.append(history_entry)

        if len(self.undo_stack) > self.MAX_HISTORY:
            self.undo_stack.pop(0)

        self.redo_stack.clear()
        self.update_undo_redo_buttons()

        logging.info(f"State saved : '{action_name}' (Stack: {len(self.undo_stack)}).")

    def restore_state(self, state):
        """Applique physiquement et visuellement un état du maillage."""
        self.original_vertices = state['vertices'].copy()
        self.original_uvs = state['uvs'].copy()
        self.original_faces = state['faces'].copy()
        self.original_tri_types = state['tri_types'].copy()
        if 'tri_levels' in state:
            self.original_tri_levels = state['tri_levels'].copy()
        self.selected_faces_indices = list(state['selection'])

        self.is_modified = True
        self.update_selection_colors()

        logging.info(f"Mesh structures restored: {len(self.original_vertices)} vertices, {len(self.original_faces)} faces.")

    def undo_action(self):
        if getattr(self, 'is_2d_mode', False):
            # Priorité 1 : Le lasso polygonal en cours de tracé
            if getattr(self, 'polygon_points', []):
                self.undo_polygon_point()
            # Priorité 2 : L'historique permanent des pixels (Mean Shift / Seamless)
            elif getattr(self, 'heal_history', []):
                self.undo_last_heal()
            else:
                logging.info("Nothing to undo in 2D mode.")
            return

        # Interception pour le lasso 3D
        if getattr(self, 'zone_draw_mode', False):
            if getattr(self, 'zone_polygon_points', []):
                self.undo_zone_point()
            return

        if getattr(self, 'is_editing_point', False) or getattr(self, 'flatten_active', False):
            QMessageBox.warning(self, _("msg_action_in_progress"), _("msg_histo_warning"))
            return

        if not getattr(self, 'undo_stack', []): return

        # On prépare l'état actuel pour le Redo, en reprenant le nom de l'action qu'on annule
        previous_entry = self.undo_stack.pop()

        self.redo_stack.append({
            'name': previous_entry['name'],
            'data': self.get_current_state()
        })

        # On restaure les données pures
        self.restore_state(previous_entry['data'])
        self.update_undo_redo_buttons()
        self.update_pivot_z()
        self.canvas.native.setFocus()

        logging.info(f"Undo action : '{previous_entry['name']}'.")

    def redo_action(self):
        if getattr(self, 'is_2d_mode', False):
            # Priorité 1 : Le lasso polygonal
            if getattr(self, 'polygon_redo_stack', []):
                self.redo_polygon_point()
            # Priorité 2 : L'historique permanent des pixels
            elif getattr(self, 'heal_redo_stack', []):
                self.redo_last_heal()
            else:
                logging.info("Nothing to redo in 2D mode.")
            return

        # Interception pour le lasso 3D
        if getattr(self, 'zone_draw_mode', False):
            if getattr(self, 'zone_polygon_redo_stack', []):
                self.redo_zone_point()
            return

        if getattr(self, 'is_editing_point', False) or getattr(self, 'flatten_active', False):
            QMessageBox.warning(self, _("msg_action_in_progress"), _("msg_histo_warning"))
            return

        if not getattr(self, 'redo_stack', []): return

        next_entry = self.redo_stack.pop()

        self.undo_stack.append({
            'name': next_entry['name'],
            'data': self.get_current_state()
        })

        self.restore_state(next_entry['data'])
        self.update_undo_redo_buttons()
        self.update_pivot_z()
        self.canvas.native.setFocus()

        logging.info(f"Redo action : '{next_entry['name']}'.")

    def update_undo_redo_buttons(self):
        """Met à jour l'état cliquable des boutons d'historique selon le contexte actif."""
        if not hasattr(self, 'btn_undo'):
            return

        # 1. Mode 2D : Lasso Polygonal ou mean shift ou seamless clone
        if getattr(self, 'is_2d_mode', False):
            # Priorité absolue au tracé de lasso en cours (si des points sont saisis)
            if getattr(self, 'polygon_points', []) or getattr(self, 'polygon_redo_stack', []):
                self.btn_undo.setEnabled(len(self.polygon_points) > 0)
                self.btn_redo.setEnabled(len(self.polygon_redo_stack) > 0)
            # Sinon, c'est l'historique permanent des réparations de pixels de la session 2D
            else:
                self.btn_undo.setEnabled(len(getattr(self, 'heal_history', [])) > 0)
                self.btn_redo.setEnabled(len(getattr(self, 'heal_redo_stack', [])) > 0)
        # 2. Mode 3D : Lasso de Zone Complexe
        elif getattr(self, 'zone_draw_mode', False):
            self.btn_undo.setEnabled(len(getattr(self, 'zone_polygon_points', [])) > 0)
            self.btn_redo.setEnabled(len(getattr(self, 'zone_polygon_redo_stack', [])) > 0)
        # 3. Par défaut : Maillage 3D global
        else:
            self.btn_undo.setEnabled(len(getattr(self, 'undo_stack', [])) > 0)
            self.btn_redo.setEnabled(len(getattr(self, 'redo_stack', [])) > 0)

    def cancel_selection(self):
        """Annule la sélection 3D ou vide le masque 2D selon le mode actif."""
        # 1. SI ON EST EN MODE RETOUCHE 2D : On vide le masque OpenCV
        if getattr(self, 'is_2d_mode', False):
            if getattr(self, 'batch_2d_state', None) is not None:
                # Si on prévisualisait les couleurs (Preview All), on détruit le calque HD volant
                if getattr(self, 'retouch_preview_visual', None) is not None:
                    self.retouch_preview_visual.parent = None
                    self.retouch_preview_visual = None

                # On déverrouille les panneaux de l'interface
                if hasattr(self, 'group_selection'):
                    self.group_selection.setEnabled(True)
                if hasattr(self, 'group_color'):
                    self.group_color.setEnabled(True)

                self.batch_2d_state = None

            self.clear_selection_mask()

            if hasattr(self, 'btn_apply_all_sel_2d'):
                self.btn_apply_all_sel_2d.setEnabled(False)
            return

        # 2. SI ON EST EN 3D
        if getattr(self, 'mesh_visual', None) is None:
            return

        # On vide la liste des triangles sélectionnés
        self.selected_faces_indices = []

        # On réactive le bouton de la baguette magique
        if hasattr(self, 'btn_magic_wand'):
            self.btn_magic_wand.setEnabled(True)

        # On met à jour l'affichage avec notre fonction centralisée
        self.update_selection_colors()

        logging.info("Selection canceled. Colors restored.")

        self.canvas.native.setFocus()

    def select_triangles_in_rect(self, rect):
        """Récupère les index et les AJOUTE à la sélection globale."""
        new_indices = self._get_triangles_in_rect(rect)
        new_indices = [idx for idx in new_indices if (self.original_tri_types[idx] & 3) == 0]

        if new_indices:
            # Utilisation d'un set pour l'union (élimine les doublons)
            current_selection = set(self.selected_faces_indices)
            current_selection.update(new_indices)
            self.selected_faces_indices = list(current_selection)

            logging.info(f"Zone added: {len(new_indices)} triangles. Total: {len(self.selected_faces_indices)}")
            self.update_selection_colors()

            if self.selected_faces_indices:
                sel_array = np.array(self.selected_faces_indices, dtype=int)
                unique_types = np.unique(self.original_tri_types[sel_array])
                logging.info(f"Triangle types included in selection : {unique_types.tolist()}")

    def unselect_triangles_in_rect(self, rect):
        """Récupère les index et les SOUSTRAIT de la sélection globale."""
        indices_to_remove = self._get_triangles_in_rect(rect)
        if indices_to_remove:
            # Utilisation d'un set pour la soustraction
            current_selection = set(self.selected_faces_indices)
            current_selection.difference_update(indices_to_remove)
            self.selected_faces_indices = list(current_selection)

            logging.info(f"Zone removed: {len(indices_to_remove)} triangles. Total remaining: {len(self.selected_faces_indices)}")
            self.update_selection_colors()

    def _get_triangles_in_rect(self, rect):
        """Moteur mathématique neutre qui renvoie les index des triangles touchant le rectangle."""
        if self.mesh_visual is None: return []

        # On garde la partie VisPy ici (car elle a besoin de la scène/caméra)
        transform = self.mesh_visual.get_transform(map_from='visual', map_to='canvas')
        screen_verts_4d = transform.map(self.original_vertices)

        n_rect = rect.normalized()
        x_min, x_max = n_rect.left(), n_rect.right()
        y_min, y_max = n_rect.top(), n_rect.bottom()

        # On délègue les maths lourdes à l'usine
        return find_triangles_in_rect(
            screen_verts_4d,
            self.original_faces,
            x_min, x_max, y_min, y_max
        )

    # =========================================================================
    #
    # 6. ENSEMBLE DES OUTILS MÉTIER 3D
    #
    # =========================================================================

    # --- Sélection Cylindrique ---
    def toggle_cylinder_tool(self):
        if self.mesh_visual is None:
            self.btn_toggle_cylinder.setChecked(False)
            return

        self.cylinder_active = self.btn_toggle_cylinder.isChecked()

        if self.cylinder_active:
            self.cancel_selection()

            # 1. On "plante" le centre du cylindre là où regarde la caméra
            pivot_x, pivot_y, pivot_z = self.view.camera.center
            self.cylinder_center = (pivot_x, pivot_y)

            # 2. Récupération des valeurs textuelles (ou valeurs par défaut)
            try:
                radius = float(self.radius_input.text())
                base_alt = float(self.base_alt_input.text())
                pivot_alt = float(self.pivot_alt_input.text())
            except ValueError:
                radius, base_alt, pivot_alt = self.CYL_RADIUS, 0.0, 0.0
                self.radius_input.setText(str(self.CYL_RADIUS))

            self._cyl_slider_center_z = base_alt

            # 3. Création du maillage du Disque Vert
            segments = 64
            angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
            verts = [[pivot_x, pivot_y, base_alt]]
            for angle in angles:
                x = pivot_x + radius * np.cos(angle)
                y = pivot_y + radius * np.sin(angle)
                verts.append([x, y, base_alt])

            self.cylinder_verts = np.array(verts, dtype=np.float32)
            faces = [[0, i, i + 1] for i in range(1, segments)]
            faces.append([0, segments, 1])
            self.cylinder_faces = np.array(faces, dtype=np.uint32)

            self.cylinder_visual = scene.visuals.Mesh(
                vertices=self.cylinder_verts,
                faces=self.cylinder_faces,
                color=(1.0, 0.5, 0.0, 0.5),
                shading=None,
                parent=self.view.scene
            )

            # 4. Activation de l'UI
            self.cylinder_radius_slider.setEnabled(True)
            self.cylinder_z_slider.setEnabled(True)
            self.btn_apply_sel.setEnabled(True)
            self.btn_cancel_cyl.setEnabled(True)

            self.cylinder_radius_slider.blockSignals(True)
            self.cylinder_radius_slider.setValue(int(radius))
            self.cylinder_radius_slider.blockSignals(False)

            self.cylinder_z_slider.blockSignals(True)
            z_offset = int(pivot_alt - base_alt)
            self.cylinder_z_slider.setValue(max(0, min(1000, z_offset)))
            self.cylinder_z_slider.blockSignals(False)

            self.btn_toggle_cylinder.setText(_("txt_deactiv_selection"))
            self.update_cylinder_geometry()

            # self.pulse_timer.start(500)
            self.group_cylinder.setStyleSheet("""
                QGroupBox {
                    border: 2px solid #e67e22;
                    margin-top: 2ex;
                    padding-top: 10px;
                    border-radius: 5px;
                }
                QGroupBox::title {
                    color: #e67e22;
                }
            """)

            logging.info(f"Cylinder tool activated. Center defined. Z={base_alt:.1f}m, R={radius:.1f}m")
        else:
            self.cancel_cylinder()

    def update_cylinder_geometry(self):
        """Met à jour les sommets du disque vert sans le recréer."""
        if not self.cylinder_active or self.cylinder_visual is None: return
        try:
            radius = float(self.radius_input.text())
            z = float(self.base_alt_input.text())
        except ValueError:
            return

        cx, cy = self.cylinder_center
        segments = 64
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)

        self.cylinder_verts[0] = [cx, cy, z]
        for i, angle in enumerate(angles):
            self.cylinder_verts[i+1] = [cx + radius * np.cos(angle), cy + radius * np.sin(angle), z]

        self.cylinder_visual.set_data(vertices=self.cylinder_verts, faces=self.cylinder_faces)

    def on_cylinder_radius_slider_changed(self, value):
        self.radius_input.setText(str(value))
        self.update_cylinder_geometry()

    def on_cylinder_z_slider_changed(self, value):
        try:
            pivot_alt = float(self.pivot_alt_input.text())
            # Plus 'value' est grand (à droite), plus on descend sous le pivot
            new_z = pivot_alt - value
            self.base_alt_input.setText(f"{new_z:.1f}")
            self.update_cylinder_geometry()
        except ValueError:
            pass

    def on_cylinder_text_edited(self):
        """Quand l'utilisateur tape manuellement dans les champs, on met à jour les sliders et le visuel."""
        if not self.cylinder_active: return
        try:
            radius = float(self.radius_input.text())
            base_alt = float(self.base_alt_input.text())
            pivot_alt = float(self.pivot_alt_input.text())

            self.cylinder_radius_slider.blockSignals(True)
            self.cylinder_radius_slider.setValue(int(radius))
            self.cylinder_radius_slider.blockSignals(False)

            self.cylinder_z_slider.blockSignals(True)
            z_offset = int(pivot_alt - base_alt)
            self.cylinder_z_slider.setValue(max(0, min(1000, z_offset)))
            self.cylinder_z_slider.blockSignals(False)

            self.update_cylinder_geometry()
        except ValueError:
            pass

    def apply_selection(self):
        """Calcule et colore en orange les triangles dans le cylindre."""
        if self.mesh_visual is None or not hasattr(self, 'original_faces'):
            return

        try:
            # On utilise le centre figé lors de l'activation, ou la caméra par défaut
            if getattr(self, 'cylinder_active', False):
                pivot_x, pivot_y = self.cylinder_center
            else:
                pivot_x, pivot_y = self.view.camera.center[:2]

            radius = float(self.radius_input.text())
            base_alt = float(self.base_alt_input.text())
        except ValueError:
            QMessageBox.warning(self, _("msg_error_title"), _("msg_valid_ray_alt"))
            return

        new_indices = get_faces_in_cylinder(
            self.original_vertices, self.original_faces,
            pivot_x, pivot_y, radius, base_alt
        )
        new_indices = [idx for idx in new_indices if (self.original_tri_types[idx] & 3) == 0]

        current_selection = set(self.selected_faces_indices)
        current_selection.update(new_indices)
        self.selected_faces_indices = list(current_selection)

        self.update_selection_colors()
        logging.info(f"Cylinder applied. Total selected : {len(self.selected_faces_indices)} triangles.")

        # On quitte l'outil proprement pour voir le résultat sans le disque vert par-dessus
        self.cancel_cylinder()
        self.canvas.native.setFocus()
        self.new_selection = True

    def cancel_cylinder(self):
        if self.cylinder_visual is not None:
            self.cylinder_visual.parent = None
            self.cylinder_visual = None

        self.cylinder_active = False
        self.btn_toggle_cylinder.setChecked(False)
        self.btn_toggle_cylinder.setText(_("txt_activ_selection"))

        self.cylinder_radius_slider.setEnabled(False)
        self.cylinder_z_slider.setEnabled(False)
        self.btn_apply_sel.setEnabled(False)
        self.btn_cancel_cyl.setEnabled(False)

        # self.btn_toggle_cylinder.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; padding: 6px;")
        # self.check_stop_pulse()
        self.group_cylinder.setStyleSheet("")

        self.canvas.native.setFocus()

    # --- Édition de Point ---
    def _init_point_edit_session(self, idx):
        """Méthode de refactorisation : Initialise le proxy visuel et l'UI pour la session."""
        cx, cy, cz = self.original_vertices[idx, :3]
        self.view.camera.center = [cx, cy, cz]

        self._create_point_proxy(cx, cy, cz)

        self.is_editing_point = True
        self._slider_base_z = cz

        self.point_spinbox.blockSignals(True)
        self.point_spinbox.setValue(cz)
        self.point_spinbox.blockSignals(False)

        self.point_slider.blockSignals(True)
        self.point_slider.setValue(0)
        self.point_slider.blockSignals(False)

        self.toggle_point_edit_ui(True)

    @wait_cursor
    def create_point(self, checked=False):
        if self.mesh_visual is None or self.is_editing_point: return

        # 1. On pousse l'état sain dans l'historique global avant toute modification
        self.save_state_to_history(action_name="create_point")

        pivot_x, pivot_y = self.view.camera.center[:2]

        # 2. Topologie : On coupe l'arête
        new_verts, new_uvs, new_faces, new_types, new_levels, new_idx = split_closest_edge(
            self.original_vertices, self.original_uvs, self.original_faces,
            self.original_tri_types, self.original_tri_levels, pivot_x, pivot_y
        )

        # --- VÉRIFICATION GLOBALE (Eau, Routes, Aéroports...) ---
        # On cherche tous les triangles qui utilisent le sommet nouvellement créé
        affected_faces = np.where((new_faces[:, 0] == new_idx) |
                                  (new_faces[:, 1] == new_idx) |
                                  (new_faces[:, 2] == new_idx))[0]

        # Si au moins un de ces triangles n'est pas du terrain brut (type > 0)
        if np.any(new_types[affected_faces] > 0):
            QMessageBox.warning(self, _("msg_prohibited_action"), _("msg_unable_to_create_a_point_"))
            # Sécurité : l'action a échoué, on jette le calcul et on retire l'état de l'historique
            self.undo_stack.pop()
            self.update_undo_redo_buttons()
            return

        self.original_vertices = new_verts
        self.original_uvs = new_uvs
        self.original_faces = new_faces
        self.original_tri_types = new_types
        self.original_tri_levels = new_levels
        self.active_new_vertex_idx = new_idx

        self.mesh_visual.set_data(vertices=self.original_vertices, faces=self.original_faces)

        # 3. Lancement de la session interactive via le Proxy
        self._init_point_edit_session(new_idx)
        logging.info("Point created on the edge. Use Z slider to adjust.")

    def edit_point(self):
        if self.mesh_visual is None or self.is_editing_point: return

        # 1. On pousse l'état sain dans l'historique global avant toute modification
        self.save_state_to_history(action_name="edit_point")

        pivot_x, pivot_y = self.view.camera.center[:2]
        dist_sq = np.sum((self.original_vertices[:, :2] - [pivot_x, pivot_y])**2, axis=1)
        closest_idx = np.argmin(dist_sq)

        # --- VÉRIFICATION GLOBALE (Eau, Routes, Aéroports...) ---
        # Utilisation de la nouvelle méthode centralisée (qui protège tout type > 0)
        is_protected_vert = self.get_protected_vertices_mask(bitmask_filter=3)

        if is_protected_vert[closest_idx]:
            QMessageBox.warning(self, _("msg_prohibited_action"), _("msg_this_point_belongs_to_the"))
            # Sécurité : l'action a échoué, on retire l'état de la pile historique
            self.undo_stack.pop()
            self.update_undo_redo_buttons()
            return

        self.active_new_vertex_idx = closest_idx

        # 2. Lancement de la session interactive via le Proxy
        self._init_point_edit_session(closest_idx)
        logging.info(f"Existing point #{closest_idx} selected. Use Z slider to adjust.")

    @wait_cursor
    def validate_point(self, checked=False):
        """Applique l'altitude finale du proxy au maillage réel et valide l'action."""
        if not self.is_editing_point or self.active_new_vertex_idx is None: return

        # L'état initial (avant modification) est déjà au sommet de l'undo_stack.
        # Nous n'avons rien à sauvegarder de plus ici !

        # 1. Application physique de l'altitude du Proxy au Mesh réel
        self.original_vertices[self.active_new_vertex_idx, 2] = self.point_proxy_z
        self.mesh_visual.set_data(vertices=self.original_vertices, faces=self.original_faces)

        # 2. Nettoyage de la session
        self._destroy_point_proxy()

        self.is_editing_point = False
        self.active_new_vertex_idx = None
        self.toggle_point_edit_ui(False)
        self.is_modified = True

        self.update_selection_colors()
        self.canvas.native.setFocus()
        self.update_pivot_z()

        logging.info(f"Point validated. New altitude : {self.point_proxy_z:.1f}m.")

    def cancel_point(self):
        """Annule l'opération en cours et restaure l'état initial via la pile Undo."""
        if not self.is_editing_point or not self.undo_stack: return

        # 1. On récupère l'état initial (avant création/édition) mis dans la pile et on le restaure
        entry = self.undo_stack.pop()
        self.restore_state(entry['data'])

        # 2. Nettoyage de la session
        self._destroy_point_proxy()

        self.is_editing_point = False
        self.active_new_vertex_idx = None
        self.toggle_point_edit_ui(False)
        self.update_undo_redo_buttons()
        self.update_pivot_z()

        logging.info("Operation canceled, mesh restored via global undo stack.")
        self.canvas.native.setFocus()

    def toggle_point_edit_ui(self, is_editing):
        """Bascule l'état des boutons pour forcer la machine à état."""
        self.btn_create_point.setEnabled(not is_editing)
        self.btn_edit_point.setEnabled(not is_editing)
        self.point_slider.setEnabled(is_editing)
        self.point_spinbox.setEnabled(is_editing)
        self.btn_validate_point.setEnabled(is_editing)
        self.btn_cancel_point.setEnabled(is_editing)

        if is_editing:
            self.group_point_edit.setStyleSheet("""
                QGroupBox {
                    border: 2px solid #e67e22;
                    margin-top: 2ex;
                    padding-top: 10px;
                    border-radius: 5px;
                }
                QGroupBox::title {
                    color: #e67e22;
                }
            """)
        else:
            self.group_point_edit.setStyleSheet("")

    def _create_point_proxy(self, x, y, z):
        """Génère le repère visuel (Marqueur + Ligne de déplacement) pour le point en cours d'édition."""
        self.point_proxy_x = x
        self.point_proxy_y = y
        self.point_proxy_orig_z = z  # On mémorise l'altitude exacte de départ
        self.point_proxy_z = z

        # Au départ, la ligne relie l'origine à elle-même (avec un micro-décalage pour éviter un bug VisPy sur les lignes de taille 0)
        line_coords = np.array([[x, y, z], [x, y, z + 0.001]])

        self.point_proxy_line = scene.visuals.Line(pos=line_coords, color='cyan', width=2, parent=self.view.scene)
        # On désactive le test de profondeur pour voir le trait même s'il passe sous le maillage
        self.point_proxy_line.set_gl_state(depth_test=False)

        self.point_proxy_marker = scene.visuals.Markers(parent=self.view.scene)
        self.point_proxy_marker.set_data(pos=np.array([[x, y, z]]), face_color='yellow', edge_color='black', size=15)
        self.point_proxy_marker.set_gl_state(depth_test=False)

    def _update_point_proxy_z(self, new_z):
        """Met à jour la position du proxy visuel et tire un trait depuis l'origine."""
        self.point_proxy_z = new_z

        # La ligne est tendue entre l'altitude d'origine (fixe) et la nouvelle altitude (mobile)
        self.point_proxy_line.set_data(pos=np.array([
            [self.point_proxy_x, self.point_proxy_y, self.point_proxy_orig_z],
            [self.point_proxy_x, self.point_proxy_y, new_z]
        ]))

        self.point_proxy_marker.set_data(pos=np.array([[self.point_proxy_x, self.point_proxy_y, new_z]]), face_color='yellow', edge_color='black', size=15)

    def _destroy_point_proxy(self):
        """Détruit les éléments visuels du proxy."""
        if getattr(self, 'point_proxy_line', None):
            self.point_proxy_line.parent = None
            self.point_proxy_line = None
        if getattr(self, 'point_proxy_marker', None):
            self.point_proxy_marker.parent = None
            self.point_proxy_marker = None

    def on_point_slider_changed(self, value):
        if not self.is_editing_point: return

        offset = value # / 10.0 # -1000 à 1000 -> -100.0m à +100.0m
        new_z = self._slider_base_z + offset

        self.point_spinbox.blockSignals(True)
        self.point_spinbox.setValue(new_z)
        self.point_spinbox.blockSignals(False)

        self._update_point_proxy_z(new_z)

    def on_point_spinbox_changed(self, value):
        if not self.is_editing_point: return
        self._update_point_proxy_z(value)

        # On recentre le slider pour lui redonner du débattement
        self._slider_base_z = value
        self.point_slider.blockSignals(True)
        self.point_slider.setValue(0)
        self.point_slider.blockSignals(False)

    # --- Topologie & Relief (Subdivision, Lissage) ---

    def _do_subdivide(self):
        """Noyau mathématique de la subdivision (sans UI ni historique)."""
        safe_indices = [idx for idx in self.selected_faces_indices if self.original_tri_types[idx] == 0]
        if not safe_indices:
            return 0

        new_v, new_u, new_f, new_t, new_sel, cuts, new_l = subdivide_mesh_selection(
            self.original_vertices, self.original_uvs, self.original_faces,
            self.original_tri_types, safe_indices, self.original_tri_levels,
            max_subdiv_level = self.MAX_SUBDIV_LEVEL
        )

        if cuts > 0:
            self.original_vertices, self.original_uvs = new_v, new_u
            self.original_faces, self.original_tri_types = new_f, new_t
            self.selected_faces_indices = new_sel
            self.original_tri_levels = new_l

        return cuts

    def _do_smooth(self):
        """Noyau mathématique du lissage (sans UI ni historique)."""
        safe_indices = [idx for idx in self.selected_faces_indices if self.original_tri_types[idx] == 0]
        if not safe_indices:
            return False

        sel_faces = self.original_faces[safe_indices]
        unique_verts = np.unique(sel_faces)
        self.last_smooth_max_z = np.max(self.original_vertices[unique_verts, 2])

        # 1. On mémorise la position de tous les sommets avant lissage
        orig_verts = self.original_vertices.copy()
        protected_mask = self.get_protected_vertices_mask() # Par défaut, protège tout (Type > 0)

        # 2. On applique le lissage
        self.original_vertices = apply_cotangent_smooth(
            self.original_vertices, self.original_faces,
            safe_indices,
            iterations=self.SMOOTH_ITER,
            factor=self.SMOOTH_ALPHA,
            feather_radius=self.SMOOTH_FEATH_RAD
        )

        # 3. Z-FREEZE : On restaure brutalement la position des sommets protégés
        self.original_vertices[protected_mask] = orig_verts[protected_mask]

        return True

    def _do_compensation(self):
        """Applique la compensation mathématique pour retrouver l'altitude perdue après un lissage."""
        if not self.selected_faces_indices:
            return

        # 1. Identifier les sommets concernés
        sel_faces = self.original_faces[self.selected_faces_indices]
        unique_verts = np.unique(sel_faces)

        # --- Z-FREEZE TOTAL (Utilisation du masque centralisé) ---
        protected_mask = self.get_protected_vertices_mask()
        # On trouve les index des sommets qui sont protégés
        protected_verts = np.where(protected_mask)[0]

        # On soustrait ces sommets de notre liste de travail
        unique_verts = np.setdiff1d(unique_verts, protected_verts)

        if len(unique_verts) == 0: return
        # --------------------

        z_values = self.original_vertices[unique_verts, 2]

        # 2. Calculer les paramètres actuels
        z_min = np.min(z_values)
        z_max = np.max(z_values)

        # Delta pour que le sommet retrouve son altitude initiale
        delta_z = getattr(self, 'last_smooth_max_z', z_max) - z_max

        # 3. Appliquer le coefficient (Z - Zmin) / (Zmax - Zmin)
        # Éviter la division par zéro si la zone est parfaitement plate
        if delta_z > 0:
            range_z = (z_max - z_min)
            coeffs = np.ones_like(z_values) if range_z < 0.1 else (z_values - z_min) / range_z
            self.original_vertices[unique_verts, 2] += delta_z * coeffs

    @wait_cursor
    def apply_all_in_one(self, checked=False, undo_first=False):
        if self.mesh_visual is None or len(self.selected_faces_indices) == 0:
            QMessageBox.information(self, _("msg_info_title"), _("msg_plz_select_area"))
            return

        # --- 0. GESTION DU RETRY & SAUVEGARDE ---
        if undo_first:
            # On vérifie qu'il y a bien un historique ET que la dernière action est bien la nôtre
            if self.undo_stack and self.undo_stack[-1].get('name') == "all_in_one" and not self.new_selection:
                logging.info("Valid Retry: Restoring previous state before applying new FBM.")
                # On utilise [-1] pour LIRE sans DÉTRUIRE l'état de la pile !
                previous_state = self.undo_stack[-1]['data']
                self.restore_state(previous_state)
            else:
                # Sécurité : Si l'utilisateur clique sur "Undo+Relief" mais que sa dernière
                # action était "Subdivide", on bloque l'Undo et on fait un Relief normal.
                logging.warning("Cannot Retry: Last action was not 'all_in_one' or another selection. Applying standard relief.")
                self.save_state_to_history(action_name="all_in_one")
        else:
            # Traitement normal : on sauvegarde l'état avec notre étiquette
            self.save_state_to_history(action_name="all_in_one")

        logging.info("Starting All-in-One process...")

        # --- 1. CAPTURE DYNAMIQUE DE L'UI ---
        smooth_level = self.slider_smooth.value()
        chaos_active = self.group_chaos_params.isChecked()

        # --- 2. PRÉPARATION DU BASE MESH (Boucle de lissage) ---
        if smooth_level > 0:
            for step in range(smooth_level):
                logging.info(f"--- Loop #{step + 1}/{smooth_level} ---")

                # A. SUBDIVISION
                self._do_subdivide()

                # B. LISSAGE
                self._do_smooth()

                # C. COMPENSATION
                self._do_compensation()

                # D. RAFRAÎCHISSEMENT VISUEL INTERMÉDIAIRE
                if not undo_first:
                    self.is_modified = True
                    self.update_selection_colors()
                    self.update_pivot_z()
                    QApplication.processEvents()

        # --- 3. APPLICATION DU CHAOS / RELIEF FRACTAL (FBM) ---
        if chaos_active:
            extra_subdiv = self.slider_precision.value()
            octaves = self.slider_octaves.value()
            amplitude = float(self.slider_amplitude.value())
            scale = float(self.slider_scale.value())

            logging.info(f"Starting FBM (Subdiv passes: {extra_subdiv}, Octaves: {octaves}, Amp: {amplitude}, Scale: {scale})...")

            # Passes de subdivisions supplémentaires dédiées au bruit
            if extra_subdiv > 0:
                logging.info(f"FBM additional subdivision ({extra_subdiv} passes)...")
                for step in range(extra_subdiv):
                    self._do_subdivide()

            # Application de l'algorithme Fractal Brownian Motion
            logging.info("Application of Fractal Brownian Motion algorithm...")

            orig_verts = self.original_vertices.copy()
            protected_mask = self.get_protected_vertices_mask()

            self.original_vertices = apply_fbm_noise_to_selection(
                self.original_vertices,
                self.original_faces,
                self.selected_faces_indices,
                self.original_tri_types,
                amplitude=amplitude,
                octaves=octaves,
                scale=scale,
                feather_radius = self.SMOOTH_FEATH_RAD
            )

            self.original_vertices[protected_mask] = orig_verts[protected_mask]

            logging.info("FBM relief successfully generated")

        # Synchronisation de fin
        self.is_modified = True
        self.update_selection_colors()
        self.update_pivot_z()
        self.canvas.native.setFocus()
        self.new_selection = False

        logging.info("All-in-One process successfully completed !")

    def apply_fbm_preset(self, name):
        """Applique les valeurs d'un préréglage sans déclencher le mode Custom."""
        if name == "Custom":
            self.update_preset_save_button_state()
            return

        if not hasattr(self, 'relief_presets_data') or name not in self.relief_presets_data:
            return

        self._is_loading_preset = True  # Verrou pour bloquer la boucle de signal

        preset = self.relief_presets_data[name]

        self.slider_smooth.setValue(int(preset.get("smooth", 2)))
        self.group_chaos_params.setChecked(bool(preset.get("chaos_on", False)))
        self.slider_precision.setValue(int(preset.get("subdiv", 0)))
        self.slider_octaves.setValue(int(preset.get("octaves", 1)))
        self.slider_amplitude.setValue(int(preset.get("amp", 10)))
        self.slider_scale.setValue(int(preset.get("scale", 10)))

        self._is_loading_preset = False
        self.update_preset_save_button_state()

    def on_preset_changed(self, text):
        """Déclenché par la liste déroulante des presets."""
        self.apply_fbm_preset(text)

    def on_user_changed_slider(self):
        """Bascule la combobox en mode 'Custom' si l'utilisateur manipule un slider."""
        if not getattr(self, '_is_loading_preset', False):
            self.combo_presets.blockSignals(True)
            self.combo_presets.setCurrentText("Custom")
            self.combo_presets.blockSignals(False)
            self.update_preset_save_button_state()

    def on_chaos_group_toggled(self, checked):
        """Bascule la combobox en mode 'Custom' si la case Chaos globale change."""
        if not getattr(self, '_is_loading_preset', False):
            self.combo_presets.blockSignals(True)
            self.combo_presets.setCurrentText("Custom")
            self.combo_presets.blockSignals(False)
            self.update_preset_save_button_state()

    def load_relief_presets(self):
        """Charge les presets de relief depuis le fichier JSON global et remplit la combobox."""
        os.makedirs("custom", exist_ok=True)
        self.presets_json_file = os.path.join("custom", "relief_presets.json")

        # Valeurs par défaut si le fichier n'existe pas encore
        default_presets = {
            "Smooth": {"smooth": int(self.SMOOTH_LOOPS), "chaos_on": False, "subdiv": 0, "octaves": 1, "amp": 10, "scale": 10},
            "Field": {"smooth": int(self.SMOOTH_LOOPS_LOW), "chaos_on": True, "subdiv": int(self.FBM_PRECISION_LOW), "octaves": int(self.FBM_OCTAVES_LOW), "amp": int(self.FBM_AMPLITUDE_LOW), "scale": int(self.FBM_SCALE_LOW)},
            "Bumpy": {"smooth": int(self.SMOOTH_LOOPS_MID), "chaos_on": True, "subdiv": int(self.FBM_PRECISION_MID), "octaves": int(self.FBM_OCTAVES_MID), "amp": int(self.FBM_AMPLITUDE_MID), "scale": int(self.FBM_SCALE_MID)},
            "Rocky": {"smooth": int(self.SMOOTH_LOOPS_HI), "chaos_on": True, "subdiv": int(self.FBM_PRECISION_HI), "octaves": int(self.FBM_OCTAVES_HI), "amp": int(self.FBM_AMPLITUDE_HI), "scale": int(self.FBM_SCALE_HI)},
            "Crest": {"smooth": int(self.SMOOTH_LOOPS_HI), "chaos_on": True, "subdiv": int(self.FBM_PRECISION_HI), "octaves": int(self.FBM_OCTAVES_HI), "amp": int(self.FBM_AMPLITUDE_HI), "scale": int(self.FBM_SCALE_HI2)}
        }

        self.relief_presets_data = {}
        if os.path.exists(self.presets_json_file):
            try:
                with open(self.presets_json_file, 'r', encoding='utf-8') as f:
                    self.relief_presets_data = json.load(f)
            except Exception as e:
                logging.error(f"Impossible to read relief_presets.json : {e}")

        # Si le fichier était vide ou corrompu, on réinjecte les builds d'origine
        if not self.relief_presets_data:
            self.relief_presets_data = default_presets
            self.save_presets_to_json_file()

        # Remplissage dynamique de la ComboBox
        self.combo_presets.blockSignals(True)
        self.combo_presets.clear()

        # On ajoute d'abord les presets enregistrés
        for name in self.relief_presets_data.keys():
            self.combo_presets.addItem(name, name)

        # On ajoute impérativement l'option "Custom" à la fin
        self.combo_presets.addItem("Custom", "Custom")
        self.combo_presets.blockSignals(False)

    def save_presets_to_json_file(self):
        """Enregistre le dictionnaire actuel des presets dans le fichier JSON."""
        try:
            with open(self.presets_json_file, 'w', encoding='utf-8') as f:
                json.dump(self.relief_presets_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error while writing in relief_presets.json : {e}")

    def save_custom_relief_preset(self):
        """Demande un nom à l'utilisateur et enregistre la configuration actuelle de l'UI dans le JSON."""
        name, ok = QInputDialog.getText(self, _("dialog_save_preset"), _("dialog_preset_name"))
        if not ok or not name.strip():
            return
        name = name.strip()

        # Protections de sécurité élémentaires
        if name == "Custom":
            QMessageBox.warning(self, _("msg_error_title"), _("msg_custom_preset"))
            return

        # Vérification d'écrasement si le nom existe déjà
        if name in self.relief_presets_data:
            reply = QMessageBox.question(self, _("msg_confirmation"), _("msg_relief_preset_exist", cn=name),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # Capture des valeurs de l'UI
        self.relief_presets_data[name] = {
            "smooth": self.slider_smooth.value(),
            "chaos_on": self.group_chaos_params.isChecked(),
            "subdiv": self.slider_precision.value(),
            "octaves": self.slider_octaves.value(),
            "amp": self.slider_amplitude.value(),
            "scale": self.slider_scale.value()
        }

        # Écriture disque et rechargement de l'UI
        self.save_presets_to_json_file()
        self.load_relief_presets()

        # On bascule automatiquement la liste sur le préréglage fraîchement créé
        self.combo_presets.setCurrentText(name)
        QMessageBox.information(self, _("msg_success_title"), _("msg_relief_preset_saved",cn=name))

    def update_preset_save_button_state(self):
        """Active ou désactive le bouton de sauvegarde selon le preset affiché."""
        if hasattr(self, 'btn_save_preset') and hasattr(self, 'combo_presets'):
            is_custom = (self.combo_presets.currentText() == "Custom")
            self.btn_save_preset.setEnabled(is_custom)

    # --- Zone Complexe & Aplanissement ---

    def toggle_zone_mode(self):
        self.zone_draw_mode = self.btn_toggle_zone.isChecked()
        self.flatten_trans_input.setEnabled(self.zone_draw_mode)

        if self.zone_draw_mode:
            self.group_tilted_plane.setEnabled(True)
            self.btn_toggle_zone.setText(_("txt_deactiv_area"))
            self.cancel_selection()
            self.pulse_timer.start(500)
            logging.info("Area Drawing Mode activated. Tabs locked.")

            # Verrouiller tous les onglets sauf "Airport" (Index 2)
            for i in range(self.tabs.count()):
                if i != 2:
                    self.tabs.setTabEnabled(i, False)
        else:
            self.btn_toggle_zone.setText(_("txt_activ_area"))

            # Sécurité : Si on quitte l'outil alors que l'étape 2 (Aplanissement) est en attente
            if getattr(self, 'flatten_active', False):
                # Restauration forcée de la topologie
                if self.undo_stack:
                    previous_state = self.undo_stack.pop()
                    self.restore_state(previous_state['data'])
                    self.redo_stack.clear()
                    self.update_undo_redo_buttons()
                    self.update_pivot_z()

                self._cleanup_flatten_ui()

            self.cancel_zone()

            # Déverrouiller les onglets
            for i in range(self.tabs.count()):
                self.tabs.setTabEnabled(i, True)

        self.canvas.native.setFocus()

    def update_zone_polygon_visual(self):
        """Affiche le polygone en cours de tracé en 3D (cyan, losanges)."""
        if not self.zone_polygon_points:
            if getattr(self, 'zone_polygon_visual', None): self.zone_polygon_visual.visible = False
            if getattr(self, 'zone_markers_visual', None): self.zone_markers_visual.visible = False
            return

        pts_3d = np.array(self.zone_polygon_points)
        # On surélève légèrement pour éviter le Z-fighting avec le maillage
        pts_3d[:, 2] += 15.0

        # 1. Tracé cyan de la bordure
        if len(pts_3d) >= 2:
            if getattr(self, 'zone_polygon_visual', None) is None:
                self.zone_polygon_visual = scene.visuals.Line(pos=pts_3d, color='cyan', width=3, parent=self.view.scene)
                self.zone_polygon_visual.set_gl_state('translucent', depth_test=False)
                self.zone_polygon_visual.order = 10
            else:
                self.zone_polygon_visual.set_data(pos=pts_3d, color='cyan')
                self.zone_polygon_visual.visible = True
        else:
            if getattr(self, 'zone_polygon_visual', None): self.zone_polygon_visual.visible = False

        # 2. Marqueurs (losanges)
        if getattr(self, 'zone_markers_visual', None) is None:
            self.zone_markers_visual = scene.visuals.Markers(parent=self.view.scene)
            self.zone_markers_visual.set_gl_state('translucent', depth_test=False)
            self.zone_markers_visual.order = 11

        self.zone_markers_visual.set_data(pos=pts_3d, symbol='diamond', edge_color='cyan', face_color='white', size=8)
        self.zone_markers_visual.visible = True

        # Un tracé valide nécessite au moins un triangle fermé (4 points dont le 1er == le dernier)
        is_closed = len(self.zone_polygon_points) >= 4 and (self.zone_polygon_points[0] == self.zone_polygon_points[-1])

        # Activation stricte des boutons
        self.btn_apply_zone.setEnabled(is_closed)
        self.btn_save_zone.setEnabled(is_closed)

        # L'annulation reste disponible dès le 1er point posé
        self.btn_cancel_zone.setEnabled(len(pts_3d) > 0)

    @wait_cursor
    def apply_zone_cut_ui(self, checked=False):
        if len(self.zone_polygon_points) < 3: return

        if self.group_tilted_plane.isChecked() and len(getattr(self, 'tilted_pts_2d', [])) < 2:
            QMessageBox.warning(self, "Warning", _("msg_plz_define_AB"))
            return

        try:
            subdivs = self.slider_area_subdiv.value()
            trans_width = float(self.flatten_trans_input.text())
        except ValueError:
            subdivs = 2
            trans_width = 100.0

        # =====================================================================
        logging.info(f"0/5 : Check out of bounds...")
        # =====================================================================
        # 1. Extraction rapide des points 2D
        raw_pts_2d = [np.array(pt[:2]) for pt in self.zone_polygon_points]
        raw_pts_arr = np.array(raw_pts_2d)

        # 2. Calcul de la Bounding Box étendue (Polygone + Talus + Marge de sécurité)
        margin = trans_width + 10.0
        min_x = np.min(raw_pts_arr[:, 0]) - margin
        max_x = np.max(raw_pts_arr[:, 0]) + margin
        min_y = np.min(raw_pts_arr[:, 1]) - margin
        max_y = np.max(raw_pts_arr[:, 1]) + margin

        # 3. Les 4 coins extrêmes de l'emprise des travaux
        poly_max_extent = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y)
        ]

        # 4. Test de sécurité
        for corner in poly_max_extent:
            if not self.is_xy_strictly_inside_mesh(corner[0], corner[1]):
                QMessageBox.warning(self, _("msg_action_denied"), _("msg_operation_canceled_the_ro"))
                # On quitte sans détruire le tracé de l'utilisateur,
                # ce qui lui laisse la chance d'annuler son dernier point avec Ctrl+Z
                return

        # =====================================================================
        logging.info(f"1/5 : Preparation and calculation of exact altitudes...")
        # =====================================================================
        self.save_state_to_history()

        # 1. Extraction des points 2D (On ignore le Z de visualisation)
        raw_pts_2d = [np.array(pt[:2]) for pt in self.zone_polygon_points]
        if np.linalg.norm(raw_pts_2d[0] - raw_pts_2d[-1]) < 0.1:
            raw_pts_2d = raw_pts_2d[:-1]

        # Bounding Box pour limiter les calculs
        raw_pts_arr = np.array(raw_pts_2d)
        min_x, max_x = np.min(raw_pts_arr[:, 0]) - 200, np.max(raw_pts_arr[:, 0]) + 200
        min_y, max_y = np.min(raw_pts_arr[:, 1]) - 200, np.max(raw_pts_arr[:, 1]) + 200

        grid_xs, grid_ys = get_texture_grid_lines(min_x, max_x, min_y, max_y, 19, self.lon_to_m, self.lat_to_m, self.x_center, self.y_center)

        # Densification 2D
        densified_2d = []
        max_len = 15.0
        for i in range(len(raw_pts_2d)):
            p1, p2 = raw_pts_2d[i], raw_pts_2d[(i+1)%len(raw_pts_2d)]
            dist = np.linalg.norm(p2 - p1)
            num_segments = max(1, int(np.ceil(dist / max_len)))
            for j in range(num_segments):
                densified_2d.append(p1 + (p2 - p1) * (j / num_segments))

        # Injection des intersections de grille
        final_p_user_2d = []
        for i in range(len(densified_2d)):
            p1, p2 = densified_2d[i], densified_2d[(i + 1) % len(densified_2d)]
            final_p_user_2d.append(p1)
            intersections = []
            for gx in grid_xs:
                if min(p1[0], p2[0]) < gx < max(p1[0], p2[0]):
                    intersections.append(p1 + ((gx - p1[0]) / (p2[0] - p1[0])) * (p2 - p1))
            for gy in grid_ys:
                if min(p1[1], p2[1]) < gy < max(p1[1], p2[1]):
                    intersections.append(p1 + ((gy - p1[1]) / (p2[1] - p1[1])) * (p2 - p1))
            if intersections:
                intersections.sort(key=lambda pt: np.linalg.norm(pt - p1))
                final_p_user_2d.extend(intersections)

        # Nettoyage des doublons
        clean_p_user_2d = [final_p_user_2d[0]]
        for pt in final_p_user_2d[1:]:
            if np.linalg.norm(pt - clean_p_user_2d[-1]) > 0.05: clean_p_user_2d.append(pt)
        if np.linalg.norm(clean_p_user_2d[-1] - clean_p_user_2d[0]) < 0.05: clean_p_user_2d.pop()
        P_user_2d = np.array(clean_p_user_2d, dtype=np.float32)

        # Altitude Z réelle du sol
        P_user_3d = np.zeros((len(P_user_2d), 3), dtype=np.float32)
        P_user_3d[:, :2] = P_user_2d

        # 1. OPTIMISATION : Extraction d'une "mini-carte" locale (Gain de perf x100)
        local_v_mask = (self.original_vertices[:, 0] >= min_x) & (self.original_vertices[:, 0] <= max_x) & \
                       (self.original_vertices[:, 1] >= min_y) & (self.original_vertices[:, 1] <= max_y)
        local_verts_indices = np.where(local_v_mask)[0]

        # Indexation directe instantanée
        mask_f_bb = local_v_mask[self.original_faces[:, 0]] | \
                    local_v_mask[self.original_faces[:, 1]] | \
                    local_v_mask[self.original_faces[:, 2]]

        local_faces = self.original_faces[mask_f_bb]
        loc_v = self.original_vertices[local_verts_indices]

        # 2. Calcul localisé pour chaque point
        is_closest = np.zeros(len(self.original_vertices), dtype=bool)
        for i, pt in enumerate(P_user_2d):
            x, y = pt[0], pt[1]

            # Recherche sur les sommets locaux uniquement
            dist_sq = (loc_v[:, 0] - x)**2 + (loc_v[:, 1] - y)**2
            num_candidates = min(50, len(dist_sq))
            closest_local_idx = np.argpartition(dist_sq, num_candidates - 1)[:num_candidates]
            closest_global_idx = local_verts_indices[closest_local_idx]

            # 1. On "allume" nos 50 index
            is_closest[closest_global_idx] = True

            # 2. Indexation booléenne
            mask = is_closest[local_faces[:, 0]] | \
                   is_closest[local_faces[:, 1]] | \
                   is_closest[local_faces[:, 2]]

            # 3. On "éteint" nos 50 index pour laisser la place propre au prochain tour !
            is_closest[closest_global_idx] = False

            candidate_faces = local_faces[mask]

            # Interpolation Barycentrique Exacte
            z_exact = None
            if len(candidate_faces) > 0:
                v0 = self.original_vertices[candidate_faces[:, 0]]
                v1 = self.original_vertices[candidate_faces[:, 1]]
                v2 = self.original_vertices[candidate_faces[:, 2]]

                denom = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
                valid = np.abs(denom) > 1e-8

                if np.any(valid):
                    v0_v, v1_v, v2_v, denom_v = v0[valid], v1[valid], v2[valid], denom[valid]
                    w1 = ((v1_v[:, 1] - v2_v[:, 1]) * (x - v2_v[:, 0]) + (v2_v[:, 0] - v1_v[:, 0]) * (y - v2_v[:, 1])) / denom_v
                    w2 = ((v2_v[:, 1] - v0_v[:, 1]) * (x - v2_v[:, 0]) + (v0_v[:, 0] - v2_v[:, 0]) * (y - v2_v[:, 1])) / denom_v
                    w3 = 1.0 - w1 - w2

                    inside = (w1 >= -1e-4) & (w2 >= -1e-4) & (w3 >= -1e-4)
                    idx = np.where(inside)[0]

                    if len(idx) > 0:
                        first = idx[0]
                        z_exact = w1[first] * v0_v[first, 2] + w2[first] * v1_v[first, 2] + w3[first] * v2_v[first, 2]

            # Assignation ou Fallback rapide
            if z_exact is not None:
                P_user_3d[i, 2] = z_exact
            else:
                # Si on est légèrement hors d'un triangle, interpolation IDW sur les 3 plus proches
                idx_nearest = np.argpartition(dist_sq, min(2, len(dist_sq)-1))[:3]
                dists = np.sqrt(dist_sq[idx_nearest])
                dists[dists < 1e-6] = 1e-6
                weights = 1.0 / (dists ** 2)
                P_user_3d[i, 2] = np.sum(weights * loc_v[idx_nearest, 2]) / np.sum(weights)

        segments_2d = [(P_user_2d[i], P_user_2d[(i+1)%len(P_user_2d)]) for i in range(len(P_user_2d))]

        # =====================================================================
        logging.info("2/5 & 3/5 : Adaptive Topological subdivision & Loop extraction...")
        # =====================================================================

        # 1. SNAPSHOT : On sauvegarde le maillage local pour pouvoir faire des retries propres
        backup_v = self.original_vertices.copy()
        backup_u = self.original_uvs.copy()
        backup_f = self.original_faces.copy()
        backup_t = self.original_tri_types.copy()
        backup_l = self.original_tri_levels.copy() if hasattr(self, 'original_tri_levels') else None

        current_subdivs = subdivs
        success = False
        final_loops = []
        hole_f, hole_t, hole_l = None, None, None

        # 0.28.1
        allow_destroy_all = self.cb_allow_destroy_all_types.isChecked()

        # 2. LA BOUCLE DE RETRY
        while current_subdivs <= self.MAX_SUBDIV_LEVEL:
            logging.info(f"-> Attempting geometry cut with subdivs = {current_subdivs}...")

            # A. Restauration de l'état pur du maillage
            self.original_vertices = backup_v.copy()
            self.original_uvs = backup_u.copy()
            self.original_faces = backup_f.copy()
            self.original_tri_types = backup_t.copy()
            if backup_l is not None: self.original_tri_levels = backup_l.copy()

            # B. La subdivision (bornée par current_subdivs)
            for step in range(current_subdivs):
                v2d = self.original_vertices[:, :2]
                v_in_bb = (v2d[:, 0] >= min_x) & (v2d[:, 0] <= max_x) & (v2d[:, 1] >= min_y) & (v2d[:, 1] <= max_y)

                f_in_bb_mask = v_in_bb[self.original_faces[:, 0]] | \
                               v_in_bb[self.original_faces[:, 1]] | \
                               v_in_bb[self.original_faces[:, 2]]
                cand_f_global_idx = np.where(f_in_bb_mask)[0]
                cand_faces = self.original_faces[cand_f_global_idx]

                v0 = self.original_vertices[cand_faces[:, 0]][:, :2]
                v1 = self.original_vertices[cand_faces[:, 1]][:, :2]
                v2 = self.original_vertices[cand_faces[:, 2]][:, :2]
                cand_centers = (v0 + v1 + v2) / 3.0

                r0 = np.linalg.norm(v0 - cand_centers, axis=1)
                r1 = np.linalg.norm(v1 - cand_centers, axis=1)
                r2 = np.linalg.norm(v2 - cand_centers, axis=1)
                max_radius = np.maximum(np.maximum(r0, r1), r2)

                dist_centers = points_to_segments_dist(cand_centers, segments_2d)
                curr_rad = trans_width * (1.5 - (step / current_subdivs))

                cand_types = self.original_tri_types[cand_f_global_idx]
                cand_levels = self.original_tri_levels[cand_f_global_idx]

                # 0.28.1 - Si l'option est cochée, on autorise tous les types (True), sinon seulement le type 0
                type_condition = True if allow_destroy_all else (cand_types == 0)
                mask_local = ((dist_centers - max_radius) <= curr_rad) & type_condition & \
                             (cand_levels < self.MAX_SUBDIV_LEVEL)

                candidate_faces_idx = cand_f_global_idx[mask_local].tolist()
                if not candidate_faces_idx: break

                v, u, f, t, _dummy, cuts, l = subdivide_mesh_selection(
                    self.original_vertices, self.original_uvs,
                    self.original_faces, self.original_tri_types,
                    candidate_faces_idx, self.original_tri_levels,
                    max_subdiv_level=self.MAX_SUBDIV_LEVEL
                )
                if cuts == 0: break
                self.original_vertices, self.original_uvs, self.original_faces, self.original_tri_types, self.original_tri_levels = v, u, f, t, l

            # C. Extraction de la clearance area
            v2d = self.original_vertices[:, :2]
            mask_v_bb = (v2d[:, 0] >= min_x) & (v2d[:, 0] <= max_x) & (v2d[:, 1] >= min_y) & (v2d[:, 1] <= max_y)
            dist_v = np.full(len(self.original_vertices), np.inf)
            cand_v_idx = np.where(mask_v_bb)[0]
            if len(cand_v_idx) > 0:
                dist_v[cand_v_idx] = points_to_segments_dist(v2d[cand_v_idx], segments_2d)

            faces_to_del = np.any(dist_v[self.original_faces] < 3.0, axis=1)

            f_centers = np.mean(self.original_vertices[self.original_faces][:, :, :2], axis=1)
            mask_f_bb = (f_centers[:, 0] >= min_x) & (f_centers[:, 0] <= max_x) & (f_centers[:, 1] >= min_y) & (f_centers[:, 1] <= max_y)
            cand_f_idx = np.where(mask_f_bb)[0]
            if len(cand_f_idx) > 0:
                intersect_mask = faces_intersecting_polygon(self.original_vertices, self.original_faces[cand_f_idx], P_user_2d)
                faces_to_del[cand_f_idx] |= intersect_mask

            # Sécurité conflits (routes, eau...)
            if not allow_destroy_all and np.any(self.original_tri_types[faces_to_del] > 0):
                logging.warning("Type conflict detected in area to delete.")
                break # Échec total, on arrête les tentatives

           # D. Test des boucles Topologiques
            del_f = self.original_faces[faces_to_del]
            loops, is_valid = self.get_all_loops(del_f)

            # E. Verdict de la tentative : La vision de l'anneau (2 boucles strictes)
            if is_valid and len(loops) == 2:
                logging.info(f"Success ! Clean topology (2 loops) achieved with subdivs = {current_subdivs}.")
                success = True
                final_loops = loops
                hole_f = self.original_faces[~faces_to_del]
                hole_t = self.original_tri_types[~faces_to_del]
                hole_l = self.original_tri_levels[~faces_to_del]
                break # On sort de la boucle while, tout est parfait !
            else:
                logging.info(f"Topology failed (Valid: {is_valid}, Loops count: {len(loops)}). Incrementing subdivs.")
                current_subdivs += 1

        # 3. GESTION DE L'ÉCHEC ABSOLU
        if not success:
            QMessageBox.warning(self, _("msg_action_denied"), "Topology issue, see log")

            # Rollback avec l'undo stack
            if self.undo_stack:
                previous_state = self.undo_stack.pop()
                self.restore_state(previous_state['data'])
                self.redo_stack.clear()
                self.update_undo_redo_buttons()
                self.update_pivot_z()

            self.btn_toggle_zone.setChecked(False)
            self.toggle_zone_mode()
            return

        # =====================================================================
        # Calcul UVs par IDW (accéléré)
        # =====================================================================
        local_v_mask = (self.original_vertices[:, 0] >= min_x) & (self.original_vertices[:, 0] <= max_x) & \
                       (self.original_vertices[:, 1] >= min_y) & (self.original_vertices[:, 1] <= max_y)
        loc_v = self.original_vertices[local_v_mask]
        loc_uv = self.original_uvs[local_v_mask]
        P_user_uvs = np.zeros((len(P_user_3d), 2), dtype=np.float32)
        for i, pt in enumerate(P_user_3d):
            d2 = (loc_v[:, 0] - pt[0])**2 + (loc_v[:, 1] - pt[1])**2
            idx = np.argpartition(d2, 2)[:3]
            w = 1.0 / (np.sqrt(d2[idx]) + 1e-6)**2
            P_user_uvs[i] = np.sum(w[:, None] * loc_uv[idx], axis=0) / np.sum(w)

        offset = len(self.original_vertices)
        P_user_idx = list(range(offset, offset + len(P_user_3d)))

        # Application finale des matrices saines (issues du succès)
        self.original_vertices = np.vstack((self.original_vertices, P_user_3d))
        self.original_uvs = np.vstack((self.original_uvs, P_user_uvs))
        self.original_faces = hole_f
        self.original_tri_types = hole_t
        self.original_tri_levels = hole_l

        # =====================================================================
        logging.info("4/5 : CDT Triangulation...")
        # =====================================================================
        # On a exactement 2 boucles dans final_loops. Il faut juste les trier.
        for loop in final_loops:
            # On prend 5 points au hasard sur la boucle pour tester
            pts_check = self.original_vertices[loop[:5], :2]

            # Si la majorité de ces points est à l'intérieur du polygone utilisateur
            if np.sum(points_in_polygons_concave(pts_check, [P_user_2d])) > len(pts_check) / 2:
                # C'est la boucle intérieure (le trou du plateau)
                self.do_cdt_local(P_user_idx, loop, grid_xs, grid_ys)
            else:
                # C'est la boucle extérieure
                self.do_cdt_local(loop, P_user_idx, grid_xs, grid_ys)

        # =====================================================================
        logging.info("5/5 : Final selection...")
        # =====================================================================
        c2d = np.mean(self.original_vertices[self.original_faces][:, :, :2], axis=1)
        mask_sel_bb = (c2d[:, 0] >= min_x) & (c2d[:, 0] <= max_x) & (c2d[:, 1] >= min_y) & (c2d[:, 1] <= max_y)
        inside = np.zeros(len(c2d), dtype=bool)
        cand_c = np.where(mask_sel_bb)[0]
        if len(cand_c) > 0:
            inside[cand_c] = points_in_polygons_concave(c2d[cand_c], [P_user_2d])
        candidates = np.where(inside)[0].tolist()

        # self.selected_faces_indices = [idx for idx in candidates if (self.original_tri_types[idx] & 3) == 0]
        # On sélectionne tous les types de triangles, même l'eau
        self.selected_faces_indices = candidates

        self.is_modified = True
        self.update_selection_colors()
        self.canvas.native.setFocus()
        self.update_pivot_z()

        # --- TRANSITION AUTOMATIQUE VERS L'APLANISSEMENT ---
        self.btn_apply_zone.setEnabled(False) # On empêche de recouper
        self.btn_load_zone.setEnabled(False)
        self.btn_save_zone.setEnabled(False)
        self.group_tilted_plane.setEnabled(False)
        self.activate_flatten_mode()

    def get_all_loops(self, f_subset):
        edge_counts = {}
        for f in f_subset:
            edges = [tuple(sorted((f[0], f[1]))), tuple(sorted((f[1], f[2]))), tuple(sorted((f[2], f[0])))]
            for e in edges:
                edge_counts[e] = edge_counts.get(e, 0) + 1

        # 1. Utiliser une liste d'adjacence pour supporter les sommets partagés (pincements)
        adjacency = {}
        for f in f_subset:
            edges = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
            for e in edges:
                if edge_counts[tuple(sorted(e))] == 1:
                    if e[0] not in adjacency:
                        adjacency[e[0]] = []
                    adjacency[e[0]].append(e[1])

        # NOUVEAU : On vérifie si un sommet possède plus d'une arête sortante (Pincement)
        is_topology_valid = True
        for node, targets in adjacency.items():
            if len(targets) > 1:
                is_topology_valid = False
                logging.warning(f"Topological pinch detected at vertex {node} (Degree: {len(targets)}).")
                break

        loops = []

        # 2. Extraction robuste des cycles simples (Ton code intact)
        while adjacency:
            start_node = next(iter(adjacency.keys()))
            path = []
            curr = start_node

            while True:
                path.append(curr)

                if curr not in adjacency or not adjacency[curr]:
                    break

                nxt = adjacency[curr].pop(0)
                if not adjacency[curr]:
                    del adjacency[curr]

                if nxt in path:
                    idx = path.index(nxt)
                    loop = path[idx:]
                    loops.append(loop)

                    path = path[:idx]
                    curr = nxt

                    if not path and (curr not in adjacency or not adjacency[curr]):
                        break
                else:
                    curr = nxt

        # NOUVEAU : On renvoie les boucles ET le statut
        return loops, is_topology_valid

    def do_cdt_local(self, polyA_indices, polyB_indices, grid_xs, grid_ys):
        polyA = self.original_vertices[polyA_indices]
        polyB = self.original_vertices[polyB_indices]
        anchors = compute_anchor_segments(polyA, polyB, grid_xs, grid_ys)

        n_faces, loc_verts = perform_stitching_cdt(polyA, polyB, anchors)

        if n_faces is not None:
            numA = len(polyA_indices)
            numB = len(polyB_indices)
            l2g = np.zeros(len(loc_verts), dtype=int)
            l2g[:numA] = polyA_indices
            l2g[numA:numA+numB] = polyB_indices

            if len(loc_verts) > numA + numB:
                steiner_pts = np.array(loc_verts[numA+numB:], dtype=np.float32)
                start_idx = len(self.original_vertices)
                self.original_vertices = np.vstack((self.original_vertices, steiner_pts))

                st_uvs = []
                for pt in steiner_pts:
                    lat, lon = meters_to_latlon(pt[0], pt[1], self.lon_to_m, self.lat_to_m, self.x_center, self.y_center)
                    st_uvs.append([lon, lat])
                self.original_uvs = np.vstack((self.original_uvs, np.array(st_uvs, dtype=np.float32)))
                l2g[numA+numB:] = np.arange(start_idx, start_idx + len(steiner_pts))

            global_faces = l2g[n_faces]
            self.original_faces = np.vstack((self.original_faces, global_faces))
            self.original_tri_types = np.concatenate([self.original_tri_types, np.zeros(len(global_faces), dtype=np.uint32)])

            if hasattr(self, 'original_tri_levels'):
                self.original_tri_levels = np.concatenate([self.original_tri_levels, np.zeros(len(global_faces), dtype=np.uint8)])

    def undo_zone_point(self):
        """Annule le dernier point placé sur le lasso 3D."""
        if getattr(self, 'zone_draw_mode', False) and self.zone_polygon_points:
            point = self.zone_polygon_points.pop()
            self.zone_polygon_redo_stack.append(point)
            self.update_zone_polygon_visual()

    def redo_zone_point(self):
        """Rétablit le dernier point annulé sur le lasso 3D."""
        if getattr(self, 'zone_draw_mode', False) and getattr(self, 'zone_polygon_redo_stack', []):
            point = self.zone_polygon_redo_stack.pop()
            self.zone_polygon_points.append(point)
            self.update_zone_polygon_visual()

    def activate_flatten_mode(self):
        """Déclenche la phase 2 : Réglage du plateau après la découpe."""
        if not self.selected_faces_indices: return

        self.flatten_active = True

        sel_faces = self.original_faces[self.selected_faces_indices]
        unique_verts = np.unique(sel_faces)
        sel_verts = self.original_vertices[unique_verts]

        cx = np.mean(sel_verts[:, 0])
        cy = np.mean(sel_verts[:, 1])
        mean_z = np.mean(sel_verts[:, 2])

        distances = np.sqrt((sel_verts[:, 0] - cx)**2 + (sel_verts[:, 1] - cy)**2)
        radius = np.max(distances) * 1.05

        segments = 64
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)

        # 0.28.0 --- GESTION DU TYPE DE PLAN ---
        is_tilted = self.group_tilted_plane.isChecked() and len(getattr(self, 'tilted_pts_2d', [])) == 2
        verts = []

        if is_tilted:
            A, B = self.tilted_pts_2d[0], self.tilted_pts_2d[1]
            vec_AB = B[:2] - A[:2]
            len_sq = np.dot(vec_AB, vec_AB)
            if len_sq == 0: len_sq = 1.0

            # Fonction locale de projection Z
            def get_tilted_z(px, py):
                vec_AP = np.array([px, py]) - A[:2]
                t = np.dot(vec_AP, vec_AB) / len_sq
                return A[2] + t * (B[2] - A[2])

            # Sommet central
            center_z = get_tilted_z(cx, cy)
            verts.append([cx, cy, center_z])

            # Sommets périphériques
            for angle in angles:
                vx = cx + radius * np.cos(angle)
                vy = cy + radius * np.sin(angle)
                verts.append([vx, vy, get_tilted_z(vx, vy)])

            plane_color = (0.2, 0.5, 1.0, 0.5) # Bleu acier pour le plan incliné
            self._slider_center_z = center_z
        else:
            verts.append([cx, cy, mean_z])
            for angle in angles:
                verts.append([cx + radius * np.cos(angle), cy + radius * np.sin(angle), mean_z])

            plane_color = (0.0, 1.0, 1.0, 0.5) # Cyan classique pour l'horizontal
            self._slider_center_z = mean_z

        self.flatten_plane_verts = np.array(verts, dtype=np.float32)

        faces = [[0, i, i + 1] for i in range(1, segments)]
        faces.append([0, segments, 1])
        self.flatten_plane_faces = np.array(faces, dtype=np.uint32)

        self.flatten_plane = scene.visuals.Mesh(
            vertices=self.flatten_plane_verts,
            faces=self.flatten_plane_faces,
            color=plane_color,
            shading=None,
            parent=self.view.scene
        )

        # 0.28.0 --- Ajustement de l'UI ---
        if is_tilted:
            # En mode incliné, le Z global n'a plus de sens (A et B commandent)
            self.flatten_slider.setEnabled(False)
            self.flatten_spinbox.setEnabled(False)
        else:
            self.flatten_slider.setEnabled(True)
            self.flatten_spinbox.setEnabled(True)

        self.btn_apply_flatten.setEnabled(True)
        self.btn_cancel_flatten.setEnabled(True)

        self.flatten_spinbox.blockSignals(True)
        self.flatten_slider.blockSignals(True)
        self.flatten_spinbox.setValue(self._slider_center_z)
        self.flatten_slider.setValue(500)
        self.flatten_spinbox.blockSignals(False)
        self.flatten_slider.blockSignals(False)

        logging.info(f"Flattening Step 2 activated. Tilted mode: {is_tilted}")

    def _cleanup_flatten_ui(self):
        """Nettoie l'UI d'aplanissement (sans toucher à la topologie)."""
        if getattr(self, 'flatten_plane', None) is not None:
            self.flatten_plane.parent = None
            self.flatten_plane = None

        self.flatten_active = False
        self.flatten_slider.setEnabled(False)
        self.flatten_spinbox.setEnabled(False)
        self.btn_apply_flatten.setEnabled(False)
        self.btn_cancel_flatten.setEnabled(False)

        self.btn_apply_flatten.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; padding: 6px;")
        self.check_stop_pulse()

    def abort_flattening(self):
        """Bouton 'Cancel flattening' : Annule la découpe et revient au tracé du polygone."""
        self._cleanup_flatten_ui()

        if self.undo_stack:
            previous_state = self.undo_stack.pop()
            self.restore_state(previous_state['data'])
            self.redo_stack.clear()
            self.update_undo_redo_buttons()
            self.update_pivot_z()

        self.btn_apply_zone.setEnabled(True)
        self.btn_load_zone.setEnabled(True)
        self.btn_save_zone.setEnabled(True)
        self.group_tilted_plane.setEnabled(True)
        self.update_zone_polygon_visual()
        self.canvas.native.setFocus()

        logging.info("Flattening process aborted by user. Restoring pre-cut topology state.")

    def on_flatten_slider_changed(self, value):
        """Traduit le mouvement du slider en altitude (1 cran = 0.5m) sans modifier le centre."""
        offset = (value - 500) * 0.5 # Le slider permet +/- 250m autour de la valeur tapée
        new_z = self._slider_center_z + offset

        # Met à jour la SpinBox sans déclencher d'événement en boucle
        self.flatten_spinbox.blockSignals(True)
        self.flatten_spinbox.setValue(new_z)
        self.flatten_spinbox.blockSignals(False)

        self.update_flatten_plane_z(new_z)

    def on_flatten_spinbox_changed(self, value):
        """Quand on tape une valeur, on déplace le plan ET on recentre le slider."""
        self.update_flatten_plane_z(value)

        # Astuce UX : Si l'utilisateur tape une altitude, on met à jour la base du slider
        # pour qu'il ait toujours de la marge pour ajuster (le slider redevient "infini")
        self._slider_center_z = value
        self.flatten_slider.blockSignals(True)
        self.flatten_slider.setValue(500)
        self.flatten_slider.blockSignals(False)

    def update_flatten_plane_z(self, z_value):
        """Déplace les sommets du plan cible sur l'axe Z."""
        if self.flatten_plane is not None and hasattr(self, 'flatten_plane_verts'):
            # On met à jour notre propre tableau de sommets
            self.flatten_plane_verts[:, 2] = z_value

            # On renvoie les sommets ET les faces à VisPy pour éviter qu'il ne se perde
            self.flatten_plane.set_data(vertices=self.flatten_plane_verts,
                                        faces=self.flatten_plane_faces)

    @wait_cursor
    def apply_flatten(self, checked=False):
        if not self.flatten_active or self.flatten_plane is None: return

        try:
            trans_width = float(self.flatten_trans_input.text())
            # 0.28.0 --- CALCUL DE L'ALTITUDE CIBLE ---
            if self.group_tilted_plane.isChecked() and len(self.tilted_pts_2d) == 2:
                # Mode Plan Incliné : target_z sera un tableau dynamique calculé plus bas
                target_z_scalar = None
            else:
                # Mode Horizontal Classique
                target_z_scalar = self.flatten_spinbox.value()

        except ValueError:
            QMessageBox.warning(self, _("msg_error_title"), _("msg_invalid_parms"))
            return

        #self.save_state_to_history()
        logging.info(f"Starting flattening...")

        # 1. Identifier le Plateau immuable
        plateau_faces = self.original_faces[self.selected_faces_indices]
        plateau_verts_idx = list(set(np.unique(plateau_faces)))

        # On extrait la frontière géométrique UNE SEULE FOIS. Elle est immuable.
        segments_2d = get_selection_boundary_2d(self.original_vertices, self.original_faces, self.selected_faces_indices)

        logging.info("Final earthworks via Unified Core...")

        # On répare les index de sélection obsolètes pour l'UI
        is_plateau_vert = np.zeros(len(self.original_vertices), dtype=bool)
        is_plateau_vert[plateau_verts_idx] = True
        is_plateau_face = is_plateau_vert[self.original_faces[:, 0]] & \
                          is_plateau_vert[self.original_faces[:, 1]] & \
                          is_plateau_vert[self.original_faces[:, 2]]
        self.selected_faces_indices = np.where(is_plateau_face)[0].tolist()

        v2d = self.original_vertices[:, :2]
        dist_verts = np.full(len(self.original_vertices), np.inf)

        if trans_width > 0:
            seg_points = np.array(segments_2d).reshape(-1, 2)
            min_x, min_y = np.min(seg_points, axis=0) - (trans_width * 2)
            max_x, max_y = np.max(seg_points, axis=0) + (trans_width * 2)

            mask_bb_verts = (v2d[:, 0] >= min_x) & (v2d[:, 0] <= max_x) & \
                            (v2d[:, 1] >= min_y) & (v2d[:, 1] <= max_y)
            candidate_verts = np.where(mask_bb_verts)[0]

            dist_verts_local = points_to_segments_dist(v2d[candidate_verts], segments_2d)

            # On retrouve l'index local correspondant aux sommets du plateau
            plateau_local_mask = np.isin(candidate_verts, plateau_verts_idx)
            dist_verts_local[plateau_local_mask] = 0.0

            # 0.28.0 --- GÉNÉRATION DU CHAMP D'ALTITUDES ---
            if target_z_scalar is None:
                # Projection de chaque sommet sur l'axe AB
                A, B = self.tilted_pts_2d[0], self.tilted_pts_2d[1]
                vec_AB = B[:2] - A[:2]
                len_sq = np.dot(vec_AB, vec_AB)
                if len_sq == 0: len_sq = 1.0

                verts_2d = self.original_vertices[candidate_verts, :2]
                vec_AP = verts_2d - A[:2]
                t_projections = np.dot(vec_AP, vec_AB) / len_sq
                target_z_array = A[2] + t_projections * (B[2] - A[2])
            else:
                target_z_array = target_z_scalar

            # Appel du noyau unifié
            self._apply_earthwork_blend_unified(candidate_verts, target_z_array, dist_verts_local, trans_width)

        # On utilise les indices du plateau identifiés juste avant le calcul
        if self.selected_faces_indices:
            sel_array = np.array(self.selected_faces_indices, dtype=int)

            # On change le type en 8 (Route) uniquement pour les triangles de type 0 (Terrain)
            mask_terrain = self.original_tri_types[sel_array] == 0
            self.original_tri_types[sel_array[mask_terrain]] = 8

            # On vide la sélection pour qu'elle ne soit plus affichée en orange
            self.selected_faces_indices = []

        self._cleanup_flatten_ui()
        self.update_selection_colors()

        self.btn_toggle_zone.setChecked(False)
        self.btn_load_zone.setEnabled(True)
        self.toggle_zone_mode()

        self.is_modified = True
        logging.info("Earthworks successfully completed!")
        self.canvas.native.setFocus()
        self.update_pivot_z()

    def cancel_zone(self):
        """Efface le tracé en cours et remet l'outil à zéro."""
        self.zone_polygon_points = []
        self.zone_polygon_redo_stack = []
        self.current_zone_name = ""
        self.current_zone_name_input.clear()

        self.group_tilted_plane.setChecked(False)
        self.group_tilted_plane.setEnabled(False)

        self.update_zone_polygon_visual()

        self.btn_apply_zone.setEnabled(False)
        self.btn_cancel_zone.setEnabled(False)
        self.btn_save_zone.setEnabled(False)

        self.btn_toggle_zone.setStyleSheet("background-color: #188034; color: white; font-weight: bold; padding: 6px;")
        self.check_stop_pulse()

        self.canvas.native.setFocus()

    def load_flat_project(self):
        """Charge un tracé polygonal 3D depuis custom_flat.json."""
        json_file = self.get_custom_file_path("custom_flat.json")

        # Vérifications de base
        if not os.path.exists(json_file):
            QMessageBox.information(self, _("msg_info_title"), _("msg_custom_area_not_found"))
            return

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, _("msg_error_title"), _("msg_json_read_error", error=str(e)))
            return

        tile_id = self.get_current_tile_id()
        if tile_id not in data or not data[tile_id]:
            QMessageBox.information(self, _("msg_info_title"), _("msg_no_area_found", tile=tile_id))
            return

        # Suite
        sel_names = [s.get("name", "Zone sans nom") for s in data[tile_id]]
        name, ok = QInputDialog.getItem(self, _("msg_load_a_complex_area"), _("msg_tile_areas", ti=tile_id), sel_names, 0, False)
        if not ok or not name: return

        sel_data = next((s for s in data[tile_id] if s.get("name") == name), None)
        if not sel_data: return

        self.cancel_zone() # Nettoie
        self.cancel_selection()

        self.zone_polygon_points = sel_data.get("points", [])
        self.current_zone_name = name
        self.current_zone_name_input.setText(name)
        if "trans_width" in sel_data:
            self.flatten_trans_input.setText(str(sel_data["trans_width"]))

        # 0.28.0 - plan incliné
        tilted_active = sel_data.get("tilted_active", False)
        self.group_tilted_plane.setChecked(tilted_active)
        if tilted_active and "tilted_pts" in sel_data and len(sel_data["tilted_pts"]) == 2:
            self.tilted_pts_2d = [np.array(p, dtype=np.float32) for p in sel_data["tilted_pts"]]
            self.btn_def_ab.setText(_("lbl_AB_restored"))
            self.init_tilted_plane_logic()

        if not self.btn_toggle_zone.isChecked():
            self.btn_toggle_zone.setChecked(True)
            self.toggle_zone_mode()

        self.update_zone_polygon_visual()
        self.canvas.native.setFocus()

    def save_flat_project(self):
        """Enregistre le tracé polygonal 3D dans custom_flat.json."""
        if len(self.zone_polygon_points) < 3: return

        if self.group_tilted_plane.isChecked() and len(getattr(self, 'tilted_pts_2d', [])) < 2:
            QMessageBox.warning(self, "Warning", _("msg_plz_define_AB"))
            return

        tile_id = self.get_current_tile_id()
        json_file = self.get_custom_file_path("custom_flat.json")
        data = {}

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f: data = json.load(f)
            except Exception: pass

        if tile_id not in data: data[tile_id] = []
        existing_names = [s.get("name") for s in data[tile_id] if s.get("name")]

        default_name = self.current_zone_name

        # Préparation robuste de la liste pour la combobox
        items = existing_names.copy()
        if default_name and default_name not in items:
            items.insert(0, default_name)

        name, ok = QInputDialog.getItem(self, _("msg_save_complex_area"), _("msg_name_cust", ti=tile_id), items, 0, True)

        if not ok or not name.strip(): return
        name = name.strip()

        if name in existing_names:
            reply = QMessageBox.question(self, _("msg_confirmation"), _("msg_replace_cust", cn=name),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No: return

        # Conversion des types NumPy en types Python natifs pour le JSON
        native_points = [[float(x), float(y), float(z)] for x, y, z in self.zone_polygon_points]

        sel_data = {
            "name": name,
            "points": native_points,
            "trans_width": self.flatten_trans_input.text(),
            "tilted_active": self.group_tilted_plane.isChecked(),
            "tilted_pts": [pt.tolist() for pt in self.tilted_pts_2d] if self.tilted_pts_2d else []
        }

        replaced = False
        for i, existing in enumerate(data[tile_id]):
            if existing.get("name") == name:
                data[tile_id][i] = sel_data
                replaced = True; break

        if not replaced: data[tile_id].append(sel_data)

        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.current_zone_name = name
            self.current_zone_name_input.setText(name)
            QMessageBox.information(self, _("msg_success_title"), _("msg_area_saved",cn=name))
        except Exception as e:
            QMessageBox.critical(self, _("msg_error_title"), _("msg_json_write_error", error=e))

        self.canvas.native.setFocus()

    # 0.28.0 - méthodes pour plan incliné
    def toggle_define_ab_mode(self):
        self.tilted_define_active = self.btn_def_ab.isChecked()
        if self.tilted_define_active:
            self.tilted_pts_2d = []
            self.btn_def_ab.setText(_("lbl_AB_ctrl"))
        else:
            self.btn_def_ab.setText(_("lbl_AB_define"))

    def on_tilted_plane_toggled(self, checked):
        if not checked:
            self.clear_tilted_visuals()
            self.update_flatten_plane_visual() # Restaure le plateau horizontal

    def init_tilted_plane_logic(self):
        z_a, z_b = self.tilted_pts_2d[0][2], self.tilted_pts_2d[1][2]

        # Initialisation UI
        self.spin_z_a.blockSignals(True); self.spin_z_a.setValue(z_a); self.spin_z_a.blockSignals(False)
        self.spin_z_b.blockSignals(True); self.spin_z_b.setValue(z_b); self.spin_z_b.blockSignals(False)

        self._tilted_base_z_a = z_a
        self._tilted_base_z_b = z_b

        self.update_tilted_visuals()

    def update_tilted_visuals(self):
        if len(self.tilted_pts_2d) < 2: return

        pA, pB = self.tilted_pts_2d[0], self.tilted_pts_2d[1]
        line_pts = np.array([pA, pB], dtype=np.float32)

        if getattr(self, 'tilted_line_fg', None) is None:
            # 1. Ligne d'arrière-plan (visible à travers la montagne, en rose translucide)
            self.tilted_line_bg = scene.visuals.Line(pos=line_pts, color='#ffb6c1', width=3, parent=self.view.scene)
            self.tilted_line_bg.set_gl_state(depth_test=False, blend=True)
            self.tilted_line_bg.order = 1

            # 2. Ligne de premier plan (visible par-dessus la montagne, en rouge vif)
            self.tilted_line_fg = scene.visuals.Line(pos=line_pts, color='red', width=5, parent=self.view.scene)
            self.tilted_line_fg.set_gl_state(depth_test=True, depth_func='lequal', blend=True)
            self.tilted_line_fg.order = 2

            # 3. Marqueurs d'arrière-plan
            self.tilted_markers_bg = scene.visuals.Markers(pos=line_pts, face_color='#ffb6c1', edge_color=(0, 0, 0, 0), size=8, parent=self.view.scene)
            self.tilted_markers_bg.set_gl_state(depth_test=False, blend=True)
            self.tilted_markers_bg.order = 3

            # 4. Marqueurs de premier plan (points jaunes)
            self.tilted_markers_fg = scene.visuals.Markers(pos=line_pts, face_color='yellow', edge_color='black', size=12, parent=self.view.scene)
            self.tilted_markers_fg.set_gl_state(depth_test=True, depth_func='lequal', blend=True)
            self.tilted_markers_fg.order = 4
        else:
            # Mise à jour rapide des coordonnées lors d'un mouvement de slider
            self.tilted_line_bg.set_data(pos=line_pts)
            self.tilted_line_fg.set_data(pos=line_pts)

            self.tilted_markers_bg.set_data(pos=line_pts, face_color='#ffb6c1', edge_color=(0, 0, 0, 0), size=8)
            self.tilted_markers_fg.set_data(pos=line_pts, face_color='yellow', edge_color='black', size=12)

        self.update_flatten_plane_visual()

    def clear_tilted_visuals(self):
        """Nettoie les objets visuels liés à l'axe du plan incliné (AB)."""
        # On parcourt les 4 objets visuels créés pour les détruire proprement
        for attr in ['tilted_line_bg', 'tilted_line_fg', 'tilted_markers_bg', 'tilted_markers_fg']:
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.parent = None
                setattr(self, attr, None)

        self.clear_temp_tilted_marker()

    def update_flatten_plane_visual(self):
        if not getattr(self, 'flatten_active', False) or self.flatten_plane is None: return

        if self.group_tilted_plane.isChecked() and len(self.tilted_pts_2d) == 2:
            A, B = self.tilted_pts_2d[0], self.tilted_pts_2d[1]
            vec_AB = B[:2] - A[:2]
            len_sq = np.dot(vec_AB, vec_AB)
            if len_sq == 0: len_sq = 1.0

            # Projection vectorielle pour calculer le Z de chaque sommet du disque visuel
            for i, pt in enumerate(self.flatten_plane_verts):
                vec_AP = pt[:2] - A[:2]
                t = np.dot(vec_AP, vec_AB) / len_sq
                self.flatten_plane_verts[i, 2] = A[2] + t * (B[2] - A[2])
        else:
            # Mode horizontal classique
            target_z = self.flatten_spinbox.value()
            self.flatten_plane_verts[:, 2] = target_z

        self.flatten_plane.set_data(vertices=self.flatten_plane_verts, faces=self.flatten_plane_faces)

    def on_tilted_z_changed(self):
        new_z_a = self._tilted_base_z_a + (self.slider_z_a.value() / 10.0)
        new_z_b = self._tilted_base_z_b + (self.slider_z_b.value() / 10.0)

        self.spin_z_a.blockSignals(True); self.spin_z_a.setValue(new_z_a); self.spin_z_a.blockSignals(False)
        self.spin_z_b.blockSignals(True); self.spin_z_b.setValue(new_z_b); self.spin_z_b.blockSignals(False)

        if len(self.tilted_pts_2d) == 2:
            self.tilted_pts_2d[0][2] = new_z_a
            self.tilted_pts_2d[1][2] = new_z_b
            self.update_tilted_visuals()

    def on_tilted_spin_changed(self):
        if len(self.tilted_pts_2d) == 2:
            self.tilted_pts_2d[0][2] = self.spin_z_a.value()
            self.tilted_pts_2d[1][2] = self.spin_z_b.value()
            self._tilted_base_z_a = self.spin_z_a.value()
            self._tilted_base_z_b = self.spin_z_b.value()

            self.slider_z_a.blockSignals(True); self.slider_z_a.setValue(0); self.slider_z_a.blockSignals(False)
            self.slider_z_b.blockSignals(True); self.slider_z_b.setValue(0); self.slider_z_b.blockSignals(False)
            self.update_tilted_visuals()

    def on_tilted_rot_pressed(self):
        if len(self.tilted_pts_2d) == 2:
            self._rot_orig_a = self.tilted_pts_2d[0].copy()
            self._rot_orig_b = self.tilted_pts_2d[1].copy()

    def on_tilted_rot_changed(self, value, pt_type):
        if not hasattr(self, '_rot_orig_a'): return
        angle_rad = math.radians(value / 10.0)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        # Rotation de A autour de B, ou B autour de A
        if pt_type == 'A':
            cx, cy = self._rot_orig_b[:2]
            ix, iy = self._rot_orig_a[:2]
            idx = 0
        else:
            cx, cy = self._rot_orig_a[:2]
            ix, iy = self._rot_orig_b[:2]
            idx = 1

        nx = cx + (ix - cx) * cos_a - (iy - cy) * sin_a
        ny = cy + (ix - cx) * sin_a + (iy - cy) * cos_a

        self.tilted_pts_2d[idx][:2] = [nx, ny]
        self.update_tilted_visuals()

    def on_tilted_rot_released(self):
        self.slider_rot_a.blockSignals(True); self.slider_rot_a.setValue(0); self.slider_rot_a.blockSignals(False)
        self.slider_rot_b.blockSignals(True); self.slider_rot_b.setValue(0); self.slider_rot_b.blockSignals(False)

    def draw_temp_tilted_marker(self, x, y, z):
        """Affiche une boule jaune temporaire pour le 1er point de l'axe AB."""
        if getattr(self, 'temp_tilted_marker', None) is None:
            self.temp_tilted_marker = scene.visuals.Markers(parent=self.view.scene)
            self.temp_tilted_marker.set_gl_state(depth_test=False) # Visible à travers le mesh

        # Surélevé légèrement pour être bien visible
        self.temp_tilted_marker.set_data(
            pos=np.array([[x, y, z + 2.0]]),
            face_color='yellow',
            edge_color='black',
            size=12
        )

    def clear_temp_tilted_marker(self):
        """Détruit la boule jaune temporaire de l'axe AB."""
        if getattr(self, 'temp_tilted_marker', None) is not None:
            self.temp_tilted_marker.parent = None
            self.temp_tilted_marker = None

    # --- Piste Altiport (Runway) ---

    def toggle_runway_mode(self):
        self.runway_active = self.btn_toggle_runway.isChecked()

        if self.runway_active:
            # 1. Nettoyage initial strict (sans déclencher tout cancel_runway)
            self.runway_pts_2d = []
            self.runway_table.setRowCount(0)

            # 2. Mise à jour de l'UI
            self.btn_toggle_runway.setText(_("txt_set_bounds"))
            self.btn_cancel_runway.setEnabled(True)

            # Note : On n'active PAS les boutons d'ajout/suppression de points ici.
            # L'utilisateur doit d'abord tracer son axe. On les activera dans init_runway_table.

            self.cancel_selection()
            self.pulse_timer.start(500)
            logging.info("Runway Mode activated. Tabs locked.")

            # 3. Verrouillage UX : Isolation de l'utilisateur sur l'onglet "Altiport" (Index 3)
            for i in range(self.tabs.count()):
                if i != 3:
                    self.tabs.setTabEnabled(i, False)
        else:
            # 1. Nettoyage de l'état
            self.btn_toggle_runway.setText(_("btn_def_axis"))
            self.cancel_runway()

            # 2. Désactivation des boutons dépendants
            self.btn_add_point_above.setEnabled(False)
            self.btn_add_point_below.setEnabled(False)
            self.btn_delete_point.setEnabled(False)

            logging.info("Runway Mode deactivated. Tabs unlocked.")

            # 3. Déverrouillage UX
            for i in range(self.tabs.count()):
                self.tabs.setTabEnabled(i, True)

        self.canvas.native.setFocus()

    def init_runway_table(self):
        self.runway_table.setRowCount(0)
        z_start = self.get_z_at_xy(self.runway_pts_2d[0][0], self.runway_pts_2d[0][1])
        z_end = self.get_z_at_xy(self.runway_pts_2d[1][0], self.runway_pts_2d[1][1])

        self._add_row_to_table(0, z_start)
        self._add_row_to_table(100, z_end)

        self.btn_apply_runway.setEnabled(True)
        self.btn_cancel_runway.setEnabled(True)
        self.btn_preview_mode.setEnabled(True)
        self.runway_global_z_slider.setEnabled(True)
        self.runway_rot_slider.setEnabled(False)
        self.btn_save_runway.setEnabled(True)
        self.runway_width_slider.setEnabled(True)

        self.btn_add_point_above.setEnabled(True)
        self.btn_add_point_below.setEnabled(True)
        self.btn_delete_point.setEnabled(True)

    def draw_temp_runway_marker(self, x, y, z):
        """Affiche une boule jaune temporaire pour le 1er point de l'axe de piste."""
        if getattr(self, 'temp_runway_marker', None) is None:
            self.temp_runway_marker = scene.visuals.Markers(parent=self.view.scene)
            self.temp_runway_marker.set_gl_state(depth_test=False) # Désactive la profondeur pour la voir à travers le mesh

        # On surélève de 2 mètres pour être visuellement cohérent avec le preview final
        self.temp_runway_marker.set_data(
            pos=np.array([[x, y, z + 2.0]]),
            face_color='yellow',
            edge_color='black',
            size=12
        )

    def clear_temp_runway_marker(self):
        """Détruit la boule jaune temporaire."""
        if getattr(self, 'temp_runway_marker', None) is not None:
            self.temp_runway_marker.parent = None
            self.temp_runway_marker = None

    def _add_row_to_table(self, pos_val, z_val):
        row = self.runway_table.rowCount()
        self.runway_table.insertRow(row)
        self.runway_table.setItem(row, 0, QTableWidgetItem(f"{pos_val:.1f}"))
        self.runway_table.setItem(row, 1, QTableWidgetItem(f"{z_val:.1f}"))

    def add_runway_point_above(self):
        self._insert_runway_point(direction=-1)
        self.canvas.native.setFocus()

    def add_runway_point_below(self):
        self._insert_runway_point(direction=1)
        self.canvas.native.setFocus()

    def _insert_runway_point(self, direction):
        """
        Insère un point au-dessus (-1) ou en dessous (+1) de la sélection actuelle.
        """
        row_count = self.runway_table.rowCount()

        # Sécurité : Limite de 7 points
        if row_count >= self.RUNWAY_MAX_PTS:
            return

        selected_items = self.runway_table.selectedItems()
        if not selected_items:
            return

        idx = selected_items[0].row()

        # Détermination des index cibles selon la direction
        if direction == -1: # Au-dessus
            if idx == 0:
                return  # On ne peut rien mettre au-dessus du point 0%
            insert_idx = idx
            idx_top, idx_bottom = idx - 1, idx
        else: # En dessous
            if idx == row_count - 1:
                return  # On ne peut rien mettre en dessous du point 100%
            insert_idx = idx + 1
            idx_top, idx_bottom = idx, idx + 1

        # Calcul des nouvelles valeurs (moyenne exacte entre les deux points encadrants)
        try:
            pos_top = float(self.runway_table.item(idx_top, 0).text())
            alt_top = float(self.runway_table.item(idx_top, 1).text())
            pos_bottom = float(self.runway_table.item(idx_bottom, 0).text())
            alt_bottom = float(self.runway_table.item(idx_bottom, 1).text())
        except (ValueError, AttributeError):
            return

        new_pos = (pos_top + pos_bottom) / 2.0
        new_alt = (alt_top + alt_bottom) / 2.0

        # Insertion visuelle dans le tableau
        self.runway_table.insertRow(insert_idx)

        item_pos = QTableWidgetItem(f"{new_pos:.1f}")
        item_pos.setTextAlignment(Qt.AlignCenter)
        self.runway_table.setItem(insert_idx, 0, item_pos)

        item_alt = QTableWidgetItem(f"{new_alt:.1f}")
        item_alt.setTextAlignment(Qt.AlignCenter)
        self.runway_table.setItem(insert_idx, 1, item_alt)

        # Sélection automatique de la nouvelle ligne
        self.runway_table.selectRow(insert_idx)

        # Mise à jour immédiate du rendu 3D de la courbe
        if hasattr(self, 'update_runway_preview'):
            self.update_runway_preview()

        logging.info(f"Inserted new runway control point at position {new_pos:.1f}% with altitude {new_alt:.1f}m.")

    def delete_runway_point(self):
        """
        Supprime le point actuellement sélectionné dans le tableau.
        Sécurité : Empêche la suppression du premier (0%) et du dernier (100%) point.
        """
        selected_items = self.runway_table.selectedItems()
        if not selected_items:
            return  # Rien n'est sélectionné

        idx = selected_items[0].row()
        row_count = self.runway_table.rowCount()

        # Sécurité : Interdiction de supprimer les points d'extrémité
        if idx == 0 or idx == row_count - 1:
            return  # On quitte silencieusement sans rien faire

        # Suppression de la ligne
        self.runway_table.removeRow(idx)

        # Sélection automatique du point précédent pour garder le focus
        if idx - 1 >= 0:
            self.runway_table.selectRow(idx - 1)

        # Mise à jour immédiate du rendu 3D de la courbe
        if hasattr(self, 'update_runway_preview'):
            self.update_runway_preview()

        self.canvas.native.setFocus()
        logging.info(f"Control point removed from runway profile table at row index {idx}.")

    def on_runway_width_changed(self, value):
        """Met à jour la largeur interne et rafraîchit la prévisualisation 3D."""
        self.runway_width_input.setText(str(value))

        if hasattr(self, 'update_runway_preview') and len(self.runway_pts_2d) >= 2:
            self.update_runway_preview()

    def on_runway_width_text_edited(self, text):
        """Met à jour le slider et le rendu 3D quand on tape une largeur au clavier."""
        try:
            width = float(text)
            # On bloque les signaux du slider pour éviter une boucle infinie
            self.runway_width_slider.blockSignals(True)
            self.runway_width_slider.setValue(int(width))
            self.runway_width_slider.blockSignals(False)

            if hasattr(self, 'update_runway_preview') and len(self.runway_pts_2d) >= 2:
                self.update_runway_preview()
        except ValueError:
            pass # Si l'utilisateur tape des lettres ou efface tout, on ignore silencieusement

    def update_runway_preview(self):
        if len(self.runway_pts_2d) < 2: return

        try:
            # ==========================================================
            # 1. CALCULS MATHÉMATIQUES (On calcule tout AVANT de dessiner)
            # ==========================================================
            pts = []
            for row in range(self.runway_table.rowCount()):
                pts.append((float(self.runway_table.item(row, 0).text()) / 100.0, float(self.runway_table.item(row, 1).text())))
            pts.sort(key=lambda x: x[0])

            p1, p2 = self.runway_pts_2d[0], self.runway_pts_2d[1]
            spline = NumpyPchipInterpolator([p[0] for p in pts], [p[1] for p in pts])

            # --- Calcul de la Ligne ---
            t_vals_line = np.linspace(0, 1, 100)
            line_verts = []
            for t in t_vals_line:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                line_verts.append([x, y, spline(t) + 2.0])
            line_array = np.array(line_verts, dtype=np.float32)

            # --- Calcul des Marqueurs (Points jaunes) ---
            ctrl_verts = []
            for t, z in pts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                ctrl_verts.append([x, y, z + 2.0])
            ctrl_array = np.array(ctrl_verts, dtype=np.float32)

            # --- Calcul du Ruban ---
            try:
                width = float(self.runway_width_input.text())
            except ValueError:
                width = 40.0 # Sécurité si le champ est vide

            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length == 0: length = 1.0
            n_dir = np.array([-vec[1], vec[0]]) / length

            segments = 50
            t_vals_ribbon = np.linspace(0, 1, segments)

            ribbon_verts = []
            ribbon_faces = []

            for i, t in enumerate(t_vals_ribbon):
                center_x = p1[0] + t * vec[0]
                center_y = p1[1] + t * vec[1]
                center_z = spline(t) + 2.0

                p_left = np.array([center_x, center_y]) + n_dir * (width / 2.0)
                p_right = np.array([center_x, center_y]) - n_dir * (width / 2.0)

                ribbon_verts.append([p_left[0], p_left[1], center_z])
                ribbon_verts.append([p_right[0], p_right[1], center_z])

                if i < segments - 1:
                    idx = i * 2
                    ribbon_faces.append([idx, idx + 1, idx + 2])
                    ribbon_faces.append([idx + 1, idx + 3, idx + 2])

            v_array = np.array(ribbon_verts, dtype=np.float32)
            f_array = np.array(ribbon_faces, dtype=np.uint32)

            # ==========================================================
            # 2. CRÉATION OU MISE À JOUR DES VISUELS (Anti-Ghosting)
            # ==========================================================
            if getattr(self, 'runway_line_fg', None) is None:
                # Création initiale en injectant les données calculées !
                self.runway_line_bg = scene.visuals.Line(pos=line_array, color='#ffb6c1', width=3, parent=self.view.scene)
                self.runway_line_bg.set_gl_state(depth_test=False, blend=True)
                self.runway_line_bg.order = 1

                self.runway_line_fg = scene.visuals.Line(pos=line_array, color='red', width=5, parent=self.view.scene)
                self.runway_line_fg.set_gl_state(depth_test=True, depth_func='lequal', blend=True)
                self.runway_line_fg.order = 2

                self.runway_markers_bg = scene.visuals.Markers(pos=ctrl_array, face_color='#ffb6c1', edge_color=(0, 0, 0, 0), size=8, parent=self.view.scene)
                self.runway_markers_bg.set_gl_state(depth_test=False, blend=True)
                self.runway_markers_bg.order = 3

                self.runway_markers_fg = scene.visuals.Markers(pos=ctrl_array, face_color='yellow', edge_color='black', size=12, parent=self.view.scene)
                self.runway_markers_fg.set_gl_state(depth_test=True, depth_func='lequal', blend=True)
                self.runway_markers_fg.order = 4

                self.runway_ribbon_fg = scene.visuals.Mesh(vertices=v_array, faces=f_array, color=(0.0, 1.0, 1.0, 0.5), shading=None, parent=self.view.scene)

                # On utilise l'état 'translucent' natif de VisPy qui gère le blending proprement sans casser le buffer
                self.runway_ribbon_fg.set_gl_state('translucent', depth_test=True, depth_func='lequal')
                self.runway_ribbon_fg.order = 5
            else:
                # Mise à jour ultra-rapide (quand tu bouges les sliders)
                self.runway_line_bg.set_data(pos=line_array)
                self.runway_line_fg.set_data(pos=line_array)

                self.runway_markers_bg.set_data(pos=ctrl_array, face_color='#ffb6c1', edge_color=(0, 0, 0, 0), size=8)
                self.runway_markers_fg.set_data(pos=ctrl_array, face_color='yellow', edge_color='black', size=12)

                self.runway_ribbon_fg.set_data(vertices=v_array, faces=f_array)

            # ==========================================================
            # 3. GESTION DE LA VISIBILITÉ (Squelette vs Ruban)
            # ==========================================================
            is_surface_mode = self.btn_preview_mode.isChecked()

            self.runway_line_bg.visible = not is_surface_mode
            self.runway_line_fg.visible = not is_surface_mode
            self.runway_ribbon_fg.visible = is_surface_mode

            # Les marqueurs restent toujours visibles pour que tu saches ce que tu édites
            self.runway_markers_bg.visible = True
            self.runway_markers_fg.visible = True

        except Exception as e:
            logging.info(f"Preview error : {e}", exc_info=True)

    def abort_runway(self):
        """Ferme proprement l'outil piste en passant par le gestionnaire d'état central."""
        self.btn_toggle_runway.setChecked(False)
        self.toggle_runway_mode()

    def cancel_runway(self):
        self.runway_active = False
        self.btn_toggle_runway.setChecked(False)
        self.btn_toggle_runway.setEnabled(True)
        self.btn_toggle_runway.setText(_("btn_def_axis"))
        self.runway_pts_2d = []
        self.clear_temp_runway_marker()
        self.runway_table.setRowCount(0)
        self.btn_apply_runway.setEnabled(False)
        self.btn_cancel_runway.setEnabled(False)
        self.runway_global_z_slider.setEnabled(False)
        self.runway_rot_slider.setEnabled(False)
        self.runway_width_slider.setEnabled(False)
        if hasattr(self, 'btn_save_runway'):
            self.btn_save_runway.setEnabled(False)
        self.current_runway_name = ""
        self.current_runway_name_input.clear()

        # Nettoyage propre et définitif des objets VisPy
        for attr in ['runway_line_bg', 'runway_line_fg', 'runway_markers_bg', 'runway_markers_fg', 'runway_ribbon_fg']:
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.parent = None
                setattr(self, attr, None)

        self.btn_toggle_runway.setStyleSheet("background-color: #188034; color: white; font-weight: bold; padding: 6px;")
        self.check_stop_pulse()

        self.canvas.native.setFocus()

    @wait_cursor
    def apply_runway(self, checked=False):
        if len(self.runway_pts_2d) < 2:
            return

        # =====================================================================
        # 0. RÉCUPÉRATION DES PARAMÈTRES
        # =====================================================================
        try:
            width = float(self.runway_width_input.text())
            t_side = float(self.runway_trans_side_input.text())
            subdivs_runway = self.slider_runway_subdiv.value()
            step_size = float(self.RUNWAY_RECT_SIZE)
            clearance_margin = 10.0  # Marge de sécurité demandée en mètres

            # Préparation de la spline pour l'altitude
            pts = []
            for row in range(self.runway_table.rowCount()):
                pts.append((float(self.runway_table.item(row, 0).text()) / 100.0, float(self.runway_table.item(row, 1).text())))
            pts.sort(key=lambda x: x[0])
            spline_func = NumpyPchipInterpolator([p[0] for p in pts], [p[1] for p in pts])

        except ValueError:
            QMessageBox.warning(self, _("msg_error_title"), _("msg_invalid_parms"))
            return

        p1, p2 = self.runway_pts_2d[0], self.runway_pts_2d[1]

        # =====================================================================
        # 0.5 SÉCURITÉ : VÉRIFICATION DES LIMITES ET DES CONFLITS (EMPREINTE)
        # =====================================================================
        logging.info("0/5 : Boundary check and triangle type conflicts on runway core...")

        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length == 0: length = 1.0
        dir_vec = vec / length
        n_dir = np.array([-vec[1], vec[0]]) / length

        # --- A. VÉRIFICATION DE DÉBORDEMENT DU MESH (OUT OF BOUNDS) ---
        # On calcule l'emprise maximale absolue (Piste + Talus) pour s'assurer
        # qu'aucun calcul de raccordement ne se fera dans le vide en dehors de la tuile.
        max_trans = t_side
        hw_total = (width / 2.0) + max_trans
        p1_ext = p1 - dir_vec * max_trans
        p2_ext = p2 + dir_vec * max_trans

        poly_max_extent = [
            p1_ext - n_dir * hw_total,
            p1_ext + n_dir * hw_total,
            p2_ext + n_dir * hw_total,
            p2_ext - n_dir * hw_total
        ]

        # Test des 4 coins extrêmes de l'emprise des travaux
        for corner in poly_max_extent:
            if not self.is_xy_strictly_inside_mesh(corner[0], corner[1]):
                QMessageBox.warning(self, _("msg_action_denied"), _("msg_unable_to_build_the_track"))
                return

        # --- B. VÉRIFICATION DES CONFLITS DE TYPE SUR L'EMPREINTE DE LA PISTE ---
        # On ne teste QUE la piste + une micro-marge technique (ex: 3 mètres)
        micro_margin = 3.0
        hw_core = (width / 2.0) + micro_margin

        poly_core_check = [
            p1 - n_dir * hw_core,
            p1 + n_dir * hw_core,
            p2 + n_dir * hw_core,
            p2 - n_dir * hw_core
        ]

        # Filtre rapide par Bounding Box locale
        poly_arr = np.array(poly_core_check)
        min_x, max_x = np.min(poly_arr[:, 0]) - 10.0, np.max(poly_arr[:, 0]) + 10.0
        min_y, max_y = np.min(poly_arr[:, 1]) - 10.0, np.max(poly_arr[:, 1]) + 10.0

        v2d = self.original_vertices[:, :2]
        mask_v_bb = (v2d[:, 0] >= min_x) & (v2d[:, 0] <= max_x) & \
                    (v2d[:, 1] >= min_y) & (v2d[:, 1] <= max_y)

        mask_f_bb = mask_v_bb[self.original_faces[:, 0]] | \
                    mask_v_bb[self.original_faces[:, 1]] | \
                    mask_v_bb[self.original_faces[:, 2]]

        cand_idx = np.where(mask_f_bb)[0]
        if len(cand_idx) > 0:
            cand_faces = self.original_faces[cand_idx]
            cand_types = self.original_tri_types[cand_idx]

            # Évaluation stricte sur l'emprise de la piste
            centers_in = points_in_polygons_concave(
                np.mean(self.original_vertices[cand_faces][:, :, :2], axis=1),
                [poly_core_check]
            )
            edges_in = faces_intersecting_polygon(self.original_vertices, cand_faces, poly_core_check)

            verts_in_poly = points_in_polygons_concave(v2d, [poly_core_check])
            any_vert_in = verts_in_poly[cand_faces[:, 0]] | \
                          verts_in_poly[cand_faces[:, 1]] | \
                          verts_in_poly[cand_faces[:, 2]]

            if np.any(cand_types[centers_in | edges_in | any_vert_in] > 0):
                QMessageBox.warning(self, _("msg_action_denied"), _("msg_unable_to_build_the_track"))
                return

        # Si tout est clair, on sauvegarde l'historique et on lance la machine !
        self.save_state_to_history()

        # =====================================================================
        # 1. SUBDIVISION PRÉALABLE DU MAILLAGE LOCAL
        # =====================================================================
        logging.info(f"1/5 : Subdivision of runway area (Runway and bank: {subdivs_runway})...")

        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        line_dir = line_vec / line_len if line_len > 0 else np.array([1.0, 0.0])

        max_trans = t_side
        total_width = width / 2.0 + max_trans
        margin_t = (max_trans / line_len) if line_len > 0 else 0

        # --- Bounding Box Globale pour optimiser les calculs locaux ---
        margin = max_trans + clearance_margin + 100.0
        min_x, max_x = min(p1[0], p2[0]) - margin, max(p1[0], p2[0]) + margin
        min_y, max_y = min(p1[1], p2[1]) - margin, max(p1[1], p2[1]) + margin

        for step in range(subdivs_runway):
            # 1. Extraction locale
            v2d = self.original_vertices[:, :2]
            v_in_bb = (v2d[:, 0] >= min_x) & (v2d[:, 0] <= max_x) & (v2d[:, 1] >= min_y) & (v2d[:, 1] <= max_y)
            v_idx_in_bb = np.where(v_in_bb)[0]

            f_in_bb_mask = v_in_bb[self.original_faces[:, 0]] | \
                           v_in_bb[self.original_faces[:, 1]] | \
                           v_in_bb[self.original_faces[:, 2]]

            cand_f_global_idx = np.where(f_in_bb_mask)[0]
            cand_faces = self.original_faces[cand_f_global_idx]

            # 2. Maths localisées : Centres et rayons des triangles
            v0 = self.original_vertices[cand_faces[:, 0]][:, :2]
            v1 = self.original_vertices[cand_faces[:, 1]][:, :2]
            v2 = self.original_vertices[cand_faces[:, 2]][:, :2]

            cand_centers = (v0 + v1 + v2) / 3.0

            # Rayon englobant pour attraper les triangles périphériques
            r0 = np.linalg.norm(v0 - cand_centers, axis=1)
            r1 = np.linalg.norm(v1 - cand_centers, axis=1)
            r2 = np.linalg.norm(v2 - cand_centers, axis=1)
            max_radius = np.maximum(np.maximum(r0, r1), r2)

            ap_centers = cand_centers - p1
            t_centers = np.dot(ap_centers, line_dir) / line_len if line_len > 0 else np.zeros(len(cand_centers))
            proj_centers = p1 + np.outer(t_centers * line_len, line_dir)
            dist_lat_centers = np.linalg.norm(cand_centers - proj_centers, axis=1)

            # CORRECTION : Marge de tolérance dynamique basée sur la taille du triangle
            t_margin_array = max_radius / line_len if line_len > 0 else np.zeros(len(cand_centers))
            effective_dist_lat = dist_lat_centers - max_radius

            mask_local = np.zeros(len(cand_centers), dtype=bool)

            # Évaluation pour la Piste et Talus
            if step < subdivs_runway:
                # Piste (prise en compte de la taille du triangle sur les axes X et Y)
                mask_runway = (t_centers >= -t_margin_array) & (t_centers <= 1.0 + t_margin_array) & (effective_dist_lat <= (width / 2.0))
                mask_local |= mask_runway

                current_width = total_width * (1.5 - (step / subdivs_runway)) if subdivs_runway > 0 else total_width

                # Talus
                mask_bank = (t_centers >= -(margin_t + t_margin_array)) & (t_centers <= 1.0 + margin_t + t_margin_array) & (effective_dist_lat <= current_width)

                # Exclusion du centre (On garde la distance stricte ici pour ne pas exclure à tort la périphérie de la piste)
                mask_runway_interior = (t_centers >= 0.0) & (t_centers <= 1.0) & (dist_lat_centers <= (width / 2.0))
                mask_bank &= ~mask_runway_interior

                mask_local |= mask_bank

            # Filtres de sécurité globaux
            mask_local &= (self.original_tri_types[cand_f_global_idx] == 0)
            mask_local &= (self.original_tri_levels[cand_f_global_idx] < self.MAX_SUBDIV_LEVEL)

            candidate_faces_idx = cand_f_global_idx[mask_local].tolist()
            if not candidate_faces_idx: break

            # Exécution de la coupe
            new_v, new_u, new_f, new_t, _dummy, cuts, new_l = subdivide_mesh_selection(
                self.original_vertices, self.original_uvs, self.original_faces,
                self.original_tri_types, candidate_faces_idx, self.original_tri_levels,
                max_subdiv_level = self.MAX_SUBDIV_LEVEL
            )
            if cuts == 0: break

            self.original_vertices, self.original_uvs = new_v, new_u
            self.original_faces, self.original_tri_types = new_f, new_t
            self.original_tri_levels = new_l

        # =====================================================================
        # 2. GÉNÉRATION DU PATCH DE PISTE (ALIGNÉ SUR LA GRILLE)
        # =====================================================================
        logging.info("2/5 : Generation of the texture-safe runway patch...")

        x_lines, y_lines = get_texture_grid_lines(
            min(p1[0], p2[0]) - 100, max(p1[0], p2[0]) + 100,
            min(p1[1], p2[1]) - 100, max(p1[1], p2[1]) + 100,
            19, self.lon_to_m, self.lat_to_m, self.x_center, self.y_center
        )

        patch_verts, patch_faces = generate_sliced_runway_mesh(p1, p2, width, spline_func, x_lines, y_lines, step_size)

        if len(patch_verts) == 0:
            logging.info("Failed to generate runway patch.")
            return

        # Fusion robuste des sommets du ruban (Welding)
        verts_f64 = np.array(patch_verts, dtype=np.float64)
        rounded_verts = np.round(verts_f64, decimals=1)
        _dummy, unique_indices, inverse_indices = np.unique(rounded_verts, axis=0, return_index=True, return_inverse=True)

        patch_verts = patch_verts[unique_indices]
        patch_faces = inverse_indices[patch_faces]

        # Suppression des triangles effondrés par la fusion
        valid_faces_mask = (patch_faces[:, 0] != patch_faces[:, 1]) & \
                           (patch_faces[:, 1] != patch_faces[:, 2]) & \
                           (patch_faces[:, 0] != patch_faces[:, 2])

        patch_faces = patch_faces[valid_faces_mask]

        # =====================================================================
        # 3. DESTRUCTION DU TERRAIN ET CLIPPING
        # =====================================================================
        logging.info("3/5 : Terrain strict clipping (Vertices + Edge Intersection) with Immunity...")

        # Extension longitudinale et latérale de la zone de sécurité
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length == 0: length = 1.0 # Sécurité anti-division par zéro

        dir_vec = vec / length
        n_dir = np.array([-vec[1], vec[0]]) / length

        hw_clear = (width / 2.0) + clearance_margin
        p1_ext = p1 - dir_vec * clearance_margin
        p2_ext = p2 + dir_vec * clearance_margin

        poly_clear = [
            p1_ext - n_dir * hw_clear, p1_ext + n_dir * hw_clear,
            p2_ext + n_dir * hw_clear, p2_ext - n_dir * hw_clear
        ]

        # Double condition de destruction
        verts_inside = points_in_polygons_concave(self.original_vertices[:, :2], [poly_clear])
        faces_with_verts_inside = np.any(verts_inside[self.original_faces], axis=1)
        faces_cutting_corners = faces_intersecting_polygon(self.original_vertices, self.original_faces, poly_clear)
        faces_to_delete = faces_with_verts_inside | faces_cutting_corners

        # Si un seul triangle à détruire possède un type > 0 (Eau, Route, etc.), on avorte tout.
        if np.any(self.original_tri_types[faces_to_delete] > 0):
            QMessageBox.warning(self, _("msg_action_denied"), _("msg_unable_to_build_the_track"))

            # On simule un Ctrl+Z topologique pour nettoyer les subdivisions de l'étape 1
            if self.undo_stack:
                previous_state = self.undo_stack.pop()
                self.restore_state(previous_state['data'])
                self.redo_stack.clear()
                self.update_undo_redo_buttons()

            # On force la prévisualisation 3D à se recalculer sur le mesh sain restauré
            self.update_runway_preview()
            self.update_pivot_z()
            return

        face_mask = ~faces_to_delete
        hole_faces = self.original_faces[face_mask]
        hole_tri_types = self.original_tri_types[face_mask]

        if hasattr(self, 'original_tri_levels'):
            hole_tri_levels = self.original_tri_levels[face_mask]
        else:
            hole_tri_levels = np.zeros(len(hole_faces), dtype=np.uint8)

        deleted_faces = self.original_faces[faces_to_delete]
        patch_center = np.mean([p1, p2], axis=0)

        hole_boundary = get_ordered_boundary_loop(deleted_faces, self.original_vertices, patch_center)
        patch_boundary = get_ordered_boundary_loop(patch_faces)

        # UVs Mapping du patch
        patch_uvs = np.zeros((len(patch_verts), 2), dtype=np.float32)
        for idx, pt in enumerate(patch_verts):
            dist_sq = (self.original_vertices[:, 0] - pt[0])**2 + (self.original_vertices[:, 1] - pt[1])**2
            patch_uvs[idx] = self.original_uvs[np.argmin(dist_sq)]

        # Décalage des index
        offset = len(self.original_vertices)
        shifted_patch_faces = patch_faces + offset
        shifted_patch_boundary = [idx + offset for idx in patch_boundary]

        combined_verts = np.vstack((self.original_vertices, patch_verts))
        combined_uvs = np.vstack((self.original_uvs, patch_uvs))

        # Stockage pour le debug
        self.debug_hole_boundary = hole_boundary
        self.debug_patch_boundary = shifted_patch_boundary
        self.debug_verts = combined_verts

        patch_types = np.full(len(patch_faces), self.RUNWAY_TYPE, dtype=np.uint32)
        patch_levels = np.zeros(len(patch_faces), dtype=np.uint8)

        self.original_vertices = combined_verts
        self.original_uvs = combined_uvs
        self.original_faces = np.vstack((hole_faces, shifted_patch_faces))
        self.original_tri_types = np.concatenate((hole_tri_types, patch_types))
        self.original_tri_levels = np.concatenate((hole_tri_levels, patch_levels))

        # BLOC POUR DEBUG
        # self.is_modified = True
        # self.update_selection_colors()
        # self.cancel_runway()
        # self.update_pivot_z()
        # return

        # =====================================================================
        # 4. GÉNÉRATION DU TALUS (STITCHING)
        # =====================================================================
        logging.info("4/5 : Generation of embankment (Stitching) via CDT...")

        hole_poly = self.original_vertices[self.debug_hole_boundary]
        runway_poly = self.original_vertices[self.debug_patch_boundary]

        margin = 10.0
        min_x, max_x = np.min(hole_poly[:, 0]) - margin, np.max(hole_poly[:, 0]) + margin
        min_y, max_y = np.min(hole_poly[:, 1]) - margin, np.max(hole_poly[:, 1]) + margin

        zl_grid = 19
        grid_xs, grid_ys = get_texture_grid_lines(
            min_x, max_x, min_y, max_y, zl_grid,
            self.lon_to_m, self.lat_to_m, self.x_center, self.y_center
        )

        anchor_segments = compute_anchor_segments(hole_poly, runway_poly, grid_xs, grid_ys)

        new_faces_local, local_vertices_3d = perform_stitching_cdt(
            hole_poly,
            runway_poly,
            anchor_segments
        )

        if new_faces_local is not None:
            num_h = len(self.debug_hole_boundary)
            num_r = len(self.debug_patch_boundary)

            local_to_global = np.zeros(len(local_vertices_3d), dtype=int)
            local_to_global[:num_h] = self.debug_hole_boundary
            local_to_global[num_h:num_h+num_r] = self.debug_patch_boundary

            if len(local_vertices_3d) > num_h + num_r:
                steiner_pts = local_vertices_3d[num_h + num_r:]
                steiner_pts = np.array(steiner_pts, dtype=np.float32)
                start_idx = len(self.original_vertices)

                self.original_vertices = np.vstack((self.original_vertices, steiner_pts))

                steiner_uvs = []
                for pt in steiner_pts:
                    lat, lon = meters_to_latlon(
                        pt[0], pt[1],
                        self.lon_to_m, self.lat_to_m,
                        self.x_center, self.y_center
                    )
                    steiner_uvs.append([lon, lat])

                new_uvs = np.array(steiner_uvs, dtype=np.float32)
                self.original_uvs = np.vstack((self.original_uvs, new_uvs))

                local_to_global[num_h + num_r:] = np.arange(start_idx, start_idx + len(steiner_pts))

            new_faces_global = local_to_global[new_faces_local]
            self.original_faces = np.vstack((self.original_faces, new_faces_global))

            new_types = np.zeros(len(new_faces_global), dtype=np.uint32)
            self.original_tri_types = np.concatenate([self.original_tri_types, new_types])

            if hasattr(self, 'original_tri_levels'):
                new_levels = np.zeros(len(new_faces_global), dtype=np.uint8)
                self.original_tri_levels = np.concatenate([self.original_tri_levels, new_levels])

            logging.info(f"Fusion done : {len(new_faces_global)} triangles of embankment integrated.")

        # =====================================================================
        # 5. LISSAGE DU TALUS (Earthworks Blend Unifié)
        # =====================================================================
        logging.info("5/5 : Final smoothing of embankment (Euclidean Distance)...")

        # On aligne le comportement sur apply_flatten : largeur unique
        trans_width = t_side

        # --- CORRECTION DE LA BOUNDING BOX ---
        # On redéfinit une zone d'action très large qui inclut la largeur de la piste,
        # le talus, et une confortable marge de sécurité de 50m.
        margin_talus = (width / 2.0) + trans_width + 50.0
        min_x = min(p1[0], p2[0]) - margin_talus
        max_x = max(p1[0], p2[0]) + margin_talus
        min_y = min(p1[1], p2[1]) - margin_talus
        max_y = max(p1[1], p2[1]) + margin_talus
        # -------------------------------------

        # 1. Isoler les sommets dans la zone locale
        v2d_hd = self.original_vertices[:, :2]
        v_in_bb_mask = (v2d_hd[:, 0] >= min_x) & (v2d_hd[:, 0] <= max_x) & \
                       (v2d_hd[:, 1] >= min_y) & (v2d_hd[:, 1] <= max_y)
        local_verts_idx = np.where(v_in_bb_mask)[0]

        # 2. Mathématiques sur l'échantillon local
        v2d_local = v2d_hd[local_verts_idx]
        ap_hd = v2d_local - p1
        t_hd = np.dot(ap_hd, line_dir) / line_len if line_len > 0 else np.zeros(len(v2d_local))
        proj_hd = p1 + np.outer(t_hd * line_len, line_dir)
        dist_lat_hd = np.linalg.norm(v2d_local - proj_hd, axis=1)

        # 3. Calcul de la distance absolue (Euclidienne) au bord du rectangle de la piste
        dist_lat_out = np.maximum(0.0, dist_lat_hd - width / 2.0)
        dist_long_out = np.where(t_hd < 0, abs(t_hd * line_len), np.where(t_hd > 1.0, abs((t_hd - 1.0) * line_len), 0.0))
        dist_to_runway = np.sqrt(dist_lat_out**2 + dist_long_out**2)

        # 4. Calcul des altitudes cibles (via la Spline) projetées sur l'axe
        target_z_array = spline_func(np.clip(t_hd, 0.0, 1.0))

        # 5. Appel du noyau unifié
        self._apply_earthwork_blend_unified(local_verts_idx, target_z_array, dist_to_runway, trans_width)

        # --- Finitions ---
        self.is_modified = True
        self.update_selection_colors()
        self.btn_toggle_runway.setChecked(False)
        self.toggle_runway_mode()
        self.update_pivot_z()

    def on_runway_table_selection(self):
        """Active et configure les sliders en fonction du point sélectionné."""
        selected_rows = self.runway_table.selectedItems()
        if not selected_rows:
            self.runway_pos_slider.setEnabled(False)
            self.runway_z_slider.setEnabled(False)
            self.runway_rot_slider.setEnabled(False)
            return

        row = selected_rows[0].row()
        pos_val = float(self.runway_table.item(row, 0).text())
        z_val = float(self.runway_table.item(row, 1).text())

        # Config du slider Z (On mémorise l'altitude de base pour le +/- 100m)
        self._active_runway_base_z = z_val
        self.runway_z_slider.blockSignals(True)
        self.runway_z_slider.setValue(0) # Centre le slider
        self.runway_z_slider.setEnabled(True)
        self.runway_z_slider.blockSignals(False)

        # Config du slider Position (Blocage ou Mode Échelle)
        self.runway_pos_slider.blockSignals(True)
        if row == 0 or row == self.runway_table.rowCount() - 1:
            # --- MODE ÉCHELLE (Extrémités) ---
            self.lbl_runway_pos.setText(_("txt_scale"))
            self.runway_pos_slider.setEnabled(True)
            self.runway_pos_slider.setRange(-50, 50) # De -50% à +50% de la taille actuelle
            self.runway_pos_slider.setValue(0)       # Toujours centré au départ
            self.runway_rot_slider.setEnabled(True)  # Rotation active
        else:
            # --- MODE POSITION (Intermédiaire) ---
            self.lbl_runway_pos.setText(_("txt_pos_pct"))
            self.runway_pos_slider.setEnabled(True)
            self.runway_rot_slider.setEnabled(False) # Rotation grisée

            # On contraint le slider entre le point précédent et le point suivant
            min_pos = int(float(self.runway_table.item(row - 1, 0).text())) + 1
            max_pos = int(float(self.runway_table.item(row + 1, 0).text())) - 1
            self.runway_pos_slider.setRange(min_pos, max_pos)
            self.runway_pos_slider.setValue(int(pos_val))
        self.runway_pos_slider.blockSignals(False)

    def on_runway_pos_pressed(self):
        """Mémorise l'état initial de la piste avant un redimensionnement."""
        selected_rows = self.runway_table.selectedItems()
        if not selected_rows: return
        row = selected_rows[0].row()

        # On n'active le mode Scale que si on attrape une extrémité
        if row == 0 or row == self.runway_table.rowCount() - 1:
            self._is_scaling_runway = True

            # 1. Sauvegarde des positions 2D
            self._scale_initial_pts_2d = [p.copy() for p in self.runway_pts_2d]

            # 2. Sauvegarde de TOUTES les altitudes Z du tableau
            self._scale_initial_zs = []
            for r in range(self.runway_table.rowCount()):
                self._scale_initial_zs.append(float(self.runway_table.item(r, 1).text()))

            # 3. Définition du point mobile et du pivot
            if row == 0:
                self._scale_moving_idx = 0
                self._scale_pivot_idx = 1
                self._scale_pivot_row = self.runway_table.rowCount() - 1
            else:
                self._scale_moving_idx = 1
                self._scale_pivot_idx = 0
                self._scale_pivot_row = 0
        else:
            self._is_scaling_runway = False

    def on_runway_pos_slider_changed(self, value):
        """Déplace un point central OU redimensionne toute la piste en préservant les pentes Z."""
        selected_rows = self.runway_table.selectedItems()
        if not selected_rows: return
        row = selected_rows[0].row()

        if getattr(self, '_is_scaling_runway', False):
            # ==========================================
            # MODE 1 : ÉCHELLE (Redimensionnement 3D)
            # ==========================================
            # Facteur d'échelle : value=0 -> 1.0 (100%). value=50 -> 1.5 (150%)
            scale_factor = 1.0 + (value / 100.0)

            # --- A. Redimensionnement 2D (X, Y) ---
            pivot_2d = self._scale_initial_pts_2d[self._scale_pivot_idx]
            moving_2d_orig = self._scale_initial_pts_2d[self._scale_moving_idx]

            # Formule du vecteur étendu : P_nouveau = P_pivot + (P_mobile_orig - P_pivot) * Scale
            new_moving_2d = pivot_2d + (moving_2d_orig - pivot_2d) * scale_factor
            self.runway_pts_2d[self._scale_moving_idx] = new_moving_2d

            # --- B. Redimensionnement de l'Altitude (Z) ---
            pivot_z = self._scale_initial_zs[self._scale_pivot_row]

            self.runway_table.blockSignals(True)
            for r in range(self.runway_table.rowCount()):
                orig_z = self._scale_initial_zs[r]
                # Formule pour conserver la pente exacte : Z_new = Z_pivot + (Z_orig - Z_pivot) * Scale
                new_z = pivot_z + (orig_z - pivot_z) * scale_factor
                self.runway_table.item(r, 1).setText(f"{new_z:.1f}")
            self.runway_table.blockSignals(False)

            self.update_runway_preview()

        else:
            # ==========================================
            # MODE 2 : POSITION CLASSIQUE (Glissement)
            # ==========================================
            self.runway_table.item(row, 0).setText(f"{float(value):.1f}")
            self.update_runway_preview()

    def on_runway_pos_released(self):
        """Recentrez le slider d'échelle pour permettre un redimensionnement infini."""
        if getattr(self, '_is_scaling_runway', False):
            self._is_scaling_runway = False

            self.runway_pos_slider.blockSignals(True)
            self.runway_pos_slider.setValue(0)
            self.runway_pos_slider.blockSignals(False)

            # On met à jour l'altitude de référence du slider Z pour éviter
            # qu'il ne "saute" si l'utilisateur l'utilise juste après
            selected_rows = self.runway_table.selectedItems()
            if selected_rows:
                row = selected_rows[0].row()
                self._active_runway_base_z = float(self.runway_table.item(row, 1).text())

    def on_runway_z_slider_changed(self, value):
        selected_rows = self.runway_table.selectedItems()
        if not selected_rows: return
        row = selected_rows[0].row()

        # Le slider va de -1000 à 1000, ce qui représente +/- 100.0 mètres
        new_z = self._active_runway_base_z + (value / 10.0)
        self.runway_table.item(row, 1).setText(f"{new_z:.1f}")
        self.update_runway_preview() # Rafraîchissement 3D en direct

    def toggle_preview_mode(self):
        if self.btn_preview_mode.isChecked():
            self.btn_preview_mode.setText(_("txt_ribbon_mode"))
        else:
            self.btn_preview_mode.setText(_("txt_line_mode"))
        self.update_runway_preview()

    def on_global_z_pressed(self):
        """Mémorise les altitudes de tous les points au moment où on clique sur le slider global."""
        self._global_base_zs = []
        for row in range(self.runway_table.rowCount()):
            z_val = float(self.runway_table.item(row, 1).text())
            self._global_base_zs.append(z_val)

    def on_global_z_changed(self, value):
        """Applique le décalage (delta) à tous les points du tableau."""
        if not hasattr(self, '_global_base_zs') or not self._global_base_zs:
            return

        # On bloque les signaux du tableau pour éviter des boucles d'événements
        self.runway_table.blockSignals(True)
        for row in range(self.runway_table.rowCount()):
            new_z = self._global_base_zs[row] + (value / 10.0)
            self.runway_table.item(row, 1).setText(f"{new_z:.1f}")
        self.runway_table.blockSignals(False)

        # Mise à jour de la visualisation 3D en direct
        self.update_runway_preview()

        # Si un point est actuellement sélectionné, on recalcule la base de son slider
        # individuel pour éviter qu'il ne "saute" si l'utilisateur interagit avec après.
        selected_rows = self.runway_table.selectedItems()
        if selected_rows:
            row = selected_rows[0].row()
            self._active_runway_base_z = float(self.runway_table.item(row, 1).text())
            self.runway_z_slider.blockSignals(True)
            self.runway_z_slider.setValue(0)
            self.runway_z_slider.blockSignals(False)

    def on_global_z_released(self):
        """Recentrez le slider global à 0 pour permettre des ajustements infinis."""
        self.runway_global_z_slider.blockSignals(True)
        self.runway_global_z_slider.setValue(0)
        self.runway_global_z_slider.blockSignals(False)

    def on_runway_rot_pressed(self):
        """Initialise le pivot et le point mobile lors du clic sur le slider de rotation."""
        selected_rows = self.runway_table.selectedItems()
        if not selected_rows: return
        row = selected_rows[0].row()

        # On identifie quel point tourne et quel point sert de pivot
        if row == 0:
            self._rot_moving_idx = 0
            self._rot_pivot_idx = 1
        elif row == self.runway_table.rowCount() - 1:
            self._rot_moving_idx = 1
            self._rot_pivot_idx = 0
        else:
            return

        # On sauvegarde les positions 2D initiales (Avant rotation)
        self._rot_initial_pt = self.runway_pts_2d[self._rot_moving_idx].copy()
        self._rot_pivot_pt = self.runway_pts_2d[self._rot_pivot_idx].copy()

    def on_runway_rot_changed(self, value):
        """Applique la matrice de rotation 2D autour du pivot."""
        if not hasattr(self, '_rot_initial_pt'): return

        # Conversion de la valeur du slider (-450 à 450) en Radians
        angle_deg = value / 10.0
        angle_rad = math.radians(angle_deg)

        cx, cy = self._rot_pivot_pt
        ix, iy = self._rot_initial_pt

        # Formule mathématique de la rotation 2D autour du point (cx, cy)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        nx = cx + (ix - cx) * cos_a - (iy - cy) * sin_a
        ny = cy + (ix - cx) * sin_a + (iy - cy) * cos_a

        # On met à jour l'extrémité dans le tableau de référence 2D
        self.runway_pts_2d[self._rot_moving_idx] = np.array([nx, ny])

        # On déclenche la mise à jour VisPy. Tous les points intermédiaires (t)
        # vont s'aligner automatiquement sans toucher à l'axe Z !
        self.update_runway_preview()

    def on_runway_rot_released(self):
        """Recentrez le slider à 0 pour permettre une rotation infinie."""
        self.runway_rot_slider.blockSignals(True)
        self.runway_rot_slider.setValue(0)
        self.runway_rot_slider.blockSignals(False)

    def load_runway_project(self):
        """Affiche les pistes disponibles pour la tuile en cours et restaure le projet choisi."""
        json_file = self.get_custom_file_path("custom_runways.json")

        # 1. Vérifications de base
        if not os.path.exists(json_file):
            QMessageBox.information(self, _("msg_info_title"), _("msg_custom_rnw_not_found"))
            return

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, _("msg_error_title"), _("msg_json_read_error", error=str(e)))
            return

        tile_id = self.get_current_tile_id()
        if tile_id not in data or not data[tile_id]:
            QMessageBox.information(self, _("msg_info_title"), _("msg_no_runway_found", tile=tile_id))
            return

        # 2. Liste des choix
        runway_names = [rw.get("name", "Unkown") for rw in data[tile_id]]
        name, ok = QInputDialog.getItem(self, _("dialog_load_runway"), _("dialog_avail_runways", tile=tile_id), runway_names, 0, False)
        if not ok or not name: return

        runway_data = next((rw for rw in data[tile_id] if rw.get("name") == name), None)
        if not runway_data: return

        # 3. Injection des données dans le moteur de l'application
        self.cancel_runway() # Nettoie tout tracé en cours
        self.cancel_selection()

        self.runway_active = False
        self.btn_toggle_runway.setChecked(True)
        self.btn_toggle_runway.setText(_("txt_axis_restored"))

        # Verrouillage des autres onglets
        for i in range(self.tabs.count()):
            if i != 3:
                self.tabs.setTabEnabled(i, False)

        # Restauration des paramètres Numpy
        self.runway_pts_2d = [np.array(runway_data["p1"]), np.array(runway_data["p2"])]

        # Restauration des inputs texte (largeur/talus)
        self.current_runway_name = name
        self.current_runway_name_input.setText(name)
        if "width" in runway_data:
            width_val = runway_data["width"]
            self.runway_width_input.setText(str(width_val))
            # Synchronisation du Slider ---
            try:
                self.runway_width_slider.blockSignals(True)
                self.runway_width_slider.setValue(int(float(width_val)))
                self.runway_width_slider.blockSignals(False)
            except ValueError:
                pass
        if "trans_side" in runway_data:
            self.runway_trans_side_input.setText(str(runway_data["trans_side"]))

        # Restauration du profil Z (Le tableau)
        self.runway_table.blockSignals(True)
        self.runway_table.setRowCount(0)
        for pt in runway_data["points"]:
            self._add_row_to_table(pt[0], pt[1])
        self.runway_table.blockSignals(False)

        # 4. Réactivation de l'interface et du rendu 3D
        self.btn_apply_runway.setEnabled(True)
        self.btn_cancel_runway.setEnabled(True)
        self.btn_preview_mode.setEnabled(True)
        self.runway_global_z_slider.setEnabled(True)
        self.btn_save_runway.setEnabled(True)
        self.runway_width_slider.setEnabled(True)
        self.btn_add_point_above.setEnabled(True)
        self.btn_add_point_below.setEnabled(True)
        self.btn_delete_point.setEnabled(True)

        self.update_runway_preview()
        self.canvas.native.setFocus()
        logging.info(f"Runway '{name}' restored.")

    def save_runway_project(self):
        """Exporte le squelette de la piste actuelle dans le fichier custom_runways.json."""
        if len(self.runway_pts_2d) < 2: return

        # 1. Demander le nom de la piste à l'utilisateur
        name, ok = QInputDialog.getText(self, _("dialog_save_runway"), _("dialog_runway_name"))
        if not ok or not name.strip(): return
        name = name.strip()

        tile_id = self.get_current_tile_id()

        # 2. Préparation des données
        pts_2d = [pt.tolist() for pt in self.runway_pts_2d] # Conversion Numpy Array -> Liste standard pour JSON
        table_data = []
        for r in range(self.runway_table.rowCount()):
            pos = float(self.runway_table.item(r, 0).text())
            z = float(self.runway_table.item(r, 1).text())
            table_data.append([pos, z])

        runway_data = {
            "name": name,
            "p1": pts_2d[0],
            "p2": pts_2d[1],
            "points": table_data,
            "width": self.runway_width_input.text(),
            "trans_side": self.runway_trans_side_input.text()
        }

        # 3. Chargement de la base de données existante (si elle existe)
        json_file = self.get_custom_file_path("custom_runways.json")
        data = {}
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                pass # Si le fichier est corrompu, on l'écrase

        if tile_id not in data:
            data[tile_id] = []

        # 4. Vérification d'écrasement (Mise à jour d'une piste existante)
        replaced = False
        for i, existing in enumerate(data[tile_id]):
            if existing.get("name") == name:
                data[tile_id][i] = runway_data
                replaced = True
                break

        if not replaced:
            data[tile_id].append(runway_data)

        # 5. Écriture sur le disque
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.current_runway_name = name
            self.current_runway_name_input.setText(name)
            QMessageBox.warning(self, _("msg_success_title"), _("msg_runway_saved", rnw_name=name, rnw_tile=tile_id))
        except Exception as e:
            QMessageBox.critical(self, _("msg_error_title"), _("msg_json_write_error", error=str(e)))

        self.canvas.native.setFocus()

    def convert_selected_triangles_type(self):
        """Convertit les triangles de la sélection vers le type 0."""
        if self.mesh_visual is None or not self.selected_faces_indices:
            QMessageBox.information(self, _("msg_info_title"), _("msg_plz_select_area"))
            return

        # 1. Conversion en array numpy pour des calculs instantanés
        sel_array = np.array(self.selected_faces_indices, dtype=int)
        types_in_sel = self.original_tri_types[sel_array]

        # 2. Filtre
        # mask_to_convert = (types_in_sel == 8) | (types_in_sel == 16)
        selected_faces = self.original_faces[sel_array]
        water_vertices_mask = self.get_protected_vertices_mask(bitmask_filter=3)
        faces_touching_water = water_vertices_mask[selected_faces].any(axis=1)
        mask_to_convert = (~faces_touching_water) & (types_in_sel != 0)

        if not np.any(mask_to_convert):
            QMessageBox.information(self, _("msg_info_title"), _("msg_no_type_to_convert"))
            return

        # 3. Sauvegarde dans l'historique global AVANT modification (pour le Ctrl+Z)
        self.save_state_to_history(action_name="convert_tri_types")

        # 4. Application vectorielle de la modification
        indices_to_convert = sel_array[mask_to_convert]
        self.original_tri_types[indices_to_convert] = 0
        nbr_tris = len(indices_to_convert)

        self.is_modified = True

        # 5. Rafraîchissement visuel (Les triangles perdront leur couleur violette/rose mais resteront sélectionnés en orange)
        self.update_selection_colors()
        self.canvas.native.setFocus()

        QMessageBox.information(self, _("msg_success_title"), _("msg_tri_converted", tris=nbr_tris))
        logging.info(f"Converted {nbr_tris} triangles (Type 8/16) to Type 0.")

    # =========================================================================
    #
    # 7. ENSEMBLE DES OUTILS MÉTIER 2D (Retouches, Textures & Mosaïques)
    #
    # =========================================================================

    # --- Gestion des Textures ---
    def browse_texture_dir(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier des JPEGs."""
        start_dir = self.tex_dir_input.text() if self.tex_dir_input.text() else self.last_mesh_dir
        dir_path = QFileDialog.getExistingDirectory(self, _("dialog_select_ortho_dir"), start_dir)
        if dir_path:
            self.tex_dir_input.setText(os.path.normpath(dir_path))

    def browse_dds_dir(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier des DDS."""
        start_dir = self.dds_dir_input.text() if self.dds_dir_input.text() else self.last_mesh_dir
        dir_path = QFileDialog.getExistingDirectory(self, _("dialog_select_tex_dir"), start_dir)
        if dir_path:
            self.dds_dir_input.setText(os.path.normpath(dir_path))

    def add_texture(self):
        """Ajoute la texture sous le pivot. Gère un roulement de 4 textures max."""
        tex_dir = self.tex_dir_input.text()
        dds_dir = self.dds_dir_input.text()

        if not tex_dir or not dds_dir or self.mesh_visual is None:
            QMessageBox.warning(self, _("msg_error_title"), _("msg_plz_load_mesh"))
            return

        # 1. Où est le pivot ?
        pivot_x, pivot_y = self.view.camera.center[:2]
        lat, lon = meters_to_latlon(
            pivot_x, pivot_y,
            self.lon_to_m, self.lat_to_m,
            self.x_center, self.y_center
        )

        # 2. Quel est le DDS compilé pour cette zone ?
        best_match = find_best_texture_match(lat, lon, dds_dir)

        if not best_match:
            QMessageBox.warning(self, _("msg_not_found"), _("msg_no_dds", dds_path=dds_dir))
            return

        til_x = best_match['til_x']
        til_y = best_match['til_y']
        zl = best_match['zl']
        provider = best_match['provider']
        expected_jpg_name = f"{til_y}_{til_x}_{provider}{zl}.jpg"

        # 3. ANTI-DOUBLON : On vérifie si cette image n'est pas déjà affichée
        if expected_jpg_name in self.active_texture_names:
            logging.info(f"Texture {expected_jpg_name} already displayed.")
            return

        logging.info(f"Loading texture : {expected_jpg_name} (ZL{zl})")

        # 4. Recherche récursive du JPEG
        img_path = None
        for root, dirs, files in os.walk(tex_dir):
            if expected_jpg_name in files:
                img_path = os.path.join(root, expected_jpg_name)
                break

        if not img_path:
            QMessageBox.warning(self, _("msg_not_found"), _("msg_no_jpg", expected_jpg=expected_jpg_name))
            return

        try:
            # 5. Calcul dynamique de la Bounding Box pour le ZL exact
            n = 2.0 ** zl
            lon_min = (til_x / n) * 360.0 - 180.0
            lon_max = ((til_x + 16) / n) * 360.0 - 180.0
            lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * til_y / n))))
            lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (til_y + 16) / n))))

            # Conversion en Mètres locaux
            x_min = (lon_min - self.x_center) * self.lon_to_m
            x_max = (lon_max - self.x_center) * self.lon_to_m
            y_min = (lat_min - self.y_center) * self.lat_to_m
            y_max = (lat_max - self.y_center) * self.lat_to_m

            # 6. Extraction des triangles couverts par l'image
            uvs = self._calculate_mercator_uvs(x_min, x_max, y_min, y_max)
            u = uvs[:, 0]
            v = uvs[:, 1]

            faces_u = u[self.original_faces]
            faces_v = v[self.original_faces]
            valid_u = (faces_u >= 0.0) & (faces_u <= 1.0)
            valid_v = (faces_v >= 0.0) & (faces_v <= 1.0)
            valid_faces_mask = np.all(valid_u & valid_v, axis=1)
            sub_faces = self.original_faces[valid_faces_mask]

            if len(sub_faces) == 0:
                logging.info("No triangle in mesh matches this texture.")
                return

            # Correction Z-Offset pour éviter le scintillement (Z-Fighting)
            tex_vertices = self.original_vertices.copy()
            # 0.27.2 suppression offset
            # tex_vertices[:, 2] += 10.0 # mètres au-dessus du mesh

            # 7. Chargement de l'image
            img = Image.open(img_path)
            img_data = np.array(img)

            # 8. Création du nouvel objet VisPy
            new_tex_mesh = scene.visuals.Mesh(
                vertices=tex_vertices,
                faces=sub_faces,
                color=(1, 1, 1, 1),
                shading=None
            )

            # --- 0.27.2 OPTIMISATION OPENGL : Superposition parfaite ---
            # order=2 force l'affichage par-dessus le mesh (0) et la mosaïque globale (1)
            new_tex_mesh.set_gl_state(depth_test=True, depth_func='lequal')
            new_tex_mesh.order = 2

            tex_filter = TextureFilter(img_data, uvs)
            new_tex_mesh.attach(tex_filter)
            self.view.add(new_tex_mesh)

            # 9. GESTION DU ROULEMENT (FIFO)
            self.active_texture_meshes.append(new_tex_mesh)
            self.active_texture_names.append(expected_jpg_name)

            if len(self.active_texture_meshes) > self.MAX_TEXTURES:
                # On retire le plus ancien (index 0)
                oldest_mesh = self.active_texture_meshes.pop(0)
                oldest_name = self.active_texture_names.pop(0)

                oldest_mesh.parent = None # Le détruit visuellement et libère la mémoire
                logging.info(f"Rolling: removal of the oldest texture ({oldest_name})")

        except Exception as e:
            QMessageBox.critical(self, _("msg_error_title"), _("msg_texture_error", error=str(e)))

        self.canvas.native.setFocus()

    def clear_textures(self):
        """Détruit tous les calques de textures actuellement affichés."""
        for mesh in self.active_texture_meshes:
            mesh.parent = None # Détache de la scène VisPy

        self.active_texture_meshes.clear()
        self.active_texture_names.clear()
        self.canvas.native.setFocus()
        logging.info("All textures erased from memory.")

    def toggle_texture(self):
        """Bascule l'affichage de la texture satellite SOUS le pivot (Raccourci T)."""
        if getattr(self, 'is_2d_mode', False):
            return

        tex_dir = self.tex_dir_input.text()
        dds_dir = self.dds_dir_input.text()

        # Sécurité : Si les dossiers ne sont pas définis ou s'il n'y a pas de mesh,
        # on délègue à add_texture() qui affichera les bons messages d'erreur.
        if not tex_dir or not dds_dir or self.mesh_visual is None:
            self.add_texture()
            return

        # 1. Quelle est la texture attendue sous le pivot actuel ?
        pivot_x, pivot_y = self.view.camera.center[:2]
        lat, lon = meters_to_latlon(
            pivot_x, pivot_y,
            self.lon_to_m, self.lat_to_m,
            self.x_center, self.y_center
        )

        best_match = find_best_texture_match(lat, lon, dds_dir)

        # S'il n'y a pas de DDS correspondant, on délègue à add_texture() pour l'erreur
        if not best_match:
            self.add_texture()
            return

        expected_jpg_name = f"{best_match['til_y']}_{best_match['til_x']}_{best_match['provider']}{best_match['zl']}.jpg"

        # 2. L'interrupteur (Toggle) intelligent
        if expected_jpg_name in self.active_texture_names:
            # CAS A : La texture est DÉJÀ affichée -> On l'efface elle uniquement
            idx = self.active_texture_names.index(expected_jpg_name)

            # Détachement visuel de la scène VisPy
            mesh_to_remove = self.active_texture_meshes[idx]
            mesh_to_remove.parent = None

            # Suppression stricte des listes de suivi
            self.active_texture_meshes.pop(idx)
            self.active_texture_names.pop(idx)

            logging.info(f"Texture toggled off: {expected_jpg_name} removed.")
        else:
            # CAS B : La texture n'est PAS affichée -> On l'ajoute
            # (La méthode add_texture gère déjà la limite des 4 textures max)
            self.add_texture()

        self.canvas.native.setFocus()

    @wait_cursor
    def generate_global_texture(self, checked=False):
        """Lit les JPEGs, crée la mosaïque en RAM et initialise les objets 3D."""
        tex_dir = self.tex_dir_input.text()
        dds_dir = self.dds_dir_input.text()

        if not tex_dir or not dds_dir or self.mesh_visual is None:
            QMessageBox.warning(self, _("msg_error_title"), _("msg_plz_load_mesh"))
            return

        # SÉCURITÉ : Avertir si des retouches sont en attente d'export
        has_color_edits = getattr(self, 'cumulative_mask', None) is not None and np.any(self.cumulative_mask)
        if has_color_edits or getattr(self, 'has_healing_edits', False):
            reply = QMessageBox.question(self, _("msg_warning_title"), _("msg_you_have_edits_not_export"),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # ==========================================
        # CONSTRUCTION MULTI-THREADS
        # ==========================================
        tex_dir = self.tex_dir_input.text()
        dds_dir = self.dds_dir_input.text() # On a besoin du dossier DDS !

        if not tex_dir or not dds_dir or self.mesh_visual is None:
            self.btn_toggle_global_tex.setChecked(False)
            QMessageBox.warning(self, _("msg_error_title"), _("msg_plz_load_mesh"))
            return

        # --- 1. Création de la Whitelist via le dossier DDS ---
        logging.info("DDS files analysis...")
        dds_pattern = re.compile(r"^(\d+)_(\d+)_([a-zA-Z0-9_-]+?)(\d{2})\.(?:dds|ter)$", re.IGNORECASE)
        valid_compiled_tiles = set()

        try:
            for filename in os.listdir(dds_dir):
                match = dds_pattern.match(filename)
                if match:
                    # On stocke l'empreinte unique : (Y, X, Provider, ZL)
                    valid_compiled_tiles.add((
                        int(match.group(1)),
                        int(match.group(2)),
                        match.group(3).lower(), # Minuscule par sécurité
                        int(match.group(4))
                    ))
        except Exception as e:
            logging.error(f"Error while scanning DDS : {e}")

        # --- 2. Scan strict des JPEGs filtré par la Whitelist ---
        logging.info("Scanning JPEGs files...")
        pattern = re.compile(r"^(\d+)_(\d+)_([a-zA-Z0-9_-]+?)(\d{2})\.(?:jpg|jpeg)$", re.IGNORECASE)
        dir_pattern = re.compile(r"^[a-zA-Z0-9_-]+_\d{2}$", re.IGNORECASE)

        images_info = []

        try:
            # On ne regarde que les dossiers à la racine (0 récursivité)
            for folder_name in os.listdir(tex_dir):
                folder_path = os.path.join(tex_dir, folder_name)

                if os.path.isdir(folder_path) and dir_pattern.match(folder_name):
                    for filename in os.listdir(folder_path):
                        match = pattern.match(filename)
                        if match:
                            y = int(match.group(1))
                            x = int(match.group(2))
                            provider = match.group(3).lower()
                            zl = int(match.group(4))

                            # LE FILTRE STRICT : On ignore la photo si elle n'a pas été compilée en DDS !
                            if (y, x, provider, zl) in valid_compiled_tiles:
                                images_info.append({
                                    'path': os.path.join(folder_path, filename),
                                    'til_y': y,
                                    'til_x': x,
                                    'zl': zl
                                })
        except Exception as e:
            logging.error(f"Error while reading JPEGs : {e}")

        if not images_info:
            self.btn_toggle_global_tex.setChecked(False)
            QMessageBox.warning(self, _("msg_error_title"), _("msg_no_jpg_2"))
            return

        # --- CALCUL DES BORNES (CORRIGÉ POUR TOUS LES ZL) ---
        for info in images_info:
            factor = 2 ** (info['zl'] - 16)
            info['block_x_16'] = (info['til_x'] / factor) / 16.0
            info['block_y_16'] = (info['til_y'] / factor) / 16.0

        min_bx = math.floor(min(i['block_x_16'] for i in images_info))
        min_by = math.floor(min(i['block_y_16'] for i in images_info))

        # 1. Calcul de l'envergure spatiale réelle pour chaque image (Span)
        max_bx_actual = max(info['block_x_16'] + (1.0 / (2 ** (info['zl'] - 16))) for info in images_info)
        max_by_actual = max(info['block_y_16'] + (1.0 / (2 ** (info['zl'] - 16))) for info in images_info)

        # 2. Rétro-compatibilité avec ton système d'indexation
        # On utilise math.ceil pour englober toute la tuile, et -1 pour repasser en "index de bloc maximum"
        max_bx = math.ceil(max_bx_actual) - 1
        max_by = math.ceil(max_by_actual) - 1

        base_size = 512

        # On sauvegarde ces infos dans la classe pour le Batch Export !
        self._global_images_info = images_info
        self._global_min_bx = min_bx
        self._global_max_bx = max_bx
        self._global_min_by = min_by
        self._global_max_by = max_by
        self._global_base_size = base_size

        # Largeur et hauteur exactes du Canvas (le +1 fait son travail correctement maintenant)
        canvas_w = int(max_bx - min_bx + 1) * base_size
        canvas_h = int(max_by - min_by + 1) * base_size

        canvas = Image.new('RGB', (canvas_w, canvas_h), color=(40, 40, 40))
        resample_filter = getattr(Image, 'Resampling', Image).BILINEAR
        images_info.sort(key=lambda x: x['zl'])

        # --- TRAVAIL POUR LES THREADS ---
        def process_single_image(info):
            try:
                with Image.open(info['path']) as img:
                    factor = 2 ** (info['zl'] - 16)
                    size = int(base_size / factor)
                    img_resized = img.resize((size, size), resample_filter)
                    px = int((info['block_x_16'] - min_bx) * base_size)
                    py = int((info['block_y_16'] - min_by) * base_size)
                    return img_resized, px, py, size
            except Exception:
                return None

        logging.info(f"Parallel processing of {len(images_info)} images...")
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_single_image, images_info))

        self._global_grid_rects = []

        for res in results:
            if res:
                img_data, px, py, size = res
                canvas.paste(img_data, (px, py))
                self._global_grid_rects.append((px, py, size, size)) # CAPTURE

        logging.info("Generation of UVs and GPU upload...")

        # --- SAUVEGARDE DU CANVAS EN MÉMOIRE POUR LE CAS 2 ---
        self._global_canvas_data = np.array(canvas)
        self._original_canvas_data = self._global_canvas_data.copy() # COPIE SANCTUARISÉE

        # Flag pour savoir si on doit rafraîchir la 3D
        self._texture_needs_refresh = False

        # Création du masque cumulatif des retouches
        h, w = self._global_canvas_data.shape[:2]
        self.cumulative_mask = np.zeros((h, w), dtype=np.uint8)

        # --- COORDONNÉES GÉOGRAPHIQUES ---
        x_left, x_right = min_bx * 16, (max_bx + 1) * 16
        y_top, y_bottom = min_by * 16, (max_by + 1) * 16
        n16 = 2.0 ** 16
        lon_min = (x_left / n16) * 360.0 - 180.0
        lon_max = (x_right / n16) * 360.0 - 180.0
        lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_top / n16))))
        lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y_bottom) / n16))))

        x_min_m = (lon_min - self.x_center) * self.lon_to_m
        x_max_m = (lon_max - self.x_center) * self.lon_to_m
        y_min_m = (lat_min - self.y_center) * self.lat_to_m
        y_max_m = (lat_max - self.y_center) * self.lat_to_m

        # SAUVEGARDE DES BORNES # POUR LE CAS 2
        self._global_bounds = (x_min_m, x_max_m, y_min_m, y_max_m)
        global_uvs = self._calculate_mercator_uvs(x_min_m, x_max_m, y_min_m, y_max_m)

        # --- 0.27.2 : Filtrage des faces et Offset Z ---
        self._build_filtered_global_texture_mesh(global_uvs)

        tex_filter = TextureFilter(self._global_canvas_data, global_uvs)
        self.global_texture_mesh.attach(tex_filter)

        # Met à jour l'UI
        self.btn_toggle_global_tex.setEnabled(True)
        self.btn_toggle_global_tex.setChecked(True)
        self.toggle_global_texture_display() # Applique l'affichage

        logging.info("Global view done !")

    def toggle_global_texture_display(self):
        """Gère l'affichage de la mosaïque, et la génère si elle n'existe pas encore."""

        # 1. CAS A : La mosaïque n'a pas encore été générée
        if not getattr(self, 'global_texture_mesh', None):
            # Sécurité UI : on empêche le bouton de rester "enfoncé" s'il y a une erreur de chargement
            self.btn_toggle_global_tex.blockSignals(True)
            self.btn_toggle_global_tex.setChecked(False)
            self.btn_toggle_global_tex.blockSignals(False)

            # On dévie vers l'action lourde de génération
            self.generate_global_texture()
            return

        # 2. CAS B : La mosaïque existe déjà (Comportement normal d'interrupteur)
        is_checked = self.btn_toggle_global_tex.isChecked()
        self.global_texture_mesh.visible = is_checked

        if self.mesh_visual is not None:
            # self.mesh_visual.visible = not is_checked
            # 0.27.2
            self.mesh_visual.visible = True

        self.canvas.native.setFocus()

    # --- Mode Retouche 2D & Lasso ---
    def toggle_2d_mode(self):
        """Bascule entre 3D et 2D avec masquage exhaustif et persistance des vues."""
        is_checked = self.btn_toggle_2d.isChecked()

        # --- SÉCURITÉ : Vérification de la présence de l'image ---
        if is_checked and getattr(self, '_global_canvas_data', None) is None:
            QMessageBox.warning(self, _("msg_warning_title"), _("msg_please_generate_the_globa"))
            self.btn_toggle_2d.setChecked(False)
            self.is_2d_mode = False
            return

        self.is_2d_mode = is_checked

        if is_checked:
            # 1. Sauvegarde de l'état de la caméra 3D
            self.camera_3d_state = {
                'center': self.view.camera.center,
                'scale_factor': self.view.camera.scale_factor,
                'azimuth': self.view.camera.azimuth,
                'elevation': self.view.camera.elevation
            }

            # 2. Masquage exhaustif de TOUS les éléments 3D
            # On boucle sur les attributs simples pour gagner en clarté
            for attr in ['mesh_visual', 'global_texture_mesh', 'airport_markers',
                        'pivot_marker', 'pivot_line', 'flatten_plane',
                        'cylinder_visual', 'point_proxy_line', 'point_proxy_marker']:
                obj = getattr(self, attr, None)
                if obj: obj.visible = False

            # On boucle sur les listes d'objets visuels (pistes, zones complexes)
            for vis_list in [getattr(self, 'runway_visuals', []), getattr(self, 'zone_visuals', [])]:
                for vis in vis_list:
                    vis.visible = False

            # Textures locales HD
            for tex_visual in self.active_texture_meshes:
                tex_visual.visible = False

            # 3. Gestion de l'Image 2D
            if self.image_2d_visual is None:
                self.image_2d_visual = scene.visuals.Image(self._global_canvas_data, parent=self.view.scene)
                self.image_2d_visual.order = 0
            else:
                # On force l'envoi des données à la carte graphique ---
                self.image_2d_visual.set_data(self._global_canvas_data)
                self.image_2d_visual.visible = True

            # --- Création du calque de sélection ---
            if self.selection_visual is None:
                h, w = self._global_canvas_data.shape[:2]
                empty_overlay = np.zeros((h, w, 4), dtype=np.uint8)
                self.selection_visual = scene.visuals.Image(
                    empty_overlay, parent=self.view.scene, interpolation='nearest'
                )

                # blend=True est obligatoire pour les images RGBA
                self.selection_visual.set_gl_state('translucent', blend=True, depth_test=False)

                # On s'assure qu'il se dessine par-dessus l'image de fond
                self.selection_visual.order = 1
            else:
                has_mask = self.selection_mask is not None and bool(np.any(self.selection_mask))
                self.selection_visual.visible = has_mask

            # 4. Gestion de la Grille 2D
            if self.grid_2d_visual is None and getattr(self, '_global_grid_rects', []):
                lines = []
                for (x, y, w, h) in self._global_grid_rects:
                    lines.extend([[x, y], [x + w, y], [x + w, y], [x + w, y + h],
                                [x + w, y + h], [x, y + h], [x, y + h], [x, y]])
                if lines:
                    self.grid_2d_visual = scene.visuals.Line(pos=np.array(lines), color='yellow',
                                                            connect='segments', parent=self.view.scene)
                    self.grid_2d_visual.set_gl_state('translucent', depth_test=False)

            if self.grid_2d_visual:
                self.grid_2d_visual.visible = self.btn_toggle_grid_2d.isChecked()

            # 5. Configuration de la Caméra 2D (PanZoom)
            self.view.camera = scene.cameras.PanZoomCamera(aspect=1)
            self.view.camera.flip = (False, True) # Inversion Y pour l'image

            if self.camera_2d_state and 'rect' in self.camera_2d_state:
                # On restaure le rectangle de vue exact
                self.view.camera.rect = self.camera_2d_state['rect']
            else:
                h, w = self._global_canvas_data.shape[:2]
                self.view.camera.set_range(x=(0, w), y=(0, h), margin=0.05)

            # --- VERROUILLAGE UX (MODE 2D) ---
            # 6. On grise les onglets 0 à 4 (Mesh, Vue, Aéroport, Altiport, Relief)
            for i in range(5):
                self.tabs.setTabEnabled(i, False)

            # 7. On déverrouille le contenu de l'onglet Édition
            self.group_selection.setEnabled(True)
            self.group_color.setEnabled(True)
            self.group_batch.setEnabled(True)
            self.group_global_tex.setEnabled(True)
            self.btn_batch_export.setEnabled(True)
            self.btn_reset_retouches.setEnabled(True)

            self.btn_toggle_2d.setText(_("txt_disable_2d_retouch_mode"))
            self.btn_toggle_grid_2d.setEnabled(True)
            self.btn_toggle_mesh_mask.setEnabled(True)

        else:
            # === RETOUR EN MODE 3D ===

            # 1. Sauvegarde de la position 2D avant de quitter (PanZoom utilise 'rect')
            self.camera_2d_state = {
                'rect': self.view.camera.rect
            }

            # 2. Masquage de la 2D
            if self.image_2d_visual: self.image_2d_visual.visible = False
            if self.grid_2d_visual: self.grid_2d_visual.visible = False
            if self.selection_visual: self.selection_visual.visible = False
            if self.polygon_visual: self.polygon_visual.visible = False
            if getattr(self, 'polygon_markers_visual', None):
                self.polygon_markers_visual.visible = False
            if getattr(self, 'retouch_preview_visual', None):
                self.retouch_preview_visual.visible = False

            # 3. Restauration de la Caméra 3D
            self.view.camera = scene.cameras.TurntableCamera(fov=45)
            if self.camera_3d_state:
                self.view.camera.center = self.camera_3d_state['center']
                self.view.camera.scale_factor = self.camera_3d_state['scale_factor']
                self.view.camera.azimuth = self.camera_3d_state['azimuth']
                self.view.camera.elevation = self.camera_3d_state['elevation']

            # 4. Ré-affichage intelligent des éléments 3D
            if getattr(self, 'btn_toggle_global_tex', None) and self.btn_toggle_global_tex.isChecked():
                if getattr(self, '_global_canvas_data', None) is not None:

                    # On vérifie le booléen
                    if getattr(self, '_texture_needs_refresh', False):
                        if getattr(self, 'global_texture_mesh', None):
                            self.global_texture_mesh.parent = None

                        global_uvs = self._calculate_mercator_uvs(*self._global_bounds)

                        # --- 0.27.2 : Filtrage des faces et Offset Z ---
                        self._build_filtered_global_texture_mesh(global_uvs)

                        tex_filter = TextureFilter(self._global_canvas_data, global_uvs)
                        self.global_texture_mesh.attach(tex_filter)

                        # On remet le flag à False puisqu'on vient de faire la mise à jour !
                        self._texture_needs_refresh = False

                    # Dans TOUS les cas (rafraîchi ou non), on s'assure qu'il est visible
                    if getattr(self, 'global_texture_mesh', None):
                        self.global_texture_mesh.visible = True

                    # 0.27.2 : Le mesh de base doit toujours être restauré
                    if self.mesh_visual:
                        self.mesh_visual.visible = True
            else:
                if self.mesh_visual: self.mesh_visual.visible = True

            # 5. Ré-affichage des marqueurs si l'option était active
            if getattr(self, 'show_airports', False) and getattr(self, 'airport_markers', None):
                self.airport_markers.visible = True

            if getattr(self, 'pivot_marker', None): self.pivot_marker.visible = True
            if getattr(self, 'pivot_line', None): self.pivot_line.visible = True

            # Ré-affichage des listes d'objets (pistes, zones)
            for vis_list in [getattr(self, 'runway_visuals', []), getattr(self, 'zone_visuals', [])]:
                for vis in vis_list:
                    vis.visible = True

            # Textures locales HD
            for tex_visual in self.active_texture_meshes:
                tex_visual.visible = True

            # --- VERROUILLAGE UX (MODE 3D) ---
            # 6. On dégrise les onglets 0 à 4
            for i in range(5):
                self.tabs.setTabEnabled(i, True)

            # 7. On verrouille le contenu de l'onglet Édition
            self.group_selection.setEnabled(False)
            self.group_color.setEnabled(False)
            self.group_batch.setEnabled(False)
            self.group_global_tex.setEnabled(False)
            self.btn_batch_export.setEnabled(False)
            self.btn_reset_retouches.setEnabled(False)

            if getattr(self, 'is_blur_mode', False):
                self.btn_toggle_meansh.setChecked(False)
                self.toggle_meansh_mode()

            if getattr(self, 'is_seamless_mode', False):
                self.btn_toggle_seamless.setChecked(False)
                self.toggle_seamless_mode()

            self.btn_toggle_2d.setText(_("btn_enable_2d_retouch_mode"))
            self.btn_toggle_grid_2d.setEnabled(False)
            self.btn_toggle_mesh_mask.setEnabled(False)

            if hasattr(self, 'update_undo_redo_buttons'):
                self.update_undo_redo_buttons()

            # Réinitialisation de l'état batch
            self.batch_2d_state = None
            if hasattr(self, 'btn_apply_all_sel_2d'):
                self.btn_apply_all_sel_2d.setEnabled(False)

        self.canvas.native.setFocus()

    def _build_filtered_global_texture_mesh(self, global_uvs):
        """
        Filtre les faces dont les sommets sont dans les limites UV [0, 1]
        et reconstruit le maillage 3D global de la texture.
        """
        u = global_uvs[:, 0]
        v = global_uvs[:, 1]

        faces_u = u[self.original_faces]
        faces_v = v[self.original_faces]

        valid_u = (faces_u >= 0.0) & (faces_u <= 1.0)
        valid_v = (faces_v >= 0.0) & (faces_v <= 1.0)
        valid_faces_mask = np.all(valid_u & valid_v, axis=1)

        sub_faces = self.original_faces[valid_faces_mask]
        tex_vertices = self.original_vertices.copy()

        if getattr(self, 'global_texture_mesh', None):
            self.global_texture_mesh.parent = None

        self.global_texture_mesh = scene.visuals.Mesh(
            vertices=tex_vertices,
            faces=sub_faces,
            color=(1, 1, 1, 1),
            shading=None,
            parent=self.view.scene
        )

        # --- OPTIMISATION OPENGL : Rendu en surcouche parfaite ---
        # order=1 force le rendu après le maillage de base (order=0)
        self.global_texture_mesh.set_gl_state(depth_test=True, depth_func='lequal')
        self.global_texture_mesh.order = 1

    def toggle_2d_grid(self, checked):
        """Affiche ou masque la grille mathématique par-dessus l'image 2D."""
        if getattr(self, 'grid_2d_visual', None):
            self.grid_2d_visual.visible = checked

        self.canvas.native.setFocus()

    def toggle_mesh_mask(self):
        """Affiche ou masque un calque sombre sur les zones hors du mesh 3D."""
        if getattr(self, '_global_canvas_data', None) is None or getattr(self, 'image_2d_visual', None) is None:
            return

        is_checked = self.btn_toggle_mesh_mask.isChecked()

        if is_checked:
            # Sécurité pro : on recrée le visuel s'il n'existe pas OU si son parent a changé
            # (au cas où la mosaïque 2D est détruite/recréée lors d'un rechargement de texture)
            if getattr(self, 'mesh_mask_visual', None) is None or self.mesh_mask_visual.parent != self.image_2d_visual:
                h, w = self._global_canvas_data.shape[:2]

                # Création d'un fond gris sombre translucide
                mask_rgba = np.full((h, w, 4), (20, 20, 20, 180), dtype=np.uint8)

                # Conversion des UVs globaux en coordonnées de pixels
                global_uvs = self._calculate_mercator_uvs(*self._global_bounds)
                pts_2d = np.column_stack((global_uvs[:, 0] * w, global_uvs[:, 1] * h)).astype(np.int32)

                # Calcul de l'enveloppe convexe
                hull = cv2.convexHull(pts_2d)

                # Évidage du centre (transparence sur le mesh)
                cv2.fillPoly(mask_rgba, [hull], (0, 0, 0, 0))

                # === MODIFICATION ARCHITECTURALE CRITIQUE ===
                # Le parent devient 'self.image_2d_visual' au lieu de 'self.view.scene'
                self.mesh_mask_visual = scene.visuals.Image(
                    mask_rgba, parent=self.image_2d_visual, interpolation='nearest'
                )
                self.mesh_mask_visual.set_gl_state('translucent', depth_test=False, blend=True)

                # S'affiche juste au-dessus de son parent direct (l'image de fond)
                self.mesh_mask_visual.order = 1
            else:
                self.mesh_mask_visual.visible = True
        else:
            if getattr(self, 'mesh_mask_visual', None) is not None:
                self.mesh_mask_visual.visible = False

        self.canvas.native.setFocus()

    def update_selection_visual(self):
        """Met à jour l'affichage du masque (remplissage rouge OU contour épais) et synchronise l'UI."""
        if getattr(self, 'batch_2d_state', None) == "show_masks":
            return

        has_mask = getattr(self, 'selection_mask', None) is not None and bool(np.any(self.selection_mask))

        # --- 1. MISE À JOUR VISUELLE VISPY ---
        if not has_mask:
            if getattr(self, 'selection_visual', None):
                self.selection_visual.visible = False
        else:
            h, w = self.selection_mask.shape
            rgba_mask = np.zeros((h, w, 4), dtype=np.uint8)

            show_outline = hasattr(self, 'chk_show_outline') and self.chk_show_outline.isChecked()

            if show_outline:
                # --- CALCUL DYNAMIQUE DE L'ÉPAISSEUR ---
                if hasattr(self.view.camera, 'rect') and self.view.camera.rect is not None:
                    visible_width = abs(self.view.camera.rect.width)
                    zoom_ratio = visible_width / w

                    # On cible une plage de 3 à 9 pour garantir une bonne visibilité
                    thickness = max(3, min(9, int(8 * zoom_ratio)))
                else:
                    thickness = 9

                # 1. FORCE IMPAIRE : Garantie absolue d'une symétrie parfaite du kernel
                if thickness % 2 == 0:
                    thickness += 1

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))

                # 2. INTERNAL GRADIENT : On érode (rétrécit) le masque, puis on soustrait
                eroded_mask = cv2.erode(self.selection_mask, kernel, iterations=1)

                # Le contour résultant sera strictement contenu DANS les limites originales
                outline_mask = cv2.subtract(self.selection_mask, eroded_mask)

                rgba_mask[outline_mask > 0] = [0, 255, 255, 255]
            else:
                rgba_mask[self.selection_mask == 255] = [255, 0, 0, 100]

            if not getattr(self, 'selection_visual', None):
                self.selection_visual = scene.visuals.Image(
                    rgba_mask, parent=self.view.scene, interpolation='nearest'
                )
                self.selection_visual.set_gl_state('translucent', depth_test=False, blend=True)
                self.selection_visual.order = 2
            else:
                self.selection_visual.method = 'auto'
                self.selection_visual.set_data(rgba_mask)
                self.selection_visual.visible = True

        # --- 2. SYNCHRONISATION UNIVERSELLE DE L'INTERFACE ---
        # Tous les éléments UI s'alignent automatiquement sur la présence (ou non) du masque
        for key in getattr(self, 'sliders_color', {}):
            self.sliders_color[key].setEnabled(has_mask)
        for key in getattr(self, 'spinboxes_color', {}):
            self.spinboxes_color[key].setEnabled(has_mask)

        if hasattr(self, 'btn_apply_color'):
            self.btn_apply_color.setEnabled(has_mask)
            self.btn_cancel_color.setEnabled(has_mask)
        if hasattr(self, 'btn_save_sel_2d'):
            self.btn_save_sel_2d.setEnabled(has_mask)
        if hasattr(self, 'btn_crop_sel_2d'):
            self.btn_crop_sel_2d.setEnabled(has_mask)
        if hasattr(self, 'btn_save_colors'):
            self.btn_save_colors.setEnabled(has_mask)
            self.btn_load_colors.setEnabled(has_mask)
        if hasattr(self, 'btn_reset_local_retouch'):
            self.btn_reset_local_retouch.setEnabled(has_mask)

    def clear_selection_mask(self):
        """Vide le masque de sélection 2D actif."""
        self.selection_mask = None

        if hasattr(self, 'cancel_color_retouch'):
            self.cancel_color_retouch()

        # On vide le champ du nom en cours
        if hasattr(self, 'current_sel_name_input'):
            self.current_sel_name_input.clear()

        self.update_selection_visual()
        self.set_selection_state(name="", modified=False)
        self.canvas.native.setFocus()

        logging.info("2D selection mask cleared.")

    def update_polygon_visual(self):
        """Met à jour le tracé bleu et les marqueurs du polygone en cours."""
        if not self.polygon_points:
            # S'il n'y a plus aucun point (ex: après un Undo ou une validation)
            if self.polygon_visual: self.polygon_visual.visible = False
            if self.polygon_markers_visual: self.polygon_markers_visual.visible = False
            return

        pts = np.array(self.polygon_points, dtype=np.float32) + 0.5

        # Choisir la couleur selon le mode
        draw_color = 'red' if getattr(self, 'lasso_mode', 'add') == 'sub' else 'cyan'

        # 1. Gestion de la ligne (nécessite au moins 2 points)
        if len(self.polygon_points) >= 2:
            if self.polygon_visual is None:
                self.polygon_visual = scene.visuals.Line(
                    pos=pts, color=draw_color, width=2, parent=self.view.scene
                )
                self.polygon_visual.set_gl_state('translucent', depth_test=False)
                self.polygon_visual.order = 3
            else:
                self.polygon_visual.set_data(pos=pts, color=draw_color) # On force la couleur
                self.polygon_visual.visible = True
        else:
            if self.polygon_visual: self.polygon_visual.visible = False

        # 2. Gestion des marqueurs (losanges, visibles dès le 1er point)
        if self.polygon_markers_visual is None:
            self.polygon_markers_visual = scene.visuals.Markers(parent=self.view.scene)
            self.polygon_markers_visual.set_gl_state('translucent', depth_test=False)
            self.polygon_markers_visual.order = 4 # Au-dessus de la ligne

        self.polygon_markers_visual.set_data(
            pos=pts,
            symbol='diamond', # Formes possibles : 'o' (cercle), 's' (carré), 'diamond', '+'
            edge_color='cyan',
            face_color='white',
            size=8
        )
        self.polygon_markers_visual.visible = True
        self.update_undo_redo_buttons()

    def apply_polygon_selection(self):
        """Ferme le polygone, génère un masque et l'applique selon le mode booléen choisi."""
        if len(self.polygon_points) < 3:
            self.polygon_points = []
            self.update_polygon_visual()
            return

        if hasattr(self._global_canvas_data, 'shape'):
            h, w = self._global_canvas_data.shape[:2]
        else:
            w, h = self._global_canvas_data.size

        poly_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([self.polygon_points], dtype=np.int32)
        cv2.fillPoly(poly_mask, pts, 255)

        if self.selection_mask is None:
            self.selection_mask = poly_mask.copy()
        elif self.lasso_mode == "add":
            self.selection_mask = cv2.bitwise_or(self.selection_mask, poly_mask)
        elif self.lasso_mode == "sub":
            self.selection_mask = cv2.bitwise_and(self.selection_mask, cv2.bitwise_not(poly_mask))

        if hasattr(self, 'cancel_color_retouch'):
            self.cancel_color_retouch()

        self.update_selection_visual() # La tour de contrôle dégrise l'UI toute seule !
        self.set_selection_state(modified=True)
        self.polygon_points = []
        self.update_polygon_visual()
        self.canvas.native.setFocus()
        logging.info(f"Boolean selection applied (Mode: {self.lasso_mode}).")

    def undo_polygon_point(self):
        """Annule le dernier point placé sur le lasso polygonal."""
        if getattr(self, 'is_2d_mode', False) and self.polygon_points:
            # On retire le dernier point et on le stocke dans le Redo
            point = self.polygon_points.pop()
            self.polygon_redo_stack.append(point)
            self.update_polygon_visual()

    def redo_polygon_point(self):
        """Rétablit le dernier point annulé sur le lasso polygonal."""
        if getattr(self, 'is_2d_mode', False) and getattr(self, 'polygon_redo_stack', []):
            # On récupère le dernier point annulé et on le remet dans le tracé
            point = self.polygon_redo_stack.pop()
            self.polygon_points.append(point)
            self.update_polygon_visual()

    def set_lasso_mode(self, mode):
        self.lasso_mode = mode
        self.update_polygon_visual() # Met à jour la couleur si on change en cours de tracé

    def save_selection_project(self):
        """Exporte le masque à l'écran en COORDONNÉES ABSOLUES vers custom_2D_selections.json."""
        if getattr(self, 'selection_mask', None) is None or not bool(np.any(self.selection_mask)):
            return

        tile_id = self.get_current_tile_id()
        # NOUVEAU FICHIER !
        json_file = self.get_custom_file_path("custom_2D_selections.json")
        data = {}
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f: data = json.load(f)
            except Exception: pass

        if tile_id not in data: data[tile_id] = []
        existing_names = [s.get("name") for s in data[tile_id] if s.get("name")]

        default_name = self.current_sel_name_input.text().strip() if hasattr(self, 'current_sel_name_input') else ""
        if default_name and default_name not in existing_names:
            existing_names.insert(0, default_name)

        name, ok = QInputDialog.getItem(self, _("msg_save_selection"), _("msg_name_cust", ti=tile_id), existing_names, 0, True)
        if not ok or not name.strip(): return
        name = name.strip()

        if name in existing_names:
            reply = QMessageBox.question(self, _("msg_confirmation"), _("msg_replace_cust", cn=name),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No: return

        # Extraction des contours
        res = cv2.findContours(self.selection_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        hierarchy = res[1] if len(res) == 2 else res[2]

        pos_polys_px, neg_polys_px = [], []

        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                if len(cnt) >= 3:
                    poly_list = cnt.reshape(-1, 2).tolist()
                    parent_idx = hierarchy[0][i][3]
                    if parent_idx == -1:
                        pos_polys_px.append(poly_list)
                    else:
                        neg_polys_px.append(poly_list)

        if not pos_polys_px:
            QMessageBox.warning(self, _("msg_error_title"), _("msg_mask_too_small_to_save"))
            return

        # === CONVERSION EN COORDONNÉES ABSOLUES AVANT SAUVEGARDE ===
        pos_polys = self._pixels_to_meters(pos_polys_px)
        neg_polys = self._pixels_to_meters(neg_polys_px)

        sel_data = {
            "name": name,
            "polygons": pos_polys,
            "negative_polygons": neg_polys
        }

        replaced = False
        for i, existing in enumerate(data[tile_id]):
            if existing.get("name") == name:
                data[tile_id][i] = sel_data
                replaced = True; break

        if not replaced: data[tile_id].append(sel_data)

        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            QMessageBox.information(self, _("msg_success_title"), _("msg_selection_saved", cn=name))
            self.set_selection_state(name=name, modified=False)
        except Exception as e:
            QMessageBox.critical(self, _("msg_error_title"), _("msg_json_write_error", error=e))

        self.canvas.native.setFocus()

    def crop_by_saved_selection(self):
        """Soustrait fidèlement une zone sauvegardée (y compris ses trous) du masque actuel."""
        if getattr(self, 'selection_mask', None) is None or not bool(np.any(self.selection_mask)): return

        # NOUVEAU FICHIER
        json_file = self.get_custom_file_path("custom_2D_selections.json")
        if not os.path.exists(json_file): return

        try:
            with open(json_file, 'r', encoding='utf-8') as f: data = json.load(f)
        except Exception: return

        tile_id = self.get_current_tile_id()
        if tile_id not in data or not data[tile_id]: return

        sel_names = [s.get("name", "Unkown") for s in data[tile_id]]
        name, ok = QInputDialog.getItem(self, _("msg_crop_the_mask"), _("msg_choose_the_area_to_subtra"), sel_names, 0, False)
        if not ok or not name: return

        sel_data = next((s for s in data[tile_id] if s.get("name") == name), None)
        if not sel_data: return

        h, w = self.selection_mask.shape
        crop_mask = np.zeros((h, w), dtype=np.uint8)

        # === CONVERSION EN PIXELS AVANT AFFICHAGE ===
        pos_polys_px = self._meters_to_pixels(sel_data.get("polygons", []))
        neg_polys_px = self._meters_to_pixels(sel_data.get("negative_polygons", []))

        # 1. On dessine les formes pleines
        for poly in pos_polys_px:
            cv2.fillPoly(crop_mask, [np.array(poly, dtype=np.int32)], 255)
        # 2. On évide les éventuels trous de l'emporte-pièce
        for poly in neg_polys_px:
            cv2.fillPoly(crop_mask, [np.array(poly, dtype=np.int32)], 0)

        # MASQUAGE BOOLÉEN
        self.selection_mask = cv2.bitwise_and(self.selection_mask, cv2.bitwise_not(crop_mask))

        if hasattr(self, 'cancel_color_retouch'):
            self.cancel_color_retouch()

        self.update_selection_visual()
        self.set_selection_state(modified=True)
        self.canvas.native.setFocus()

        logging.info(f"Mask cropped by '{name}'.")

    def load_selection_project(self):
        """Charge un masque sauvegardé en COORDONNÉES ABSOLUES (custom_2D_selections.json)."""
        if not getattr(self, 'is_2d_mode', False) or getattr(self, '_global_canvas_data', None) is None:
            return

        json_file = self.get_custom_file_path("custom_2D_selections.json")
        if not os.path.exists(json_file):
            QMessageBox.information(self, _("msg_info_title"), _("msg_no_2d_selection_file"))
            return

        try:
            with open(json_file, 'r', encoding='utf-8') as f: data = json.load(f)
        except Exception: return

        tile_id = self.get_current_tile_id()
        if tile_id not in data or not data[tile_id]: return

        sel_names = [s.get("name", "Unknown") for s in data[tile_id]]
        name, ok = QInputDialog.getItem(self, _("msg_load_a_selection"), _("msg_tile_selections", ti=tile_id), sel_names, 0, False)
        if not ok or not name: return

        sel_data = next((s for s in data[tile_id] if s.get("name") == name), None)
        if not sel_data: return

        self.clear_selection_mask()

        if hasattr(self._global_canvas_data, 'shape'):
            h, w = self._global_canvas_data.shape[:2]
        else:
            w, h = self._global_canvas_data.size

        poly_mask = np.zeros((h, w), dtype=np.uint8)

        # === CONVERSION EN PIXELS AVANT AFFICHAGE ===
        pos_polys_px = self._meters_to_pixels(sel_data.get("polygons", []))
        neg_polys_px = self._meters_to_pixels(sel_data.get("negative_polygons", []))

        for poly in pos_polys_px:
            cv2.fillPoly(poly_mask, [np.array(poly, dtype=np.int32)], 255)
        for poly in neg_polys_px:
            cv2.fillPoly(poly_mask, [np.array(poly, dtype=np.int32)], 0)

        self.selection_mask = poly_mask
        self.update_selection_visual()

        if hasattr(self, 'current_sel_name_input'):
            self.current_sel_name_input.setText(name)

        self.set_selection_state(name=name, modified=False)
        self.canvas.native.setFocus()

    def set_selection_state(self, name=None, modified=None):
        """Met à jour le nom interne et gère l'affichage de l'astérisque de modification."""
        if name is not None:
            self.current_sel_name = name
        if modified is not None:
            self.is_selection_modified = modified

        if not hasattr(self, 'current_sel_name_input'): return

        display_name = self.current_sel_name
        if not display_name:
            self.current_sel_name_input.setText("")
            return

        if self.is_selection_modified:
            display_name += " *"

        self.current_sel_name_input.setText(display_name)

    def reset_selection_retouch(self):
        """Restaure la mosaïque d'origine et efface le masque cumulatif uniquement sous la sélection active."""
        if not getattr(self, 'is_2d_mode', False) or getattr(self, '_original_canvas_data', None) is None:
            return

        if getattr(self, 'selection_mask', None) is None or not np.any(self.selection_mask):
            return

        # 1. Demande de confirmation
        reply = QMessageBox.question(self, _("msg_warning"), _("msg_reset_color"),
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        # 2. Bounding Box pour limiter drastiquement la charge CPU (Optimisation OpenCV)
        x, y, w, h = cv2.boundingRect(self.selection_mask)

        # 3. Masque booléen local
        roi_mask = self.selection_mask[y:y+h, x:x+w]
        active_mask = roi_mask > 0

        # 4. Restauration ciblée des pixels originaux via masquage booléen NumPy
        original_roi = self._original_canvas_data[y:y+h, x:x+w]
        current_roi = self._global_canvas_data[y:y+h, x:x+w]
        current_roi[active_mask] = original_roi[active_mask]

        # 5. Effacement de la zone dans le masque cumulatif global (pour l'export)
        if getattr(self, 'cumulative_mask', None) is not None:
            cumulative_roi = self.cumulative_mask[y:y+h, x:x+w]
            cumulative_roi[active_mask] = 0

        # 6. Mise à jour des drapeaux d'état
        self._texture_needs_refresh = True
        self.is_retouch_exported = False

        # 7. Rafraîchissement direct de VisPy (Mode 2D)
        if getattr(self, 'image_2d_visual', None):
            self.image_2d_visual.set_data(self._global_canvas_data)

        # 8. Réinitialisation des sliders pour éviter toute confusion
        self.cancel_color_retouch()

        self.canvas.native.setFocus()
        logging.info("Local retouch reset applied for the current selection.")

    def _pixels_to_meters(self, polys_pixels):
        """Convertit des polygones en coordonnées Pixel (2D) vers Mètres locaux (Absolus)."""
        if not polys_pixels or getattr(self, '_global_bounds', None) is None: return []
        x_min_m, x_max_m, y_min_m, y_max_m = self._global_bounds
        h, w = self._global_canvas_data.shape[:2]

        lat_max = (y_max_m / self.lat_to_m) + self.y_center
        lat_min = (y_min_m / self.lat_to_m) + self.y_center

        m_y_max = np.log(np.tan(np.radians(lat_max)) + 1/np.cos(np.radians(lat_max)))
        m_y_min = np.log(np.tan(np.radians(lat_min)) + 1/np.cos(np.radians(lat_min)))

        polys_meters = []
        for poly in polys_pixels:
            arr = np.array(poly, dtype=np.float64)
            if len(arr) == 0: continue

            u = arr[:, 0] / w
            v = arr[:, 1] / h

            x_m = x_min_m + u * (x_max_m - x_min_m)
            m_y = m_y_max - v * (m_y_max - m_y_min)
            lats = np.degrees(np.arctan(np.sinh(m_y)))
            y_m = (lats - self.y_center) * self.lat_to_m

            poly_m = np.column_stack((x_m, y_m)).tolist()
            polys_meters.append(poly_m)
        return polys_meters

    def _meters_to_pixels(self, polys_meters):
        """Convertit des polygones en Mètres locaux (Absolus) vers Pixel (2D)."""
        if not polys_meters or getattr(self, '_global_bounds', None) is None: return []
        x_min_m, x_max_m, y_min_m, y_max_m = self._global_bounds
        h, w = self._global_canvas_data.shape[:2]

        lat_max = (y_max_m / self.lat_to_m) + self.y_center
        lat_min = (y_min_m / self.lat_to_m) + self.y_center

        m_y_max = np.log(np.tan(np.radians(lat_max)) + 1/np.cos(np.radians(lat_max)))
        m_y_min = np.log(np.tan(np.radians(lat_min)) + 1/np.cos(np.radians(lat_min)))

        polys_pixels = []
        for poly in polys_meters:
            arr = np.array(poly, dtype=np.float64)
            if len(arr) == 0: continue

            x_m = arr[:, 0]
            y_m = arr[:, 1]

            u = (x_m - x_min_m) / (x_max_m - x_min_m)
            lats = (y_m / self.lat_to_m) + self.y_center
            m_y = np.log(np.tan(np.radians(lats)) + 1/np.cos(np.radians(lats)))
            v = (m_y_max - m_y) / (m_y_max - m_y_min)

            px = np.round(u * w).astype(np.int32)
            py = np.round(v * h).astype(np.int32)

            poly_p = np.column_stack((px, py)).tolist()
            polys_pixels.append(poly_p)
        return polys_pixels

    # --- Colorimétrie & Batch ---
    def open_batch_export_dialog(self):
        """Ouvre la fenêtre de mise à jour des JPEGs (Batch Processing)."""
        if not getattr(self, 'is_2d_mode', False) or getattr(self, '_global_canvas_data', None) is None:
            QMessageBox.warning(self, _("msg_warning_title"), _("msg_please_generate_the_overa"))
            return

        if not getattr(self, '_global_images_info', []):
            QMessageBox.warning(self, _("msg_error_title"), _("msg_no_information_about_jpeg"))
            return

        dialog = BatchExportDialog(self)
        dialog.exec_()

    def update_retouch_preview(self):
        """Moteur temps réel : applique les couleurs sur un calque ROI temporaire."""
        if not getattr(self, 'is_2d_mode', False) or self.selection_mask is None:
            return

        # 1. INITIALISATION DE LA ROI (Ne s'exécute qu'au 1er mouvement de slider)
        if self.retouch_roi_data is None:
            x, y, w, h = cv2.boundingRect(self.selection_mask)

            roi_img = self._global_canvas_data[y:y+h, x:x+w].copy()
            roi_mask = self.selection_mask[y:y+h, x:x+w].copy()

            max_dim = 1500
            scale_factor = 1.0
            if w > max_dim or h > max_dim:
                scale_factor = max_dim / max(w, h)
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                proxy_img = cv2.resize(roi_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                proxy_mask = cv2.resize(roi_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                proxy_img, proxy_mask = roi_img, roi_mask

            self.retouch_roi_data = {
                'x': x, 'y': y, 'w': w, 'h': h,
                'orig_img': roi_img, 'orig_mask': roi_mask,
                'proxy_img': proxy_img, 'proxy_mask': proxy_mask,
                'scale': scale_factor
            }
            self.is_retouching = True

            if self.selection_visual:
                self.selection_visual.visible = False

        # 2. RÉCUPÉRATION DES VALEURS DES SLIDERS
        vals = {k: v.value() for k, v in self.sliders_color.items()}

        # 3. TRAITEMENT COLORIMÉTRIQUE (Sur le Proxy)
        data = self.retouch_roi_data
        modified_proxy = self._process_color_math(data['proxy_img'], data['proxy_mask'], vals)

        # 4. AFFICHAGE DANS VISPY (Calque Volant)
        if getattr(self, 'retouch_preview_visual', None) is None:
            self.retouch_preview_visual = scene.visuals.Image(
                modified_proxy, parent=self.view.scene, method='auto'
            )

            # On met Z à 1.0 pour forcer le calque à flotter AU-DESSUS du fond
            s = 1.0 / data['scale']
            self.retouch_preview_visual.transform = STTransform(
                translate=(data['x'], data['y'], 1.0),
                scale=(s, s, 1.0)
            )

            # On désactive le depth_test pour éviter les conflits de pixels
            self.retouch_preview_visual.set_gl_state('translucent', depth_test=False)
            self.retouch_preview_visual.order = 10
        else:
            self.retouch_preview_visual.set_data(modified_proxy)
            self.retouch_preview_visual.visible = True

        # On force le rendu immédiat de la fenêtre
        self.canvas.update()
        self.is_color_preset_modified = True

    def _process_color_math(self, img_rgb, mask, vals):
        """Applique les mathématiques colorimétriques via NumPy et OpenCV."""
        # Conversion en float32 (Int16) pour éviter l'écrêtage pendant les calculs
        img_f = img_rgb.astype(np.float32)

        # 1. Luminosité et Contraste
        # Contraste linéaire simple : f(x) = alpha * (x - 127.5) + 127.5 + beta
        alpha = 1.0 + (vals["contrast"] / 100.0)
        beta = vals["brightness"]
        img_f = alpha * (img_f - 127.5) + 127.5 + beta

        # 2. Température (Rouge/Bleu) et Teinte (Vert/Magenta)
        # Bricolage simple : on booste certains canaux
        img_f[:, :, 0] += vals["temp"] * 0.5  # R : +Temp
        img_f[:, :, 2] -= vals["temp"] * 0.5  # B : -Temp
        img_f[:, :, 1] += vals["tint"] * 0.5  # G : +Tint

        # On sécurise la plage 0-255 avant la conversion HSV
        img_f = np.clip(img_f, 0, 255).astype(np.uint8)

        # 3. Saturation (Via HSV)
        if vals["saturation"] != 0:
            hsv = cv2.cvtColor(img_f, cv2.COLOR_RGB2HSV).astype(np.int16)
            # Saturation est le canal 1. Multiplicateur ou Additionnel.
            hsv[:, :, 1] += int(vals["saturation"] * 2.55)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img_f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 4. Seuils Noirs/Blancs (Masquage conditionnel)
        # On calcule le niveau de gris de l'image D'ORIGINE pour déterminer les zones
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Le pixel est modifié SI il est dans le masque ET dans la plage de seuil
        active_mask = (mask > 0) & (gray >= vals["thresh_b"]) & (gray <= vals["thresh_w"])

        # 5. Fusion finale (Blending)
        result = img_rgb.copy()
        result[active_mask] = img_f[active_mask]

        return result

    def apply_color_retouch(self):
        """Logique décisionnelle et sécurité des sauvegardes avant l'application HD."""
        if not self.is_retouching or self.retouch_roi_data is None:
            return

        # Évaluation des états de sauvegarde
        needs_sel_save = not self.current_sel_name or getattr(self, 'is_selection_modified', False)
        needs_preset_save = getattr(self, 'is_color_preset_modified', True)

        if needs_sel_save:
            reply = QMessageBox.question(self, _("msg_warning_title"), _("msg_warn_unsaved_sel_preset"),
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Save:
                # 1. On tente de sauvegarder la sélection
                self.save_selection_project()

                # Si l'utilisateur a cliqué sur "Cancel" dans l'input text de la sélection
                if not self.current_sel_name or getattr(self, 'is_selection_modified', False):
                    logging.info("Application aborted: selection save cancelled.")
                    return

                # 2. La sélection est sécurisée, on enchaine avec la sauvegarde des couleurs
                self.save_color_preset()

        elif needs_preset_save:
            reply = QMessageBox.question(self, _("msg_warning_title"), _("msg_warn_unsaved_preset"),
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Save:
                self.save_color_preset()

                # Si l'écriture JSON a échoué (flag resté à True)
                if getattr(self, 'is_color_preset_modified', True):
                    logging.info("Application aborted: preset save failed.")
                    return

        # Si l'utilisateur a choisi "Save" (et que ça a réussi) ou "Discard", on applique la retouche.
        self._do_apply_color_retouch()

    def _do_apply_color_retouch(self):
        """Moteur mathématique lourd isolé sous le sablier (ancien apply_color_retouch)."""
        data = self.retouch_roi_data
        vals = {k: v.value() for k, v in self.sliders_color.items()}

        logging.info("Applying hi-res color retouch...")

        # 1. On applique le moteur mathématique sur l'image HD d'origine
        hd_result = self._process_color_math(data['orig_img'], data['orig_mask'], vals)

        # 2. On injecte la ROI modifiée dans le Canvas Global
        x, y, w, h = data['x'], data['y'], data['w'], data['h']
        self._global_canvas_data[y:y+h, x:x+w] = hd_result

        # On signale que la mosaïque a été modifiée
        self._texture_needs_refresh = True

        # On "imprime" cette zone dans le masque cumulatif global
        if not hasattr(self, 'cumulative_mask') or self.cumulative_mask is None:
            canvas_h, canvas_w = self._global_canvas_data.shape[:2]
            self.cumulative_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        self.cumulative_mask[y:y+h, x:x+w] = cv2.bitwise_or(
            self.cumulative_mask[y:y+h, x:x+w],
            data['orig_mask']
        )

        self.is_retouch_exported = False

        # 3. Mise à jour de l'image de fond VisPy
        if self.image_2d_visual:
            self.image_2d_visual.set_data(self._global_canvas_data)

        # 4. Nettoyage
        self.cancel_color_retouch() # Réinitialise l'UI et détruit le Proxy
        self.clear_selection_mask()
        self.canvas.native.setFocus()

        logging.info("Color retouch applied successfully.")

    def cancel_color_retouch(self):
        """Annule la retouche en cours, détruit le calque et remet les sliders à zéro."""
        # self._color_apply_needs_pulse = False
        if hasattr(self, 'btn_apply_color'):
            self.btn_apply_color.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        self.check_stop_pulse()

        # 1. Destruction du Proxy Visuel
        if getattr(self, 'retouch_preview_visual', None) is not None:
            self.retouch_preview_visual.parent = None
            self.retouch_preview_visual = None

        # 2. Remise à zéro sécurisée via le dictionnaire des valeurs par défaut
        for key, slider in self.sliders_color.items():
            # On bloque les signaux pour ne pas recalculer 7 fois l'image pendant qu'on remet à zéro !
            slider.blockSignals(True)
            self.spinboxes_color[key].blockSignals(True)

            # On récupère le bon défaut (ex: 0 pour luminosité, 220 pour Blancs)
            def_val = self.retouch_defaults[key]

            slider.setValue(def_val)
            self.spinboxes_color[key].setValue(def_val)

            slider.blockSignals(False)
            self.spinboxes_color[key].blockSignals(False)

        # 3. Nettoyage des variables
        self.retouch_roi_data = None
        self.is_retouching = False

        # 4. Ré-affichage du masque rouge (contour ou rempli)
        if hasattr(self, 'chk_show_outline'):
            self.update_selection_visual()

        self.canvas.native.setFocus()

    def save_color_preset(self):
        """Sauvegarde les paramètres colorimétriques dans custom_colors.json avec protections strictes."""
        if not self.current_sel_name:
            QMessageBox.warning(self, _("msg_action_blocked"), _("msg_please_save_the_selection"))
            return

        if self.is_selection_modified:
            QMessageBox.warning(self, _("msg_action_blocked"), _("msg_the_selection_mask_has_be"))
            return

        tile_id = self.get_current_tile_id()
        json_file = self.get_custom_file_path("custom_colors.json")
        data = {}

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f: data = json.load(f)
            except Exception: pass

        if tile_id not in data:
            data[tile_id] = {}

        name = self.current_sel_name

        # Vérification d'écrasement
        if name in data[tile_id]:
            reply = QMessageBox.question(
                self, _("msg_confirmation"), _("msg_color_preset_exist", cn=name),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No: return

        # Capture de TOUTES les valeurs actuelles (Sliders + Seuils)
        preset = {k: v.value() for k, v in self.sliders_color.items()}
        data[tile_id][name] = preset

        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            self.is_color_preset_modified = False
            # self._color_apply_needs_pulse = True
            # self.pulse_timer.start(500)

            QMessageBox.information(self, _("msg_success_title"), _("msg_color_preset_saved", cn=name))
        except Exception as e:
            QMessageBox.critical(self, _("msg_error_title"), _("msg_json_write_error", error=e))

        self.canvas.native.setFocus()

    def load_color_preset(self):
        """Charge et prévisualise les paramètres colorimétriques associés à la sélection en cours."""
        if not self.current_sel_name:
            QMessageBox.warning(self, _("msg_info_title"), _("msg_no_selections_are_current"))
            return

        tile_id = self.get_current_tile_id()
        json_file = self.get_custom_file_path("custom_colors.json")

        if not os.path.exists(json_file):
            QMessageBox.information(self, _("msg_info_title"), _("msg_no_settings_file_custom_c"))
            return

        try:
            with open(json_file, 'r', encoding='utf-8') as f: data = json.load(f)
        except Exception:
            return

        if tile_id not in data or self.current_sel_name not in data[tile_id]:
            QMessageBox.information(self, _("msg_info_title"), _("msg_no_color_preset", cn=self.current_sel_name))
            return

        preset = data[tile_id][self.current_sel_name]

        # 1. On bloque les signaux pour éviter de recalculer la prévisualisation 7 fois de suite
        for key, val in preset.items():
            if key in self.spinboxes_color:
                self.sliders_color[key].blockSignals(True)
                self.spinboxes_color[key].blockSignals(True)

                self.sliders_color[key].setValue(val)
                self.spinboxes_color[key].setValue(val)

                self.sliders_color[key].blockSignals(False)
                self.spinboxes_color[key].blockSignals(False)

        # 2. Une fois tous les sliders en place, on force UNE SEULE mise à jour du calque de preview
        self.update_retouch_preview()
        self.is_color_preset_modified = False
        logging.info(f"Color adjustements loaded and applied for '{self.current_sel_name}'.")

        # self._color_apply_needs_pulse = True
        # self.pulse_timer.start(500)
        self.canvas.native.setFocus()

    def reset_all_retouch(self):
        """Restaure la mosaïque d'origine et efface le masque cumulatif et l'historique de réparation."""
        if getattr(self, '_original_canvas_data', None) is None:
            QMessageBox.information(self, _("msg_info_title"), _("msg_no_mosaics_are_currently_"))
            return

        # Vérification des deux types de retouches
        has_color_edits = getattr(self, 'cumulative_mask', None) is not None and np.any(self.cumulative_mask)
        has_healing_edits = getattr(self, 'has_healing_edits', False)

        # Si ni la couleur ni les pixels bruts n'ont été modifiés, on annule
        if not has_color_edits and not has_healing_edits:
            QMessageBox.information(self, _("msg_info_title"), _("msg_no_retouching_has_been_do"))
            return

        reply = QMessageBox.question(
            self, _("msg_warning"),  _("msg_are_you_sure_you_want_to_"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.cancel_color_retouch()
            self.clear_selection_mask()

            # 1. Restauration de la matrice des pixels (copie stricte)
            self._global_canvas_data = self._original_canvas_data.copy()

            # 2. Réinitialisation des états (Couleurs)
            if getattr(self, 'cumulative_mask', None) is not None:
                self.cumulative_mask.fill(0)

            # 3. Réinitialisation des états (Réparation)
            self.has_healing_edits = False
            if hasattr(self, 'heal_history'):
                self.heal_history.clear()
            if hasattr(self, 'heal_redo_stack'):
                self.heal_redo_stack.clear()

            self.update_undo_redo_buttons()

            # 4. Mise à jour instantanée de l'affichage si on est en mode 2D
            if getattr(self, 'is_2d_mode', False) and getattr(self, 'image_2d_visual', None):
                self.image_2d_visual.set_data(self._global_canvas_data)

            # 5. On prévient la 3D qu'il faudra se mettre à jour au retour
            self._texture_needs_refresh = True

            # 6. Réinitialisation de l'état batch
            self.batch_2d_state = None
            if hasattr(self, 'btn_apply_all_sel_2d'):
                self.btn_apply_all_sel_2d.setEnabled(False)

            logging.info("All changes have been undone. Origin mosaic restored.")
            QMessageBox.information(self, _("msg_success_title"), _("msg_the_alterations_have_been"))

        self.canvas.native.setFocus()

    @wait_cursor
    def show_all_selections_2d(self, checked=False, only_with_presets=False):
        """Affiche toutes les sélections 2D enregistrées avec gestion des superpositions (Cyan).
           Si only_with_presets est True, filtre pour ne garder que celles qui ont une retouche associée."""
        if getattr(self, '_global_canvas_data', None) is None:
            return

        self.clear_selection_mask()

        tile_id = self.get_current_tile_id()
        # NOUVEAU FICHIER
        sel_file = self.get_custom_file_path("custom_2D_selections.json")
        col_file = self.get_custom_file_path("custom_colors.json")

        # 1. Vérification des fichiers
        if not os.path.exists(sel_file):
            QMessageBox.information(self, _("msg_info_title"), _("msg_no_selections_are_current"))
            return

        try:
            with open(sel_file, 'r', encoding='utf-8') as f:
                selections_data = json.load(f).get(tile_id, [])
        except Exception:
            return

        # --- Filtrage optionnel des sélections ---
        if only_with_presets:
            if not os.path.exists(col_file):
                QMessageBox.information(self, _("msg_info_title"), _("msg_no_settings_file_custom_c"))
                return
            try:
                with open(col_file, 'r', encoding='utf-8') as f:
                    colors_data = json.load(f).get(tile_id, {})
            except Exception:
                return
            selections_data = [sel for sel in selections_data if sel.get("name") in colors_data]

        if not selections_data:
            QMessageBox.information(self, _("msg_info_title"), _("msg_no_selections_are_current"))
            return

        # 2. Préparation des masques
        h, w = self._global_canvas_data.shape[:2]
        total_mask = np.zeros((h, w), dtype=np.uint8)
        overlap_mask = np.zeros((h, w), dtype=bool)

        for sel in selections_data:
            poly_mask = np.zeros((h, w), dtype=np.uint8)

            # === CONVERSION EN PIXELS AVANT AFFICHAGE ===
            pos_polys_px = self._meters_to_pixels(sel.get("polygons", []))
            neg_polys_px = self._meters_to_pixels(sel.get("negative_polygons", []))

            for poly in pos_polys_px:
                cv2.fillPoly(poly_mask, [np.array(poly, dtype=np.int32)], 255)
            for poly in neg_polys_px:
                cv2.fillPoly(poly_mask, [np.array(poly, dtype=np.int32)], 0)

            # Détection des superpositions
            overlap = (total_mask > 0) & (poly_mask > 0)
            overlap_mask |= overlap
            total_mask = cv2.bitwise_or(total_mask, poly_mask)

        # 3. Création du visuel (Rouge normal, Cyan pour superpositions)
        rgba_mask = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_mask[total_mask > 0] = [255, 0, 0, 100]  # Rouge
        rgba_mask[overlap_mask] = [0, 0, 255, 100]  # Bleu

        if not getattr(self, 'selection_visual', None):
            self.selection_visual = scene.visuals.Image(rgba_mask, parent=self.view.scene, interpolation='nearest')
            self.selection_visual.set_gl_state('translucent', depth_test=False, blend=True)
            self.selection_visual.order = 2
        else:
            self.selection_visual.set_data(rgba_mask)
            self.selection_visual.visible = True

        # S'assurer que le rendu couleur est masqué s'il était actif
        if getattr(self, 'retouch_preview_visual', None):
            self.retouch_preview_visual.visible = False

        # 4. Verrouillage UI
        self.group_selection.setEnabled(False)
        self.group_color.setEnabled(False)
        self.btn_apply_all_sel_2d.setEnabled(True)
        self.batch_2d_state = "show_masks"

        # Nettoyage masque individuel
        self.selection_mask = None
        if hasattr(self, 'current_sel_name_input'):
            self.current_sel_name_input.clear()

        self.canvas.native.setFocus()
        logging.info(f"Displayed {len(selections_data)} selections for batch preview.")

    @wait_cursor
    def apply_all_color_presets(self, checked=False):
        """Applique séquentiellement les retouches sur l'image HD et met à jour le cumulatif."""
        if self.batch_2d_state != "show_masks":
            return

        tile_id = self.get_current_tile_id()
        # NOUVEAU FICHIER
        sel_file = self.get_custom_file_path("custom_2D_selections.json")
        col_file = self.get_custom_file_path("custom_colors.json")

        try:
            with open(sel_file, 'r', encoding='utf-8') as f:
                selections_data = json.load(f).get(tile_id, [])
            with open(col_file, 'r', encoding='utf-8') as f:
                colors_data = json.load(f).get(tile_id, {})
        except Exception:
            return

        logging.info("Applying all presets in HD...")
        h, w = self._global_canvas_data.shape[:2]
        hd_img = self._global_canvas_data.copy()

        if not hasattr(self, 'cumulative_mask') or self.cumulative_mask is None:
            self.cumulative_mask = np.zeros((h, w), dtype=np.uint8)

        for sel in selections_data:
            name = sel.get("name")
            if name in colors_data:
                vals = colors_data[name]

                # Masque HD
                hd_mask = np.zeros((h, w), dtype=np.uint8)

                # === CONVERSION EN PIXELS AVANT CALCUL SUR LE MASQUE ===
                pos_polys_px = self._meters_to_pixels(sel.get("polygons", []))
                neg_polys_px = self._meters_to_pixels(sel.get("negative_polygons", []))

                for poly in pos_polys_px:
                    cv2.fillPoly(hd_mask, [np.array(poly, dtype=np.int32)], 255)
                for poly in neg_polys_px:
                    cv2.fillPoly(hd_mask, [np.array(poly, dtype=np.int32)], 0)

                if np.any(hd_mask > 0):
                    hd_img = self._process_color_math(hd_img, hd_mask, vals)
                    self.cumulative_mask = cv2.bitwise_or(self.cumulative_mask, hd_mask)

        # 1. Sauvegarde dans le Global Canvas
        self._global_canvas_data = hd_img
        self._texture_needs_refresh = True
        self.is_retouch_exported = False

        # 2. Mise à jour VisPy
        if self.image_2d_visual:
            self.image_2d_visual.set_data(self._global_canvas_data)

        # 3. Nettoyage de l'UI et des états
        if getattr(self, 'retouch_preview_visual', None):
            self.retouch_preview_visual.parent = None
            self.retouch_preview_visual = None

        if getattr(self, 'selection_visual', None):
            self.selection_visual.visible = False

        self.group_selection.setEnabled(True)
        self.group_color.setEnabled(True)
        self.btn_apply_all_sel_2d.setEnabled(False)
        self.batch_2d_state = None

        self.canvas.native.setFocus()
        logging.info("All color presets successfully applied in HD.")

    def toggle_meansh_mode(self):
        """Active le pinceau de mélange des couleurs (Mean Shift)."""
        if self.is_seamless_mode == True:
            return

        if self.btn_toggle_meansh.isChecked():
            self.is_blur_mode = True
            self.btn_toggle_meansh.setStyleSheet("background-color: #d35400; color: white; font-weight: bold;")
            self.btn_toggle_meansh.setText(_("btn_toggle_meansh_active"))
        else:
            self.is_blur_mode = False
            self.btn_toggle_meansh.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
            self.btn_toggle_meansh.setText(_("btn_toggle_meansh"))
            if getattr(self, 'brush_cursor_visual', None) is not None:
                self.brush_cursor_visual.visible = False
                self.canvas.update()
        self.update_undo_redo_buttons()

    def on_brush_radius_changed(self, value):
        self.brush_radius = value
        if getattr(self, 'brush_cursor_visual', None) is not None:
            # Vispy attend souvent un tuple (rx, ry) pour le rayon
            self.brush_cursor_visual.radius = (self.brush_radius, self.brush_radius)

    def update_brush_visual(self):
        """Affiche le trait jaune temporaire pendant le tracé du pinceau."""
        if not self.brush_points:
            if getattr(self, 'brush_visual', None):
                self.brush_visual.visible = False
            return

        pts = np.array(self.brush_points)
        if self.brush_visual is None:
            self.brush_visual = scene.visuals.Line(pos=pts, color='yellow', width=self.brush_radius, parent=self.view.scene)
            self.brush_visual.set_gl_state('translucent', depth_test=False)
            self.brush_visual.order = 15
        else:
            self.brush_visual.set_data(pos=pts, width=self.brush_radius)
            self.brush_visual.visible = True

    def apply_mean_shift(self):
        """Applique un Mean Shift sur la zone tracée par l'utilisateur."""
        if not self.brush_points:
            return

        h, w = self._global_canvas_data.shape[:2]

        # 1. Calcul strict de la Bounding Box du tracé
        pts = np.array(self.brush_points, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)

        # 2. Padding de sécurité (Taille du pinceau + Marge pour l'algo)
        pad = self.brush_radius + 15
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        roi_w = x2 - x1
        roi_h = y2 - y1
        local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

        # 3. Transposition des coordonnées globales -> locales
        local_pts = pts - np.array([x1, y1])

        # 4. Dessin du masque selon les mouvements de la souris
        if len(local_pts) == 1:
            # Simple clic
            cv2.circle(local_mask, tuple(local_pts[0]), self.brush_radius, 255, -1)
        else:
            # Traînée
            for i in range(len(local_pts) - 1):
                cv2.line(local_mask, tuple(local_pts[i]), tuple(local_pts[i+1]), 255, thickness=self.brush_radius*2)

        # Lissage des bords du masque pour un inpainting plus organique
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        local_mask = cv2.dilate(local_mask, kernel)

        # 5. Extraction, Inpainting et Réinjection
        roi_img = self._global_canvas_data[y1:y2, x1:x2].copy()

        # Gestion UNDO
        self.heal_history.append((x1, y1, x2, y2, roi_img.copy()))

        # Limiter l'historique
        if len(self.heal_history) > self.MAX_HISTORY:
            self.heal_history.pop(0)

        # Gestion REDO
        self.heal_redo_stack.clear()

        # Choix de l'algorithme selon le bouton actif
        # Utilisation du Mean Shift pour créer des "aplats" de couleurs uniformes (idéal champs/lacs)

        # sp (Spatial Radius) : distance de l'effet (ex: 20 pixels)
        # sr (Color Radius) : tolérance de couleur (ex: 40)
        # Plus ces valeurs sont hautes, plus l'image devient "peinture à l'huile"
        filtered_roi = cv2.pyrMeanShiftFiltering(roi_img, sp=20, sr=40)

        # Création du masque de fusion adouci
        alpha = cv2.GaussianBlur(local_mask, (21, 21), 0).astype(np.float32) / 255.0
        alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

        # Application
        blended_roi = roi_img.astype(np.float32) * (1.0 - alpha_3d) + filtered_roi.astype(np.float32) * alpha_3d
        healed_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)

        self._global_canvas_data[y1:y2, x1:x2] = healed_roi
        self._texture_needs_refresh = True
        self.is_retouch_exported = False
        self.has_healing_edits = True

        # 6. Mise à jour de l'affichage
        if self.image_2d_visual:
            self.image_2d_visual.set_data(self._global_canvas_data)

        # 7. Nettoyage
        self.brush_points = []
        if self.brush_visual:
            self.brush_visual.visible = False

        self.canvas.update()
        self.update_undo_redo_buttons()
        logging.info("Healing brush stroke applied.")

    def undo_last_heal(self):
        """Annule la dernière réparation de pixels (Mean Shift ou Seamless Clone)."""
        if not getattr(self, 'heal_history', []):
            return

        # 1. On récupère la dernière sauvegarde (état avant le coup de pinceau)
        x1, y1, x2, y2, original_roi = self.heal_history.pop()

        # 2. On sauvegarde l'état actuel (le résultat du pinceau) dans le Redo
        current_roi = self._global_canvas_data[y1:y2, x1:x2].copy()
        self.heal_redo_stack.append((x1, y1, x2, y2, current_roi))

        # 3. On restaure le morceau dans l'image globale
        self._global_canvas_data[y1:y2, x1:x2] = original_roi
        self._texture_needs_refresh = True
        self.is_retouch_exported = False

        # 4. On met à jour l'affichage VisPy
        if self.image_2d_visual:
            self.image_2d_visual.set_data(self._global_canvas_data)

        self.canvas.update()
        self.update_undo_redo_buttons()
        logging.info("Healing brush stroke undone.")

    def redo_last_heal(self):
        """Rétablit la dernière réparation de pixels annulée."""
        if not getattr(self, 'heal_redo_stack', []):
            return

        # 1. On récupère l'état annulé
        x1, y1, x2, y2, healed_roi = self.heal_redo_stack.pop()

        # 2. On repousse l'état actuel dans l'Undo pour pouvoir annuler à nouveau
        current_roi = self._global_canvas_data[y1:y2, x1:x2].copy()
        self.heal_history.append((x1, y1, x2, y2, current_roi))

        # 3. On applique le Redo
        self._global_canvas_data[y1:y2, x1:x2] = healed_roi
        self._texture_needs_refresh = True
        self.is_retouch_exported = False

        # 4. On met à jour l'affichage
        if self.image_2d_visual:
            self.image_2d_visual.set_data(self._global_canvas_data)

        self.canvas.update()
        self.update_undo_redo_buttons()
        logging.info("Healing brush stroke redone.")

    def toggle_seamless_mode(self):
        # Bloque l'activation si le mean shift est déjà en cours
        if getattr(self, 'is_blur_mode', False):
            self.btn_toggle_seamless.setChecked(False) # On décoche par sécurité
            return

        if self.btn_toggle_seamless.isChecked():
            # Vérification de la sélection
            if getattr(self, 'selection_mask', None) is None or not np.any(self.selection_mask):
                # On force le bouton à se décocher puisqu'on refuse l'action
                self.btn_toggle_seamless.setChecked(False)
                QMessageBox.warning(self, _("msg_warning_title"), _("msg_plz_create_or_load_selection"))
                return

            self.is_seamless_mode = True
            self.btn_toggle_seamless.setStyleSheet("background-color: #d35400; color: white; font-weight: bold;")
            self.btn_toggle_seamless.setText(_("btn_toggle_seamless_active"))
        else:
            self.is_seamless_mode = False
            self.btn_toggle_seamless.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
            self.btn_toggle_seamless.setText(_("btn_toggle_seamless"))
            if getattr(self, 'seamless_ghost_visual', None) is not None:
                self.seamless_ghost_visual.visible = False
                self.canvas.update()

        self.update_undo_redo_buttons()

    def apply_seamless_clone(self, end_pos):
        """Applique l'algorithme Seamless Clone (OpenCV) entre la position initiale et finale."""
        delta = end_pos - self.seamless_start_pos

        # 1. Dimensions avec Padding (Marge) pour éviter le bug d'OpenCV
        pad = 10
        x, y, w, h = cv2.boundingRect(self.selection_mask)

        # Sécurité basique pour ne pas sortir de l'image (destination) avec le padding
        x_pad = max(0, x - pad)
        y_pad = max(0, y - pad)
        w_pad = min(self._global_canvas_data.shape[1] - x_pad, w + 2*pad)
        h_pad = min(self._global_canvas_data.shape[0] - y_pad, h + 2*pad)

        # Le centre demandé par OpenCV est le centre du rectangle dans l'image globale
        center = (x_pad + w_pad // 2, y_pad + h_pad // 2)

        # 2. Source : extraction avec le même décalage
        x_src, y_src = x_pad + delta[0], y_pad + delta[1]

        # Vérification anti-crash (si la source sort de l'image)
        if x_src < 0 or y_src < 0 or x_src + w_pad > self._global_canvas_data.shape[1] or y_src + h_pad > self._global_canvas_data.shape[0]:
            logging.warning("Error : The source area extends beyond the image.")
            return

        src_patch = self._global_canvas_data[y_src:y_src+h_pad, x_src:x_src+w_pad]

        # 3. Masque : Nettoyage strict pour OpenCV
        mask_patch_raw = self.selection_mask[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        # OpenCV exige un masque en 8-bits (0 ou 255), sur un seul canal
        mask_patch = np.where(mask_patch_raw > 0, 255, 0).astype(np.uint8)
        if len(mask_patch.shape) == 3:
            mask_patch = mask_patch[:, :, 0]

        # undo-redo
        if hasattr(self, 'heal_history'):
            roi_backup = self._global_canvas_data[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad].copy()
            self.heal_history.append((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad, roi_backup))
            if hasattr(self, 'heal_redo_stack'):
                self.heal_redo_stack.clear()

        try:
            # 4. L'algorithme
            cloned = cv2.seamlessClone(
                src_patch,
                self._global_canvas_data,
                mask_patch,
                center,
                cv2.NORMAL_CLONE
            )

            # 5. Injection du résultat
            self._global_canvas_data = cloned
            self._texture_needs_refresh = True
            self.is_retouch_exported = False
            self.has_healing_edits = True

            if self.image_2d_visual:
                self.image_2d_visual.set_data(self._global_canvas_data)

            self.update_undo_redo_buttons()
            logging.info("Seamless Clone succeeded")

            # Sortie auto et contour cyan
            self.btn_toggle_seamless.setChecked(False)
            self.toggle_seamless_mode()
            if hasattr(self, 'chk_show_outline') and not self.chk_show_outline.isChecked():
                self.chk_show_outline.setChecked(True)

        except Exception as e:
            logging.error(f"Seamless Clone Error : {e}")
            # En cas de crash, on annule l'entrée d'historique qu'on vient de préparer
            if hasattr(self, 'heal_history') and self.heal_history:
                self.heal_history.pop()

    def save_global_tex_2d_to_imgfile(self):
        """Sauvegarde la texture 2D restreinte au rectangle (Bounding Box) du mesh."""
        if getattr(self, '_global_canvas_data', None) is None:
            return

        tile_id = self.get_current_tile_id()

        # 1. Demander le nom/suffixe via UI
        suffix, ok = QInputDialog.getText(self, _("msg_export_texture"), _("msg_tile_texture", ti=tile_id))
        if not ok or not suffix.strip():
            return

        # Nettoyage des caractères non autorisés
        safe_suffix = re.sub(r'[^a-zA-Z0-9_-]', '_', suffix.strip())

        # CORRECTION : On retire le tile_id_ du suffixe s'il y a été inséré
        if safe_suffix.startswith(f"{tile_id}_"):
            safe_suffix = safe_suffix[len(f"{tile_id}_"):]
        elif safe_suffix.startswith(tile_id):
            safe_suffix = safe_suffix[len(tile_id):].lstrip('_')

        filename = f"{safe_suffix}.png"
        filepath = self.get_texture_file_path(filename)

        # 2. Vérification de l'écrasement
        if os.path.exists(filepath):
            reply = QMessageBox.question(self, _("msg_confirmation"), _("msg_texture_exists", tn=safe_suffix),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No: return

        # 3. Calcul de la Bounding Box du mesh (Pixels)
        h, w = self._global_canvas_data.shape[:2]
        global_uvs = self._calculate_mercator_uvs(*self._global_bounds)
        pts_2d = np.column_stack((global_uvs[:, 0] * w, global_uvs[:, 1] * h)).astype(np.int32)

        x_min_px, y_min_px = np.min(pts_2d, axis=0)
        x_max_px, y_max_px = np.max(pts_2d, axis=0)

        # Sécurité bords de l'image
        x_min_px, y_min_px = max(0, x_min_px), max(0, y_min_px)
        x_max_px, y_max_px = min(w, x_max_px), min(h, y_max_px)

        # 4. Rognage géométrique (Crop strict sans transparence)
        cropped_img = self._global_canvas_data[y_min_px:y_max_px, x_min_px:x_max_px].copy()

        # Conversion RGB(A) (VisPy) vers BGR (OpenCV)
        if cropped_img.shape[2] == 4:
            export_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGBA2BGR)
        else:
            export_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)

        # 5. Sauvegarde physique
        try:
            cv2.imwrite(filepath, export_img)
            QMessageBox.information(self, _("msg_success_title"), _("msg_texture_saved", tn=safe_suffix))
        except Exception as e:
            QMessageBox.critical(self, _("msg_error_title"), _("msg_write_error", error=str(e)))

    def load_global_tex_2d_from_imgfile(self):
        """Charge un PNG et l'applique en surcouche (Alpha Blending) dans la zone du mesh."""
        if getattr(self, '_global_canvas_data', None) is None:
            return

        tile_id = self.get_current_tile_id()

        # Déduction du dossier 'custom'
        dummy_path = self.get_texture_file_path("dummy.txt")
        custom_dir = os.path.dirname(dummy_path)

        # 1. Lister et mapper les fichiers pour un affichage propre
        file_mapping = {}
        prefix = f"{tile_id}_"

        if os.path.exists(custom_dir):
            for f in os.listdir(custom_dir):
                if f.startswith(prefix) and f.lower().endswith(".png"):
                    # On crée le nom propre : on enlève le préfixe et on enlève l'extension '.png' (-4 caractères)
                    clean_name = f[len(prefix):-4]
                    file_mapping[clean_name] = f

        if not file_mapping:
            QMessageBox.information(self, _("msg_info_title"), _("msg_no_texture_found", ti=tile_id))
            return

        # On extrait la liste des "noms propres" pour le menu déroulant
        display_names = list(file_mapping.keys())
        display_names.sort()

        # 2. Fenêtre de sélection UI
        chosen_name, ok = QInputDialog.getItem(self, _("msg_load_texture"), _("msg_select_texture"), display_names, 0, False)
        if not ok or not chosen_name: return

        # On récupère le vrai nom du fichier grâce au mapping
        filename = file_mapping[chosen_name]
        filepath = os.path.join(custom_dir, filename)

        # 3. Chargement OpenCV (IMREAD_UNCHANGED pour forcer la lecture de l'Alpha éventuel)
        overlay_bgra = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if overlay_bgra is None:
            QMessageBox.critical(self, _("msg_error_title"), _("msg_texture_read_error"))
            return

        # Conversion BGR(A) vers RGB(A)
        if len(overlay_bgra.shape) == 3 and overlay_bgra.shape[2] == 4:
            overlay = cv2.cvtColor(overlay_bgra, cv2.COLOR_BGRA2RGBA)
        else:
            overlay = cv2.cvtColor(overlay_bgra, cv2.COLOR_BGR2RGB)

        # 4. Calcul de la fenêtre cible (Bounding Box)
        h, w = self._global_canvas_data.shape[:2]
        global_uvs = self._calculate_mercator_uvs(*self._global_bounds)
        pts_2d = np.column_stack((global_uvs[:, 0] * w, global_uvs[:, 1] * h)).astype(np.int32)

        x_min_px, y_min_px = np.min(pts_2d, axis=0)
        x_max_px, y_max_px = np.max(pts_2d, axis=0)
        x_min_px, y_min_px = max(0, x_min_px), max(0, y_min_px)
        x_max_px, y_max_px = min(w, x_max_px), min(h, y_max_px)

        target_w = x_max_px - x_min_px
        target_h = y_max_px - y_min_px
        if target_w <= 0 or target_h <= 0: return

        # 5. Redimensionnement adaptatif
        overlay_h, overlay_w = overlay.shape[:2]
        if overlay_w != target_w or overlay_h != target_h:
            overlay = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # 6. Fusion Mathématique
        target_roi = self._global_canvas_data[y_min_px:y_max_px, x_min_px:x_max_px]

        if overlay.shape[2] == 4:
            alpha_mask = overlay[:, :, 3] / 255.0
            inv_alpha = 1.0 - alpha_mask
            for c in range(3):
                target_roi[:, :, c] = (alpha_mask * overlay[:, :, c] + inv_alpha * target_roi[:, :, c]).astype(np.uint8)
        else:
            target_roi[:, :, :3] = overlay[:, :, :3]

        self._global_canvas_data[y_min_px:y_max_px, x_min_px:x_max_px] = target_roi

        # 7. Mise à jour de l'affichage
        if getattr(self, 'image_2d_visual', None):
            self.image_2d_visual.set_data(self._global_canvas_data)

        self._texture_needs_refresh = True
        self.canvas.native.setFocus()

    # =========================================================================
    #
    # Utilitaires de calcul bas niveau (Maths & Filtres)
    #
    # =========================================================================

    def _get_world_xy_from_canvas(self, canvas_x, canvas_y, target_z):
        """Simule un rayon depuis la caméra jusqu'au sol pour convertir l'écran 2D en monde 3D."""
        transform = self.mesh_visual.get_transform(map_from='canvas', map_to='visual')

        # On projette un point très près de la caméra, et un point très loin
        p_near = transform.map([canvas_x, canvas_y, -1, 1])
        p_far  = transform.map([canvas_x, canvas_y, 1, 1])

        # Normalisation
        p_near = p_near[:3] / p_near[3]
        p_far  = p_far[:3] / p_far[3]

        ray_dir = p_far - p_near
        if abs(ray_dir[2]) < 1e-6: return p_near[0], p_near[1] # Sécurité

        # Calcul de l'intersection mathématique avec l'altitude (Z) du sol
        t = (target_z - p_near[2]) / ray_dir[2]
        return p_near[0] + t * ray_dir[0], p_near[1] + t * ray_dir[1]

    def _calculate_mercator_uvs(self, x_min_m, x_max_m, y_min_m, y_max_m):
        """Calcule les UVs de la texture globale en compensant la distorsion Mercator."""
        # 1. On déduit les latitudes réelles (en degrés) à partir des mètres locaux
        lats = (self.original_vertices[:, 1] / self.lat_to_m) + self.y_center
        lat_max = (y_max_m / self.lat_to_m) + self.y_center
        lat_min = (y_min_m / self.lat_to_m) + self.y_center

        # 2. Fonction mathématique de projection Mercator
        def mercator_y(lat_deg):
            return np.log(np.tan(np.radians(lat_deg)) + 1/np.cos(np.radians(lat_deg)))

        # 3. Projection des latitudes sur l'axe Y de l'image
        m_y = mercator_y(lats)
        m_y_max = mercator_y(lat_max) # Bord Nord
        m_y_min = mercator_y(lat_min) # Bord Sud

        # 4. Calcul final des UVs
        u = (self.original_vertices[:, 0] - x_min_m) / (x_max_m - x_min_m)
        v = (m_y_max - m_y) / (m_y_max - m_y_min)

        return np.column_stack((u, v))

    def get_z_at_xy(self, x, y):
        """
        Trouve le triangle contenant (x,y) et calcule l'altitude Z exacte
        sur son plan incliné par coordonnées barycentriques.
        """
        # 1. Filtrage local (Boîte englobante) pour ne tester que les triangles proches
        dist_sq = (self.original_vertices[:, 0] - x)**2 + (self.original_vertices[:, 1] - y)**2
        num_candidates = min(50, len(dist_sq))
        closest_verts = np.argpartition(dist_sq, num_candidates - 1)[:num_candidates]

        is_closest = np.zeros(len(self.original_vertices), dtype=bool)
        is_closest[closest_verts] = True

        mask = is_closest[self.original_faces[:, 0]] | \
               is_closest[self.original_faces[:, 1]] | \
               is_closest[self.original_faces[:, 2]]

        candidate_faces = self.original_faces[mask]

        if len(candidate_faces) > 0:
            v0 = self.original_vertices[candidate_faces[:, 0]]
            v1 = self.original_vertices[candidate_faces[:, 1]]
            v2 = self.original_vertices[candidate_faces[:, 2]]

            # 2. Calcul du dénominateur barycentrique
            denom = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
            valid = np.abs(denom) > 1e-8 # Évite la division par zéro (triangles dégénérés)

            if np.any(valid):
                v0_v = v0[valid]
                v1_v = v1[valid]
                v2_v = v2[valid]
                denom_v = denom[valid]

                # 3. Poids barycentriques w1, w2 (w3 est déduit)
                w1 = ((v1_v[:, 1] - v2_v[:, 1]) * (x - v2_v[:, 0]) + (v2_v[:, 0] - v1_v[:, 0]) * (y - v2_v[:, 1])) / denom_v
                w2 = ((v2_v[:, 1] - v0_v[:, 1]) * (x - v2_v[:, 0]) + (v0_v[:, 0] - v2_v[:, 0]) * (y - v2_v[:, 1])) / denom_v
                w3 = 1.0 - w1 - w2

                # 4. Le point (X,Y) est DANS le triangle si les 3 poids sont entre 0 et 1
                inside = (w1 >= -1e-4) & (w2 >= -1e-4) & (w3 >= -1e-4)
                idx = np.where(inside)[0]

                if len(idx) > 0:
                    first = idx[0]
                    # On interpole le Z en fonction du poids des 3 sommets du triangle
                    z_exact = w1[first] * v0_v[first, 2] + w2[first] * v1_v[first, 2] + w3[first] * v2_v[first, 2]
                    return z_exact

        # 5. Fallback de sécurité si on clique dans le vide ou en bordure de tuile :
        # Interpolation (IDW) sur les 3 sommets les plus proches
        idx_nearest = np.argpartition(dist_sq, min(2, len(dist_sq)-1))[:3]
        dists = np.sqrt(dist_sq[idx_nearest])
        dists[dists < 1e-6] = 1e-6
        weights = 1.0 / (dists ** 2)
        return np.sum(weights * self.original_vertices[idx_nearest, 2]) / np.sum(weights)

    def is_xy_strictly_inside_mesh(self, x, y):
        """
        Vérifie de manière stricte si un point 2D est à l'intérieur des limites du maillage.
        Contrairement à get_z_at_xy, pas d'approximation IDW : si on est dehors, c'est False.
        """
        # 1. Vérification ultra-rapide par la Bounding Box globale
        # (Suppose que tu stockes les min/max du mesh lors du chargement)
        if not (self._global_min_x <= x <= self._global_max_x and
                self._global_min_y <= y <= self._global_max_y):
            return False

        # 2. Recherche locale (comme dans get_z_at_xy)
        dist_sq = (self.original_vertices[:, 0] - x)**2 + (self.original_vertices[:, 1] - y)**2
        num_candidates = min(50, len(dist_sq))
        closest_verts = np.argpartition(dist_sq, num_candidates - 1)[:num_candidates]

        is_closest = np.zeros(len(self.original_vertices), dtype=bool)
        is_closest[closest_verts] = True
        mask = is_closest[self.original_faces[:, 0]] | \
            is_closest[self.original_faces[:, 1]] | \
            is_closest[self.original_faces[:, 2]]

        candidate_faces = self.original_faces[mask]

        if len(candidate_faces) == 0:
            return False

        # 3. Test des coordonnées barycentriques
        v0 = self.original_vertices[candidate_faces[:, 0]]
        v1 = self.original_vertices[candidate_faces[:, 1]]
        v2 = self.original_vertices[candidate_faces[:, 2]]

        denom = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
        valid = np.abs(denom) > 1e-8

        if not np.any(valid): return False

        v0_v, v1_v, v2_v, denom_v = v0[valid], v1[valid], v2[valid], denom[valid]

        w1 = ((v1_v[:, 1] - v2_v[:, 1]) * (x - v2_v[:, 0]) + (v2_v[:, 0] - v1_v[:, 0]) * (y - v2_v[:, 1])) / denom_v
        w2 = ((v2_v[:, 1] - v0_v[:, 1]) * (x - v2_v[:, 0]) + (v0_v[:, 0] - v2_v[:, 0]) * (y - v2_v[:, 1])) / denom_v
        w3 = 1.0 - w1 - w2

        # Tolérance infinitésimale pour les points exactement sur une arête
        inside = (w1 >= -1e-4) & (w2 >= -1e-4) & (w3 >= -1e-4)

        return np.any(inside)

    def get_current_tile_id(self):
        """Déduit mathématiquement l'identifiant de la tuile (ex: +45+006) à partir du centre du mesh."""
        if getattr(self, 'x_center', None) is None or getattr(self, 'y_center', None) is None:
            return "unknown_tile"

        lat_floor = int(math.floor(self.y_center))
        lon_floor = int(math.floor(self.x_center))

        # Formatage Ortho4XP : +45 ou -02 pour Latitude, +006 ou -012 pour Longitude
        tile_id = f"{lat_floor:+03d}{lon_floor:+04d}"
        return tile_id

    def get_protected_vertices_mask(self, bitmask_filter=None):
        """
        Génère un masque booléen des SOMMETS protégés.
        - Si bitmask_filter est None : Protège tout ce qui n'est pas du terrain pur (type > 0).
        - Si bitmask_filter = 3 : Protège uniquement l'eau (types 1, 2 et leurs combinaisons).
        """
        if bitmask_filter is None:
            protected_faces_mask = self.original_tri_types > 0
        else:
            protected_faces_mask = (self.original_tri_types & bitmask_filter) != 0

        protected_faces = self.original_faces[protected_faces_mask]

        is_protected_vert = np.zeros(len(self.original_vertices), dtype=bool)
        if len(protected_faces) > 0:
            is_protected_vert[np.unique(protected_faces)] = True

        return is_protected_vert

    def _apply_earthwork_blend_unified(self, global_indices, target_z, dist_to_edge, trans_width):
        """
        Noyau unifié pour déformer le terrain sur une distance `trans_width`.
        - global_indices: Indices originaux des sommets concernés.
        - target_z: Altitude cible (scalaire pour flatten, ou array Numpy pour piste).
        - dist_to_edge: Array des distances euclidiennes strictes de chaque sommet au bord (0 = intérieur).
        - trans_width: Largeur globale du talus en mètres.
        """
        # Masque d'application de base
        mask_apply = (dist_to_edge <= trans_width)

        # --- Z-FREEZE INTELLIGENT (Talus Uniquement) ---
        protected_verts_mask = self.get_protected_vertices_mask(bitmask_filter=19)
        # Un sommet appartient exclusivement au talus si sa distance au bord est > 0
        is_talus_vert = (dist_to_edge > 0.0) & mask_apply

        # On désélectionne les sommets s'ils sont à la fois dans le talus ET protégés
        mask_apply &= ~(is_talus_vert & protected_verts_mask[global_indices])
        # -----------------------------------------------

        idx_to_move = np.where(mask_apply)[0]
        final_global_indices = global_indices[idx_to_move]

        # Gestion dynamique de target_z (scalaire vs array d'altitudes)
        if isinstance(target_z, np.ndarray):
            final_target_z = target_z[idx_to_move]
        else:
            final_target_z = target_z

        current_z = self.original_vertices[final_global_indices, 2]

        # Application de la courbure trigonométrique
        self.original_vertices[final_global_indices, 2] = calculate_earthwork_blend(
            current_z, final_target_z, dist_to_edge[idx_to_move], trans_width
        )

    # =========================================================================
    #
    # Utilitaires variés
    #
    # =========================================================================

    def get_custom_file_path(self, filename):
        """Retourne le chemin vers un fichier custom et crée le sous-dossier si besoin."""
        custom_dir = "custom"

        # Crée le dossier s'il n'existe pas encore
        os.makedirs(custom_dir, exist_ok=True)

        # On récupère l'ID de la tuile actuelle (ex: +45+006)
        tile_id = self.get_current_tile_id()

        # On préfixe le nom du fichier (ex: +45+006_custom_2D_selections.json)
        tile_specific_filename = f"{tile_id}_{filename}"

        return os.path.join(custom_dir, tile_specific_filename)

    def get_texture_file_path(self, filename):
        """Retourne le chemin vers un fichier de texture et crée le sous-dossier si besoin."""
        custom_dir = "texture"

        # Crée le dossier s'il n'existe pas encore
        os.makedirs(custom_dir, exist_ok=True)

        # On récupère l'ID de la tuile actuelle (ex: +45+006)
        tile_id = self.get_current_tile_id()

        # On préfixe le nom du fichier (ex: +45+006_custom_2D_selections.json)
        tile_specific_filename = f"{tile_id}_{filename}"

        return os.path.join(custom_dir, tile_specific_filename)

    def animate_pulse(self):
        """Alterne la couleur des boutons toggle actifs pour créer un effet de pulsation."""
        self.pulse_state = not self.pulse_state
        # Alternance entre un Orange vif et un Orange un peu plus sombre
        color = "#e67e22" if self.pulse_state else "#d35400"
        style = f"background-color: {color}; color: white; font-weight: bold; padding: 6px; border: 2px solid #a84300; border-radius: 4px;"

        # On applique le style uniquement aux outils actuellement activés
        if getattr(self, 'zone_draw_mode', False) and hasattr(self, 'btn_toggle_zone'):
            self.btn_toggle_zone.setStyleSheet(style)

        if getattr(self, 'flatten_active', False) and hasattr(self, 'btn_apply_flatten'):
            self.btn_apply_flatten.setStyleSheet(style)

        if getattr(self, 'runway_active', False) and hasattr(self, 'btn_toggle_runway'):
            self.btn_toggle_runway.setStyleSheet(style)

        if getattr(self, 'cylinder_active', False) and hasattr(self, 'btn_toggle_cylinder'):
            self.btn_toggle_cylinder.setStyleSheet(style)

        if getattr(self, '_color_apply_needs_pulse', False) and hasattr(self, 'btn_apply_color'):
            self.btn_apply_color.setStyleSheet(style)

    def check_stop_pulse(self):
        """Arrête le timer si plus aucun outil toggle n'est actif."""
                # getattr(self, 'cylinder_active', False) or
        if not (getattr(self, 'zone_draw_mode', False) or
                getattr(self, 'flatten_active', False) or
                getattr(self, 'runway_active', False) or
                getattr(self, '_color_apply_needs_pulse', False)):
            self.pulse_timer.stop()

# =======================================
#
# Sortie de la classe OrthoMeshViewer ici
#
# =======================================

class NumpyPchipInterpolator:
    """
    Interpolation Cubique Monotone (PCHIP) 100% NumPy.
    Garantit une courbe lisse sans 'overshoot' (pas de bosses entre les points).
    """
    def __init__(self, x, y):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        n = len(self.x)

        if n < 2:
            raise ValueError(_("msg_error_interpolation"))
        elif n == 2:
            self.slopes = np.array([(self.y[1] - self.y[0]) / (self.x[1] - self.x[0])] * 2)
            return

        h = np.diff(self.x)
        delta = np.diff(self.y) / h
        d = np.zeros(n)

        # Calcul des tangentes intérieures (moyenne harmonique pondérée)
        for i in range(1, n - 1):
            if delta[i-1] * delta[i] > 0:
                w1 = 2 * h[i] + h[i-1]
                w2 = h[i] + 2 * h[i-1]
                d[i] = (w1 + w2) / (w1 / delta[i-1] + w2 / delta[i])
            else:
                d[i] = 0.0 # Force la pente à 0 aux extremums locaux pour éviter l'overshoot

        # Tangentes aux extrémités
        d[0] = delta[0]
        d[-1] = delta[-1]
        self.slopes = d

    def __call__(self, xi):
        xi = np.atleast_1d(xi)
        yi = np.zeros_like(xi, dtype=float)

        for j, x_val in enumerate(xi):
            # Extrapolation linéaire si en dehors des limites
            if x_val <= self.x[0]:
                yi[j] = self.y[0] + self.slopes[0] * (x_val - self.x[0])
                continue
            if x_val >= self.x[-1]:
                yi[j] = self.y[-1] + self.slopes[-1] * (x_val - self.x[-1])
                continue

            idx = np.searchsorted(self.x, x_val) - 1
            idx = max(0, min(idx, len(self.x) - 2))

            h = self.x[idx+1] - self.x[idx]
            t = (x_val - self.x[idx]) / h

            # Polynômes d'Hermite cubiques
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2

            yi[j] = (h00 * self.y[idx] + h10 * h * self.slopes[idx] +
                     h01 * self.y[idx+1] + h11 * h * self.slopes[idx+1])

        return yi[0] if yi.size == 1 else yi

# =======================================
#
# Sortie de la classe NumpyPchipInterpolator ici
#
# =======================================

class BatchExportDialog(QDialog):
    def __init__(self, parent_viewer):
        super().__init__(parent_viewer)
        self.viewer = parent_viewer
        self.setWindowTitle(_("title_ortho_update"))
        self.resize(1024, 768) # Fenêtre un peu plus large pour le mode Côte-à-Côte

        self.nb_impacted = 0
        self.nb_intact = 0
        self.stats_impacted = {}
        self.stats_intact = {}

        self.setup_ui()
        self.populate_grid()

        logging.info("Opening Orthophotos update window.")

    def setup_ui(self):
        # LAYOUT : Horizontal (Gauche = Grille, Droite = Contrôles)
        main_layout = QHBoxLayout(self)

        # --- GAUCHE : GRILLE D'APERÇU ---
        group_grid = QGroupBox(_("chk_tile_preview"))
        layout_grid = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("gridline-color: #444;")

        # Cacher le quadrillage
        self.table.setShowGrid(False)

        layout_grid.addWidget(self.table)

        # Légende
        lbl_legend = QLabel(
            "<span style='color:#9B59B6;'>ZL12</span> | "
            "<span style='color:#1ABC9C;'>ZL13</span> | "
            "<span style='color:#FF96C8;'>ZL14</span> | "
            "<span style='color:#64C8FF;'>ZL15</span> | "
            "<span style='color:#228B22;'>ZL16</span> | "
            "<span style='color:#FFFF32;'>ZL17</span> | "
            "<span style='color:#FF9600;'>ZL18</span> | "
            "<span style='color:#E61E1E;'>ZL19</span>"
        )
        lbl_legend.setAlignment(Qt.AlignCenter)
        lbl_legend.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout_grid.addWidget(lbl_legend)

        group_grid.setLayout(layout_grid)
        main_layout.addWidget(group_grid, stretch=2)

        # --- DROITE : CONTRÔLES ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 1. DOSSIERS
        group_dirs = QGroupBox(_("chk_processing_records"))
        layout_dirs = QVBoxLayout()

        layout_dirs.addWidget(QLabel(_("lbl_source_jpegs")))
        normalized_source = os.path.normpath(self.viewer.tex_dir_input.text()) if self.viewer.tex_dir_input.text() else ""
        self.source_input = QLineEdit(normalized_source)
        self.source_input.setReadOnly(True)
        self.source_input.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa; border: 1px solid #555;")
        layout_dirs.addWidget(self.source_input)

        layout_dirs.addWidget(QLabel(_("lbl_destination")))
        lay_dest = QHBoxLayout()
        default_dest = os.path.normpath(os.path.join(normalized_source, "TXP")) if normalized_source else ""
        self.dest_input = QLineEdit(default_dest)
        btn_browse = QPushButton(_("btn_browse"))
        btn_browse.clicked.connect(self.browse_dest)
        lay_dest.addWidget(self.dest_input)
        lay_dest.addWidget(btn_browse)
        layout_dirs.addLayout(lay_dest)

        group_dirs.setLayout(layout_dirs)
        right_layout.addWidget(group_dirs)

        # 2. BILAN ET ACTIONS
        group_actions = QGroupBox(_("chk_treatment"))
        layout_actions = QVBoxLayout()

        self.lbl_summary = QLabel(_("lbl_balance_sheet_being_calcu"))
        self.lbl_summary.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setMinimumHeight(180)
        layout_actions.addWidget(self.lbl_summary)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout_actions.addWidget(self.progress_bar)

        # --- ÉVOLUTION 2 : La case à cocher ---
        self.cb_copy_unmodified = QCheckBox(_("chk_copy_unedited_jpegs"))
        self.cb_copy_unmodified.setChecked(True)
        self.cb_copy_unmodified.setStyleSheet("margin-top: 10px;")
        self.cb_copy_unmodified.stateChanged.connect(self.update_summary_text)
        layout_actions.addWidget(self.cb_copy_unmodified)

        self.cb_lab_mode = QCheckBox(_("chk_lab_mode"))
        self.cb_lab_mode.setChecked(False)
        self.cb_lab_mode.setStyleSheet("margin-bottom: 5px;")
        layout_actions.addWidget(self.cb_lab_mode)

        lay_btns = QVBoxLayout()
        self.btn_validate = QPushButton(_("btn_validate_launch"))
        self.btn_validate.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 12px;")
        self.btn_validate.setEnabled(False)
        self.btn_validate.setMinimumHeight(45) # Protège l'espace du texte + padding
        self.btn_validate.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.btn_validate.clicked.connect(self.start_batch)

        self.btn_cancel = QPushButton(_("btn_cancel"))
        self.btn_cancel.setStyleSheet("padding: 8px;")
        self.btn_cancel.setMinimumHeight(35)
        self.btn_cancel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.btn_cancel.clicked.connect(self.reject)

        lay_btns.addWidget(self.btn_validate)
        lay_btns.addWidget(self.btn_cancel)
        layout_actions.addLayout(lay_btns)

        group_actions.setLayout(layout_actions)
        right_layout.addWidget(group_actions)

        # --- ÉVOLUTION 3 : Le texte d'information (Warning) ---
        self.lbl_warning_multi = QLabel(_("lbl_it_is_strongly_recommende"))
        self.lbl_warning_multi.setWordWrap(True)
        self.lbl_warning_multi.setStyleSheet("color: #e67e22; font-size: 20px; font-style: italic; margin-top: 10px;")
        right_layout.addWidget(self.lbl_warning_multi)

        right_layout.addStretch()
        main_layout.addWidget(right_panel, stretch=1)

    def update_summary_text(self):
        """Met à jour le texte du bilan dynamiquement avec ventilation par ZL."""
        do_copy = self.cb_copy_unmodified.isChecked()
        nb_copy = self.nb_intact if do_copy else 0

        # Construction du texte du bilan
        summary = _("lbl_summary_1")

        # 1. Détail des fichiers à modifier (Rouges)
        summary += f"• {self.nb_impacted} "
        summary += _("lbl_summary_2")
        if self.nb_impacted > 0:
            # On trie les clés pour afficher ZL16, puis ZL17, etc.
            for zl in sorted(self.stats_impacted.keys()):
                summary += f"    - ZL{zl} : {self.stats_impacted[zl]}\n"

        # 2. Détail des fichiers à copier (Verts)
        summary += f"• {nb_copy} "
        summary += _("lbl_summary_3")
        if do_copy and self.nb_intact > 0:
            for zl in sorted(self.stats_intact.keys()):
                summary += f"    - ZL{zl} : {self.stats_intact[zl]}\n"

        self.lbl_summary.setText(summary)

    def browse_dest(self):
        dir_path = QFileDialog.getExistingDirectory(self, _("dialog_select_dest_dir"), self.dest_input.text())
        if dir_path:
            clean_path = os.path.normpath(dir_path)
            self.dest_input.setText(clean_path)
            logging.info(f"Target folder modified by user : {dir_path}")

    def populate_grid(self):
        """Remplit la grille en respectant les différents niveaux de zoom (ZL16, ZL17, ZL18...)."""
        images_info = getattr(self.viewer, '_global_images_info', [])
        if not images_info:
            return

        logging.info("Analysis of JPEGs tiles with retouch mask in progress...")

        self.files_to_process = []
        self.files_to_copy = []

        min_bx = self.viewer._global_min_bx
        max_bx = self.viewer._global_max_bx
        min_by = self.viewer._global_min_by
        max_by = self.viewer._global_max_by
        base_size = self.viewer._global_base_size

        # =========================================================
        # 1. GESTION DYNAMIQUE DE LA RÉSOLUTION DE LA GRILLE
        # =========================================================
        # On trouve le ZL maximum pour définir la résolution de la grille
        max_zl = max((info['zl'] for info in images_info), default=16)
        if max_zl < 16:
            max_zl = 16

        # Le multiplicateur définit combien de sous-cellules composent un bloc ZL16
        # Ex: Si max_zl = 18, un bloc ZL16 est découpé en 4x4 cellules (2**(18-16) = 4)
        grid_mult = int(2 ** (max_zl - 16))

        # Dimensions totales de la table avec la nouvelle résolution
        rows = int((max_by - min_by + 1) * grid_mult)
        cols = int((max_bx - min_bx + 1) * grid_mult)

        self.table.setRowCount(rows)
        self.table.setColumnCount(cols)

        # FORCER DES CELLULES CARRÉES ET COMPACTES
        # On veut qu'un bloc ZL16 fasse toujours 48 pixels à l'écran.
        # Si on a du ZL18 (grid_mult = 4), la sous-cellule de base fera 48 / 4 = 12 pixels.
        # Si on est en full ZL16 (grid_mult = 1), la cellule fera 48 / 1 = 48 pixels.
        base_zl16_size = 48
        cell_size = max(1, int(base_zl16_size / grid_mult))

        # 1. LE SECRET EST ICI : On abaisse la limite minimale à 1 pixel
        self.table.horizontalHeader().setMinimumSectionSize(1)
        self.table.verticalHeader().setMinimumSectionSize(1)

        # 2. On fige le mode de redimensionnement
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)

        # 3. On applique notre taille cible qui sera enfin acceptée
        self.table.horizontalHeader().setDefaultSectionSize(cell_size)
        self.table.verticalHeader().setDefaultSectionSize(cell_size)

        # 4. (Sécurité absolue) On force chaque ligne/colonne
        for r in range(rows):
            self.table.setRowHeight(r, cell_size)
        for c in range(cols):
            self.table.setColumnWidth(c, cell_size)

        # Remplissage par défaut (Noir = Océan/Vide)
        for r in range(rows):
            for c in range(cols):
                item = QTableWidgetItem()
                item.setBackground(QColor(30, 30, 30))
                item.setToolTip("No file")
                self.table.setItem(r, c, item)

        mask = getattr(self.viewer, 'cumulative_mask', None)

        # =========================================================
        # NOUVEAU CODE : La source de vérité absolue (Delta Pixels)
        # =========================================================
        diff_mask = None
        global_data = getattr(self.viewer, '_global_canvas_data', None)
        original_data = getattr(self.viewer, '_original_canvas_data', None)

        if global_data is not None and original_data is not None:
            # Crée un masque booléen 2D ultra-rapide : True là où le pixel a changé
            # axis=-1 vérifie les 3 canaux (R, G, B) simultanément
            diff_mask = np.any(global_data != original_data, axis=-1)

        # =========================================================
        # 2. ANALYSE ET PLACEMENT DES IMAGES (DAMIER PAR ZL)
        # =========================================================

        # Nuances de gris pour les fichiers intacts (identiques pour tous les ZL)
        grey_pair = (QColor(100, 100, 100), QColor(80, 80, 80))

        # Palette selon les standards Ortho4XP pour les fichiers impactés
        color_palette = {
            12: {'impacted': (QColor(155, 89, 182),  QColor(142, 68, 173)),  'intact': grey_pair}, # Violet
            13: {'impacted': (QColor(26, 188, 156),  QColor(22, 160, 133)),  'intact': grey_pair}, # Turquoise
            14: {'impacted': (QColor(255, 150, 200), QColor(230, 120, 180)), 'intact': grey_pair}, # Rose
            15: {'impacted': (QColor(100, 200, 255), QColor(70, 170, 230)), 'intact': grey_pair}, # Bleu ciel
            16: {'impacted': (QColor(34, 139, 34),   QColor(0, 100, 0)),     'intact': grey_pair}, # Vert foncé
            17: {'impacted': (QColor(255, 255, 50),  QColor(220, 220, 0)),   'intact': grey_pair}, # Jaune
            18: {'impacted': (QColor(255, 150, 0),   QColor(220, 120, 0)),   'intact': grey_pair}, # Orange
            19: {'impacted': (QColor(230, 30, 30),   QColor(180, 0, 0)),     'intact': grey_pair}, # Rouge
        }

        # Fallback pour les ZL non définis
        default_colors = {'impacted': (QColor(200, 200, 200), QColor(150, 150, 150)), 'intact': grey_pair}

        for info in images_info:
            filename = os.path.basename(info['path'])
            zl = info['zl']

            # --- CALCUL DES COORDONNÉES DANS LA GRILLE HAUTE RÉSOLUTION ---
            col = int(round((info['block_x_16'] - min_bx) * grid_mult))
            row = int(round((info['block_y_16'] - min_by) * grid_mult))

            # --- TAILLE EN CELLULES (SPAN) ---
            span = int(2 ** (max_zl - zl))

            # Position et taille du JPEG dans le canevas global NumPy
            factor = 2 ** (zl - 16)
            size = int(base_size / factor)
            px = int((info['block_x_16'] - min_bx) * base_size)
            py = int((info['block_y_16'] - min_by) * base_size)

            # --- ANALYSE DE LA RETOUCHE ---
            is_impacted = False
            if diff_mask is not None:
                sub_mask = diff_mask[py:py+size, px:px+size]
                if np.any(sub_mask): # Plus besoin de "> 0", sub_mask est déjà booléen
                    is_impacted = True
                    self.files_to_process.append(info)
                else:
                    self.files_to_copy.append(info)
            else:
                self.files_to_copy.append(info)

            # --- PRÉPARATION VISUELLE ET COMPTAGE ---
            colors = color_palette.get(zl, default_colors)

            # Damier
            checker_index = ((info['til_x'] // 16) + (info['til_y'] // 16)) % 2

            if is_impacted:
                # On choisit la nuance rouge 0 ou 1 pour TOUT le fichier
                bg_color = colors['impacted'][checker_index]
                tooltip_text = f"🟥 {filename} (ZL{zl})"
                self.nb_impacted += 1
                self.stats_impacted[zl] = self.stats_impacted.get(zl, 0) + 1
            else:
                # On choisit la nuance verte 0 ou 1 pour TOUT le fichier
                bg_color = colors['intact'][checker_index]
                tooltip_text = f"⬜ {filename} (ZL{zl})"
                self.nb_intact += 1
                self.stats_intact[zl] = self.stats_intact.get(zl, 0) + 1

            # --- APPLICATION VISUELLE ---
            for r in range(row, row + span):
                for c in range(col, col + span):
                    if r < rows and c < cols:
                        item = self.table.item(r, c)
                        if item is not None:
                            item.setBackground(bg_color)
                            item.setToolTip(tooltip_text)

        # Mise à jour du résumé
        self.update_summary_text()

        # Activer le bouton uniquement s'il y a des modifs
        self.btn_validate.setEnabled(self.nb_impacted > 0)

        logging.info(f"Analysis done. Summary : {self.nb_impacted} to retouch, {self.nb_intact} to copy.")

    def get_optimal_dop(self, ram_per_worker_mb=500, max_ram_ratio=0.80):
        """Calcule le Degree Of Parallelism (DOP) idéal en fonction du CPU et de la RAM."""
        # 1. Calcul de la limite CPU (Standard Python 3.8+)
        cpu_cores = os.cpu_count() or 4
        cpu_limit = min(32, cpu_cores + 4)

        # 2. Calcul de la limite RAM
        available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)

        # On calcule combien de workers peuvent tenir dans 80% de cette RAM
        ram_limit = int((available_ram_mb * max_ram_ratio) / ram_per_worker_mb)

        logging.info(f"Detected resources : CPU_Limit={cpu_limit}, Avail_RAM={available_ram_mb:.0f}Mo -> RAM_Limit={ram_limit}")

        # 3. Arbitrage final
        optimal_dop = max(1, min(cpu_limit, ram_limit))

        return optimal_dop

    def start_batch(self):
        """Lance l'export parallèle ultra-rapide (Retouches + Copies) avec l'arborescence Ortho4XP."""
        nb_to_process = len(getattr(self, 'files_to_process', []))

        # Filtrer les copies selon la case à cocher
        do_copy = self.cb_copy_unmodified.isChecked()
        files_to_copy_actual = self.files_to_copy if do_copy else []
        nb_to_copy = len(files_to_copy_actual)

        if nb_to_process == 0 and nb_to_copy == 0:
            QMessageBox.information(self, _("msg_finished"), _("msg_no_images_to_process"))
            return

        dest_dir = self.dest_input.text()
        if not dest_dir: return

        # 1. Préparation de la liste des tâches unifiée
        tasks = [(info, True) for info in self.files_to_process] + \
                [(info, False) for info in files_to_copy_actual]
        total_files = len(tasks)

        # Détection préventive des fichiers existants
        existing_files = False
        # On utilise "is_impacted" au lieu de "_" pour ne pas écraser la fonction de traduction
        for info, is_impacted in tasks:
            provider_folder = os.path.basename(os.path.dirname(info['path']))
            check_path = os.path.join(dest_dir, provider_folder, os.path.basename(info['path']))
            if os.path.exists(check_path):
                existing_files = True
                break

        if existing_files:
            reply = QMessageBox.warning(self, _("msg_warning_existing_files"), _("msg_jpeg_files_already_exist_"),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                logging.info("Batch canceled by user (existing files).")
                return

        logging.info(f"Starting Batch to export to : {dest_dir}")

        if self.cb_lab_mode.isChecked():
            mode_str = "LAB Mode"
        else:
            mode_str = "RGB Mode"
        logging.info(mode_str)

        # On utilise "is_impacted"
        unique_folders = set(os.path.basename(os.path.dirname(info['path'])) for info, is_impacted in tasks)
        for folder in unique_folders:
            os.makedirs(os.path.join(dest_dir, folder), exist_ok=True)

        # 2. On fige l'interface
        self.btn_validate.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setFormat(_("msg_parallel_proc"))
        self.progress_bar.setValue(0)
        self.lbl_summary.setText(_("msg_multithread_on", tf=total_files, np=nb_to_process, nc=nb_to_copy))

        QApplication.processEvents()

        # On force l'affichage du sablier juste après le raffraichissement de l'UI
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            if self.cb_lab_mode.isChecked():
                # 3. Récupération du canevas global
                modified_canvas = self.viewer._global_canvas_data
            else:
                # 3. Pré-calcul mathématique du Delta
                original_f32 = self.viewer._original_canvas_data.astype(np.float32)
                modified_f32 = self.viewer._global_canvas_data.astype(np.float32)
                delta_global = modified_f32 - original_f32

            min_bx = self.viewer._global_min_bx
            min_by = self.viewer._global_min_by
            base_size = self.viewer._global_base_size

            # 4. La fonction de travail (Worker) isolée
            def process_task(task):
                info, is_impacted = task
                try:
                    filepath = info['path']
                    filename = os.path.basename(filepath)

                    # On cible le bon sous-dossier
                    provider_folder = os.path.basename(os.path.dirname(filepath))
                    dest_path = os.path.join(dest_dir, provider_folder, filename)

                    # CAS A : Fichier intact (Copie binaire)
                    if not is_impacted:
                        shutil.copy2(filepath, dest_path)
                        return True

                    # CAS B : Fichier impacté (Retouche OpenCV)
                    factor = 2 ** (info['zl'] - 16)
                    size_lr = int(base_size / factor)
                    px = int((info['block_x_16'] - min_bx) * base_size)
                    py = int((info['block_y_16'] - min_by) * base_size)

                    if self.cb_lab_mode.isChecked():
                        patch_ld_rgb = modified_canvas[py:py+size_lr, px:px+size_lr]

                        # 2. Chargement de l'image HD (qui est lue en BGR par OpenCV)
                        img_hd_bgr = cv2.imread(filepath)
                        if img_hd_bgr is None: return False
                        h_hd, w_hd = img_hd_bgr.shape[:2]

                        # 3. Conversion du patch LD en BGR pour harmoniser l'espace colorimétrique
                        img_ld_bgr = cv2.cvtColor(patch_ld_rgb, cv2.COLOR_RGB2BGR)

                        # 4. Redimensionnement de l'image LD à la taille HD
                        # INTER_CUBIC un peu moins net que INTER_LANCZOS4 proche de INTER_LINEAR
                        img_ld_resized = cv2.resize(img_ld_bgr, (w_hd, h_hd), interpolation=cv2.INTER_LANCZOS4)

                        # 5. Conversion dans l'espace LAB
                        lab_hd = cv2.cvtColor(img_hd_bgr, cv2.COLOR_BGR2LAB)
                        lab_ld = cv2.cvtColor(img_ld_resized, cv2.COLOR_BGR2LAB)

                        # 6. Séparation des canaux
                        l_hd, a_hd, b_hd = cv2.split(lab_hd)
                        l_ld, a_ld, b_ld = cv2.split(lab_ld)

                        # 7. Fusion : Luminance HD + Chrominance LD nettoyée
                        result_lab = cv2.merge([l_hd, a_ld, b_ld])

                        # 8. Reconversion en BGR pour l'enregistrement
                        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
                    else:
                        delta_patch = delta_global[py:py+size_lr, px:px+size_lr]

                        img_hd_bgr = cv2.imread(filepath)
                        if img_hd_bgr is None: return False

                        img_hd_rgb = cv2.cvtColor(img_hd_bgr, cv2.COLOR_BGR2RGB)
                        h_hd, w_hd = img_hd_rgb.shape[:2]

                        # INTER_CUBIC fait trop ressortir les imperfections
                        delta_hd = cv2.resize(delta_patch, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)

                        result_hd = img_hd_rgb.astype(np.float32) + delta_hd
                        result_hd = np.clip(result_hd, 0, 255).astype(np.uint8)

                        result_bgr = cv2.cvtColor(result_hd, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(dest_path, result_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                    # On supprime les grosses variables NumPy
                    try:
                        del img_hd_bgr     # L'image originale 8-bit
                        del img_hd_rgb     # La version RGB
                        del delta_hd       # Le masque/delta redimensionné (très lourd)
                        del result_hd      # Le résultat final en float32 (très lourd)
                        del result_bgr     # L'image finale
                        del patch_ld_rgb, img_ld_bgr, img_ld_resized, result_lab
                        del lab_hd, lab_ld, l_hd, a_hd, b_hd, l_ld, a_ld, b_ld
                    except NameError:
                        pass # Sécurité si une variable n'a pas été créée suite à une erreur

                    # On force Python à vider la corbeille immédiatement
                    gc.collect()

                    return True
                except Exception as e:
                    logging.error(f"Error on {info.get('path', 'unknown')}: {e}")
                    return False

            # 5. EXÉCUTION PARALLÈLE BRUTE
            # Calcul dynamique du nombre de threads idéal
            dop = self.get_optimal_dop()
            logging.info(f"Starting export with {dop} parallel workers.")

            with ThreadPoolExecutor(max_workers=dop) as executor:
                results = list(executor.map(process_task, tasks))

            processed_count = sum(results)
            logging.info(f"Export Batch done. {processed_count}/{total_files} files processed successfully.")

            self.viewer.is_retouch_exported = True

            # 6. Restauration de l'interface
            self.progress_bar.setFormat("%p%")
            self.progress_bar.setValue(100)
            self.lbl_summary.setText(_("txt_processing_completed"))

            self.btn_validate.setEnabled(True)
            self.btn_cancel.setEnabled(True)
            self.btn_cancel.setText(_("txt_close"))
            self.btn_cancel.clicked.disconnect()
            self.btn_cancel.clicked.connect(self.accept)

        finally:
            # On restaure le curseur normal quoiqu'il arrive
            QApplication.restoreOverrideCursor()

        # Le popup s'affiche avec le pointeur classique restauré
        QMessageBox.information(self, _("msg_success_title"), _("msg_files_processed", pc=processed_count, tf=total_files, dd=dest_dir))

# =======================================
#
# Sortie de la classe BatchExportDialog ici
#
# =======================================

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)

        # 1. Le bouton qui sert de titre #353535 #34495e
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("""
            QToolButton {
                background-color: #54799e;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
                text-align: left;
            }
            QToolButton:hover { background-color: #454545; }
            QToolButton:checked { background-color: #2a82da; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px;}
        """)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        # Pour que le bouton prenne toute la largeur disponible
        self.toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 2. La zone de contenu (qui sera masquée/affichée)
        self.content_area = QFrame()
        # self.content_area.setStyleSheet("""
        #     QFrame {
        #         border: 1px solid #555555;
        #         border-top: none;
        #         background-color: #2b2b2b;
        #     }
        # """)
        self.content_area.setVisible(False) # Masqué par défaut

        # 3. Layout principal du widget
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

    def on_pressed(self):
        # Inverse l'état (la flèche et la visibilité)
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if not checked else Qt.RightArrow)
        self.content_area.setVisible(not checked)

    def setContentLayout(self, layout):
        # Permet d'injecter ton layout de contrôles facilement
        self.content_area.setLayout(layout)

# =======================================
#
# Sortie de la classe CollapsibleBox ici
#
# =======================================

def setup_logging():
    """Configure le système de journalisation global de l'application."""

    # Création du logger principal
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # On écoute tout ce qui est INFO et supérieur

    # Définition du format visuel : [2026-04-19 14:30:15] [INFO] Texture chargée.
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 1. Sortie Fichier (Avec rotation)
    # maxBytes=5*1024*1024 (5 Mo max par fichier)
    # backupCount=3 (Garde app.log, app.log.1, app.log.2, app.log.3)
    file_handler = RotatingFileHandler("triangles_xp.log", maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Sortie Console (Pour continuer à voir les messages dans PyCharm/VSCode)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info("=== Starting Triangles XP application ===")

def apply_dark_theme(app):
    """Applique un thème sombre professionnel à l'ensemble de l'application PyQt5."""
    app.setStyle("Fusion")

    dark_palette = QPalette()

    # Couleurs de fond principales
    dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))

    # Infobulles
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)

    # Textes et Boutons (ceux qui n'ont pas de styleSheet personnalisé)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)

    # Éléments sélectionnés (Bleu)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(dark_palette)

    # Ajout d'une feuille de style globale pour affiner les bordures (facultatif mais plus propre)
    app.setStyleSheet("""
        QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }
        QGroupBox { font-weight: bold; border: 1px solid #555555; margin-top: 2ex; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }
        QTabWidget::pane { border: 1px solid #555555; }
        QTabBar::tab {
            background: #353535;
            color: white;
            padding: 8px 12px;
            border: 1px solid #555;
        }
        QTabBar::tab:selected {
            background: #34495e;
            border-bottom-color: #454545;
        }
    """)

if __name__ == "__main__":
    setup_logging()

    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    config_file = "settings.ini"
    config = configparser.ConfigParser()
    lang = "en"
    if os.path.exists(config_file):
        config.read(config_file, encoding='utf-8')
        if config.has_section('Lang'):
            lang = config.get('Lang', 'language', fallback='en')
    load_language(lang)

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = OrthoMeshViewer()
    window.show()

    exit_code = app.exec_()
    logging.info("=== Closing application ===")
    sys.exit(exit_code)