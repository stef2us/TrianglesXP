import sys
import json
import os

# Dictionnaire global contenant les traductions chargées
_strings = {}

def get_base_dir():
    """Retourne le dossier où se trouve l'exécutable (ou le script)."""
    if getattr(sys, 'frozen', False):
        # Si on est compilé en .exe par PyInstaller
        return os.path.dirname(sys.executable)
    else:
        # Si on tourne en Python normal
        return os.path.dirname(os.path.abspath(__file__))

def load_language(lang_code="en"):
    """Charge le fichier JSON correspondant à la langue."""
    global _strings

    base_path = get_base_dir()
    file_path = os.path.join(base_path, "locales", f"{lang_code}.json")

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            _strings = json.load(f)
    else:
        # Fallback de sécurité si le fichier n'existe pas
        _strings = {}

def _(key, **kwargs):
    """
    Fonction de traduction principale.
    Utilisation: _("btn_load_mesh") ou _("msg_mesh_loaded", nodes=100, tris=50)
    """
    # Si la clé n'existe pas, on retourne la clé elle-même pour le repérer vite (ou une valeur par défaut)
    text = _strings.get(key, f"[{key}]")

    # Gestion de l'interpolation dynamique (variables dans les textes)
    if kwargs:
        try:
            return text.format(**kwargs)
        except KeyError:
            return text

    return text