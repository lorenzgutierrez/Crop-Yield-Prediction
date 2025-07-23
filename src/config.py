
import os
from pathlib import Path

# Configuración básica del proyecto
SETTINGS = {
    'data_path': r'data\archive',
    'output_path': 'data/processed',
    'models_path': 'models',
    'project_name': 'Crop Yield Prediction',
    'version': '1.0.0',
    'tune_rf': True
}

# Crear directorios si no existen
def create_directories():
    """Crea directorios necesarios"""
    dirs = [SETTINGS['output_path'], SETTINGS['models_path']]
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Función para obtener configuración
def get_settings():
    """Retorna configuración del proyecto"""
    create_directories()
    return SETTINGS.copy()