import tempfile
from data_pipeline import Preprocesser
import pytest
import shutil

@pytest.fixture
def temp_data_dir():
    """Crea directorio temporal con datos de prueba"""
    temp_dir = tempfile.mkdtemp()
    
    # Crear datos de prueba (igual que arriba)
    # ...
    
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def preprocesser(temp_data_dir):
    """Crea instancia de Preprocesser"""
    settings = {'data_path': temp_data_dir}
    return Preprocesser(settings)