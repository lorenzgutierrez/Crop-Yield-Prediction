import pytest
import tempfile
import pandas as pd
from pathlib import Path
import shutil

@pytest.fixture
def temp_data_dir():
    """Crea un directorio temporal con datos de prueba"""
    temp_dir = tempfile.mkdtemp()
    print(f"\n🔧 Creando directorio temporal: {temp_dir}")
    
    try:
        # Crear datos de prueba básicos
        basic_data = pd.DataFrame({
            'Area': ['USA', 'Brazil', 'China'],
            'Year': [2010, 2011, 2012],
            'Value': [100, 200, 300]
        })
        
        # Datos específicos para cada archivo
        rainfall_data = pd.DataFrame({
            ' Area': ['USA', 'Brazil', 'China'],  # Nota el espacio
            'Year': [2010, 2011, 2012],
            'average_rain_fall_mm_per_year': [800, 1200, 600]
        })
        
        temp_data = pd.DataFrame({
            'country': ['USA', 'Brazil', 'China'],
            'year': [2010, 2011, 2012],
            'avg_temp': [15.5, 25.2, 12.8]
        })
        
        yield_data = pd.DataFrame({
            'Area': ['USA', 'Brazil', 'China'],
            'Year': [2010, 2011, 2012],
            'Item': ['Wheat', 'Soybean', 'Rice'],
            'Value': [1000, 2000, 3000]
        })
        
        # Crear archivos CSV
        files_to_create = {
            'pesticides.csv': basic_data,
            'rainfall.csv': rainfall_data,
            'temp.csv': temp_data,
            'yield.csv': yield_data
        }
        
        for filename, data in files_to_create.items():
            file_path = Path(temp_dir) / filename
            data.to_csv(file_path, index=False)
            print(f"📁 Creado: {filename} ({len(data)} filas)")
            
            # Verificar que se creó correctamente
            assert file_path.exists(), f"No se pudo crear {filename}"
            assert file_path.stat().st_size > 0, f"{filename} está vacío"
        
        print(f"✅ Todos los archivos creados en: {temp_dir}")
        yield temp_dir
        
    finally:
        # Limpiar
        print(f"🧹 Limpiando directorio temporal: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestDataLoading:
    def test_read_data_success(self, preprocesser):
        """Test de lectura exitosa de datos"""
        print(f"\n🧪 Iniciando test_read_data_success")
        print(f"📂 Data path: {preprocesser.data_path}")
        
        # Verificar que el directorio existe
        data_path = Path(preprocesser.data_path)
        assert data_path.exists(), f"Directorio no existe: {data_path}"
        
        # Verificar que los archivos existen
        required_files = ['pesticides.csv', 'rainfall.csv', 'temp.csv', 'yield.csv']
        for filename in required_files:
            file_path = data_path / filename
            print(f"🔍 Verificando: {filename} - Existe: {file_path.exists()}")
            assert file_path.exists(), f"Archivo faltante: {filename}"
        
        # Ejecutar read_data
        try:
            print("📖 Ejecutando read_data()...")
            preprocesser.read_data()
            print("✅ read_data() completado")
        except Exception as e:
            print(f"❌ Error en read_data(): {e}")
            print(f"Tipo de error: {type(e).__name__}")
            raise
        
        # Verificaciones
        assert preprocesser.pesticides is not None, "pesticides no se cargó"
        assert preprocesser.rainfall is not None, "rainfall no se cargó"
        assert preprocesser.temp is not None, "temp no se cargó"
        assert preprocesser.yield_df is not None, "yield_df no se cargó"
        
        # Verificar que tienen datos
        assert len(preprocesser.pesticides) > 0, "pesticides está vacío"
        assert len(preprocesser.rainfall) > 0, "rainfall está vacío"
        assert len(preprocesser.temp) > 0, "temp está vacío"
        assert len(preprocesser.yield_df) > 0, "yield_df está vacío"
        
        print("✅ Todos los DataFrames cargados correctamente")



    def test_generate_data_dict(self, preprocesser):
        """Test de generación del diccionario de datos"""
        preprocesser.read_data()
        preprocesser.rename_columns()
        preprocesser.select_columns()
        preprocesser.remove_duplicates()
        preprocesser.generate_data_dict()
        
        # Verificar que el diccionario fue creado correctamente
        assert preprocesser.dfs_dict is not None
        assert 'rainfall' in preprocesser.dfs_dict
        assert 'pesticides' in preprocesser.dfs_dict
        assert 'temperature' in preprocesser.dfs_dict

    def test_remove_duplicates(self, preprocesser):
        """Test de remoción de duplicados"""
        preprocesser.read_data()
        preprocesser.rename_columns()
        preprocesser.select_columns()
        
        preprocesser.remove_duplicates()
        
        # Verificar que no hay duplicados en combinaciones Area-Year
        pesticides_unique = preprocesser.pesticides.groupby(['Area', 'Year']).size()
        assert all(pesticides_unique == 1)
        
        rainfall_unique = preprocesser.rainfall.groupby(['Area', 'Year']).size()
        assert all(rainfall_unique == 1)
        
        temp_unique = preprocesser.temp.groupby(['Area', 'Year']).size()
        assert all(temp_unique == 1)
        
        # Para yield_df, la combinación única debe ser Area-Item-Year
        yield_unique = preprocesser.yield_df.groupby(['Area', 'Item', 'Year']).size()
        assert all(yield_unique == 1)

    def test_no_empty_dataframes(self, preprocesser):
        """Test que verifica que no haya DataFrames vacíos"""
        preprocesser.read_data()
        
        assert not preprocesser.pesticides.empty
        assert not preprocesser.rainfall.empty
        assert not preprocesser.temp.empty
        assert not preprocesser.yield_df.empty

if __name__ == "__main__":
    pytest.main([__file__, "-v"])