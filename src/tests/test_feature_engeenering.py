# test_feature_generator.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from feature_engineering import FeatureGenerator  

@pytest.fixture
def sample_data():
    """Fixture con datos de muestra para testing"""
    np.random.seed(42)  # Para reproducibilidad
    
    data = {
        'Area': ['USA', 'Brazil', 'China', 'India', 'France'] * 4,
        'Item': ['Wheat', 'Maize', 'Rice', 'Soybeans', 'Potatoes'] * 4,
        'Year': [2018, 2019, 2020, 2021] * 5,
        'Value': np.random.uniform(1000, 5000, 20),  # Producción
        'Value_pesticides': np.random.uniform(10, 100, 20),  # Pesticidas
        'Value_temperature': np.random.uniform(15, 35, 20),  # Temperatura
        'Value_rainfall': np.random.uniform(300, 1500, 20)   # Lluvia
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def feature_generator(sample_data):
    """Fixture que crea una instancia de FeatureGenerator"""
    settings = {'data_path': '/tmp'}
    return FeatureGenerator(sample_data, settings)

@pytest.fixture
def problematic_data():
    """Fixture con datos problemáticos para testing de edge cases"""
    data = {
        'Area': ['USA', 'Brazil', 'China'],
        'Item': ['Wheat', 'Maize', 'Rice'],
        'Year': [2020, 2021, 2022],
        'Value': [1000, 2000, 3000],
        'Value_pesticides': [np.nan, np.nan, np.nan],  # Todos NaN
        'Value_temperature': [20, 25, 30],
        'Value_rainfall': [500, 800, 1200]
    }
    return pd.DataFrame(data)

class TestPesticidesQuartiles:
    """Test 1: Generación de quartiles de pesticidas"""
    
    def test_generate_pesticides_quartiles_success(self, feature_generator):
        """Test exitoso de generación de quartiles de pesticidas"""
        # Ejecutar el método
        feature_generator.generate_pesticides_quartiles()
        
        # Verificar que se creó la nueva columna
        assert 'pesticide_quartile_global' in feature_generator.df.columns
        
        # Verificar que tiene las categorías correctas
        expected_categories = ['Q1', 'Q2', 'Q3', 'Q4']
        actual_categories = feature_generator.df['pesticide_quartile_global'].cat.categories.tolist()
        assert actual_categories == expected_categories
        
        # Verificar que no hay valores nulos donde había datos originales
        original_not_null = feature_generator.df['Value_pesticides'].notna()
        quartile_not_null = feature_generator.df['pesticide_quartile_global'].notna()
        
        # Donde había datos de pesticidas, debe haber quartiles
        assert (original_not_null == quartile_not_null).all()
        
        # Verificar que los quartiles están balanceados aproximadamente
        quartile_counts = feature_generator.df['pesticide_quartile_global'].value_counts()
        # Cada quartil debería tener aproximadamente 25% de los datos
        total_data = quartile_counts.sum()
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
            percentage = (quartile_counts[quartile] / total_data) * 100
            assert 15 <= percentage <= 35, f"Quartil {quartile} tiene {percentage:.1f}% de datos"
    
    def test_generate_pesticides_quartiles_with_nulls(self, feature_generator):
        """Test con algunos valores nulos en pesticidas"""
        # Introducir algunos NaN
        feature_generator.df.loc[0:2, 'Value_pesticides'] = np.nan
        
        # Ejecutar el método
        feature_generator.generate_pesticides_quartiles()
        
        # Verificar que se creó la columna
        assert 'pesticide_quartile_global' in feature_generator.df.columns
        
        # Verificar que los NaN originales siguen siendo NaN en quartiles
        original_nulls = feature_generator.df['Value_pesticides'].isna()
        quartile_nulls = feature_generator.df['pesticide_quartile_global'].isna()
        assert (original_nulls == quartile_nulls).all()
    
    def test_generate_pesticides_quartiles_all_nulls_raises_error(self, problematic_data):
        """Test que verifica error cuando todos los valores de pesticidas son nulos"""
        settings = {'data_path': '/tmp'}
        feature_generator = FeatureGenerator(problematic_data, settings)
        
        # Debe lanzar ValueError
        with pytest.raises(ValueError, match="No hay valores no nulos en Value_pesticides"):
            feature_generator.generate_pesticides_quartiles()
    
    def test_quartiles_mathematical_correctness(self, feature_generator):
        """Test que verifica la correctitud matemática de los quartiles"""
        # Usar datos controlados para verificar cálculos
        controlled_data = pd.DataFrame({
            'Area': ['A'] * 12,
            'Item': ['Wheat'] * 12,
            'Year': list(range(2010, 2022)),
            'Value': [1000] * 12,
            'Value_pesticides': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Datos ordenados
            'Value_temperature': [20] * 12,
            'Value_rainfall': [500] * 12
        })
        
        settings = {'data_path': '/tmp'}
        fg = FeatureGenerator(controlled_data, settings)
        fg.generate_pesticides_quartiles()
        
        # Verificar que los quartiles se asignan correctamente
        quartiles = fg.df.groupby('pesticide_quartile_global')['Value_pesticides'].agg(['min', 'max'])
        
        # Q1 debe tener los valores más bajos, Q4 los más altos
        assert quartiles.loc['Q1', 'max'] <= quartiles.loc['Q2', 'min']
        assert quartiles.loc['Q2', 'max'] <= quartiles.loc['Q3', 'min']
        assert quartiles.loc['Q3', 'max'] <= quartiles.loc['Q4', 'min']

class TestClimateFeatures:
    """Test 2: Generación de características climáticas"""
    
    def test_generate_climate_features_success(self, feature_generator):
        """Test exitoso de generación de características climáticas"""
        # Ejecutar el método
        feature_generator.generate_climate_features()
        
        # Verificar que se crearon todas las columnas esperadas
        expected_columns = [
            'optimal_temp', 'temp_tolerance', 'temp_stress', 
            'temp_dentro_optimo', 'temp_lag1', 'temp_lag2',
            'stress_category', 'temp_category'
        ]
        
        for col in expected_columns:
            assert col in feature_generator.df.columns, f"Columna faltante: {col}"
        
        # Verificar que optimal_temp tiene valores razonables
        assert feature_generator.df['optimal_temp'].min() >= 10, "Temperatura óptima muy baja"
        assert feature_generator.df['optimal_temp'].max() <= 40, "Temperatura óptima muy alta"
        
        # Verificar que temp_stress es siempre positivo (es una diferencia al cuadrado)
        assert (feature_generator.df['temp_stress'] >= 0).all(), "temp_stress debe ser >= 0"
        
        # Verificar que temp_dentro_optimo es binario (0 o 1)
        temp_dentro_values = feature_generator.df['temp_dentro_optimo'].unique()
        assert set(temp_dentro_values).issubset({0, 1}), "temp_dentro_optimo debe ser 0 o 1"
        
        # Verificar categorías de stress
        expected_stress_categories = ['Bajo', 'Moderado', 'Alto', 'Extremo']
        actual_stress_categories = feature_generator.df['stress_category'].cat.categories.tolist()
        assert actual_stress_categories == expected_stress_categories
        
        # Verificar categorías de temperatura
        expected_temp_categories = ['Frío', 'Moderado', 'Caliente']
        actual_temp_categories = feature_generator.df['temp_category'].cat.categories.tolist()
        assert actual_temp_categories == expected_temp_categories
    
    def test_crop_specific_optimal_temperatures(self, feature_generator):
        """Test que verifica temperaturas óptimas específicas por cultivo"""
        feature_generator.generate_climate_features()
        
        # Verificar algunos valores específicos conocidos
        crop_temps = feature_generator.df.groupby('Item')['optimal_temp'].first()
        
        # Verificar algunos cultivos específicos (basado en crop_profiles del código)
        if 'Wheat' in crop_temps:
            assert crop_temps['Wheat'] == 18, "Temperatura óptima de trigo debe ser 18°C"
        
        if 'Rice' in crop_temps:
            assert crop_temps['Rice'] == 28, "Temperatura óptima de arroz debe ser 28°C"
        
        if 'Maize' in crop_temps:
            assert crop_temps['Maize'] == 25, "Temperatura óptima de maíz debe ser 25°C"
    
    def test_lag_features_correctness(self, feature_generator):
        """Test que verifica la correctitud de las características lag"""
        # Usar datos ordenados para verificar lag
        feature_generator.df = feature_generator.df.sort_values(['Area', 'Item', 'Year'])
        feature_generator.generate_climate_features()
        
        # Verificar que lag1 y lag2 son correctos
        df = feature_generator.df
        
        # Para cada grupo Area-Item, verificar que lag funciona
        for (area, item), group in df.groupby(['Area', 'Item']):
            if len(group) >= 2:
                # lag1: el valor actual debe ser igual al lag1 del siguiente año
                for i in range(len(group) - 1):
                    current_temp = group.iloc[i]['Value_temperature']
                    next_lag1 = group.iloc[i + 1]['temp_lag1']
                    
                    # Si ambos no son NaN, deben ser iguales
                    if pd.notna(current_temp) and pd.notna(next_lag1):
                        assert abs(current_temp - next_lag1) < 1e-10, "Error en cálculo de lag1"

class TestLabelEncodedFeatures:
    """Test 3: Creación de características codificadas"""
    
    def test_create_label_encoded_features_success(self, feature_generator):
        """Test exitoso de creación de características codificadas"""
        # Primero necesitamos generar las características categóricas
        feature_generator.generate_pesticides_quartiles()
        feature_generator.generate_climate_features()
        
        # Ejecutar el método de encoding
        feature_generator.create_label_encoded_features()
        
        # Verificar que se crearon las columnas codificadas
        expected_encoded_columns = [
            'Area_encoded', 'Item_encoded', 'Area_Item'
        ]
        
        for col in expected_encoded_columns:
            assert col in feature_generator.df.columns, f"Columna codificada faltante: {col}"
        
        # Verificar que los encoders se guardaron
        assert feature_generator.encoders is not None
        expected_encoders = [
            'area_encoder', 'item_encoder', 'area_item_encoder',
            'pesticide_quartile_encoder', 'stress_category_encoder', 'temp_category_encoder'
        ]
        
        for encoder_name in expected_encoders:
            assert encoder_name in feature_generator.encoders, f"Encoder faltante: {encoder_name}"
        
        # Verificar que las características codificadas son numéricas
        assert feature_generator.df['Area_encoded'].dtype in ['int32', 'int64']
        assert feature_generator.df['Item_encoded'].dtype in ['int32', 'int64']
        
        # Verificar que el rango de valores codificados es correcto
        n_areas = feature_generator.df['Area'].nunique()
        n_items = feature_generator.df['Item'].nunique()
        
        assert feature_generator.df['Area_encoded'].min() >= 0
        assert feature_generator.df['Area_encoded'].max() == n_areas - 1
        assert feature_generator.df['Item_encoded'].min() >= 0
        assert feature_generator.df['Item_encoded'].max() == n_items - 1
    
    def test_encoder_consistency(self, feature_generator):
        """Test que verifica consistencia en el encoding"""
        # Generar características necesarias
        feature_generator.generate_pesticides_quartiles()
        feature_generator.generate_climate_features()
        feature_generator.create_label_encoded_features()
        
        # Verificar que mismo valor categórico = mismo valor codificado
        df = feature_generator.df
        
        # Test para Area
        for area in df['Area'].unique():
            area_codes = df[df['Area'] == area]['Area_encoded'].unique()
            assert len(area_codes) == 1, f"Area '{area}' tiene múltiples códigos: {area_codes}"
        
        # Test para Item
        for item in df['Item'].unique():
            item_codes = df[df['Item'] == item]['Item_encoded'].unique()
            assert len(item_codes) == 1, f"Item '{item}' tiene múltiples códigos: {item_codes}"
    
    def test_encoder_inverse_transform(self, feature_generator):
        """Test que verifica que los encoders pueden hacer transformación inversa"""
        # Generar características necesarias
        feature_generator.generate_pesticides_quartiles()
        feature_generator.generate_climate_features()
        feature_generator.create_label_encoded_features()
        
        # Test de transformación inversa para Area
        area_encoder = feature_generator.encoders['area_encoder']
        original_areas = feature_generator.df['Area'].unique()
        encoded_areas = feature_generator.df['Area_encoded'].unique()
        
        # Transformación inversa debe recuperar valores originales
        decoded_areas = area_encoder.inverse_transform(encoded_areas)
        assert set(decoded_areas) == set(original_areas), "Error en transformación inversa de Area"
        
        # Test de transformación inversa para Item
        item_encoder = feature_generator.encoders['item_encoder']
        original_items = feature_generator.df['Item'].unique()
        encoded_items = feature_generator.df['Item_encoded'].unique()
        
        decoded_items = item_encoder.inverse_transform(encoded_items)
        assert set(decoded_items) == set(original_items), "Error en transformación inversa de Item"

class TestIntegrationAndEdgeCases:
    """Tests de integración y casos extremos"""
    
    def test_generate_features_full_pipeline(self, feature_generator):
        """Test del pipeline completo de generación de características"""
        initial_columns = set(feature_generator.df.columns)
        
        # Ejecutar pipeline completo
        feature_generator.generate_features(n_years=3)
        
        final_columns = set(feature_generator.df.columns)
        new_columns = final_columns - initial_columns
        
        # Verificar que se agregaron características
        assert len(new_columns) > 10, f"Pocas características nuevas: {len(new_columns)}"
        
        # Verificar que el DataFrame mantiene el mismo número de filas
        assert len(feature_generator.df) == 20, "El número de filas cambió inesperadamente"
        
        # Verificar que los encoders están disponibles
        assert feature_generator.encoders is not None
        assert len(feature_generator.encoders) == 6
    
    def test_empty_dataframe_handling(self):
        """Test con DataFrame vacío"""
        empty_df = pd.DataFrame(columns=['Area', 'Item', 'Year', 'Value', 'Value_pesticides', 'Value_temperature', 'Value_rainfall'])
        settings = {'data_path': '/tmp'}
        
        fg = FeatureGenerator(empty_df, settings)
        
        # Debe manejar DataFrames vacíos sin crashear
        with pytest.raises((ValueError, IndexError)):
            fg.generate_features()

# Configuración para ejecutar los tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])