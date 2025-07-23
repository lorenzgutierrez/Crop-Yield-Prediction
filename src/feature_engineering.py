import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Union
import warnings
class FeatureGenerator(object):

    def __init__(self, df: pd.DataFrame, settings: Dict):
        """
        Inicializa el generador de características con logging y validación
        
        Args:
            df: DataFrame con los datos base para generar características
        """
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('feature_generator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.df =  df.copy()
        self.settings = settings
        self.encoders: Optional[Dict[str, LabelEncoder]] = None
        self.logger.info(f"FeatureGenerator inicializado con DataFrame de shape: {self.df.shape}")
        self.logger.info(f"Columnas disponibles: {list(self.df.columns)}")

    def generate_pesticides_quartiles(self):
        """Genera quartiles globales para pesticidas"""
        self.logger.info("Generando quartiles de pesticidas...")
        df = self.df.copy()
        non_null_pesticides = df['Value_pesticides'].notna().sum()
        if non_null_pesticides == 0:
                raise ValueError(f"No hay valores no nulos en Value_pesticides")
        global_quartiles = df.Value_pesticides.quantile([0.25, 0.5, 0.75])
        self.logger.info(f"Quartiles calculados: Q1={global_quartiles[0.25]:.2f}, "
                           f"Q2={global_quartiles[0.5]:.2f}, Q3={global_quartiles[0.75]:.2f}")

        df['pesticide_quartile_global'] = pd.cut(
                df['Value_pesticides'],
                bins=[-np.inf, global_quartiles[0.25], global_quartiles[0.5], 
                    global_quartiles[0.75], np.inf],
                labels=['Q1', 'Q2', 'Q3', 'Q4']
            )
        self.df = df

    def generate_pesticide_efficiency_feature(self, n_years: int = 3)-> None:
        """
        Genera característica de eficiencia de pesticidas
        
        Args:
            n_years: Número de años para ventana móvil
        """
        self.logger.info(f"Generando característica de eficiencia de pesticidas (ventana: {n_years} años)...")
        df = self.df.copy()
        df = df.sort_values(['Area', 'Item', 'Year'])
        if df['Value_pesticides'].isna().all():
                raise ValueError("No hay datos de pesticidas disponibles")
        df['pesticide_efficiency'] = df.Value/(df.Value_pesticides + 1)
        df[f'pesticide_efficiency_{n_years}'] = df.groupby(['Area', 'Item'])['pesticide_efficiency'].transform(
                lambda x: x.rolling(window=n_years, min_periods=1).mean()
            )
        self.df = df

    def generate_climate_features(self)-> None:
        """Genera características relacionadas con clima y temperatura"""
        self.logger.info("Generando características climáticas...")
        df = self.df.copy()
        crop_profiles = { # Creados con IA (ver Sandbox.ipynb)
                'Maize': {'optimal_temp': 25, 'temp_tolerance': 5, 'water_intensive': True, 'tech_responsive': True},
                'Wheat': {'optimal_temp': 18, 'temp_tolerance': 7, 'water_intensive': False, 'tech_responsive': True},
                'Rice': {'optimal_temp': 28, 'temp_tolerance': 4, 'water_intensive': True, 'tech_responsive': True},
                'Soybeans': {'optimal_temp': 23, 'temp_tolerance': 6, 'water_intensive': False, 'tech_responsive': True},
                'Barley': {'optimal_temp': 16, 'temp_tolerance': 8, 'water_intensive': False, 'tech_responsive': False},
                'Cassava': {'optimal_temp': 30, 'temp_tolerance': 3, 'water_intensive': False, 'tech_responsive': False},
                'Potatoes': {'optimal_temp': 20, 'temp_tolerance': 5, 'water_intensive': True, 'tech_responsive': True},
                'Sugar cane': {'optimal_temp': 32, 'temp_tolerance': 4, 'water_intensive': True, 'tech_responsive': False},
            }
        def get_crop_profile(crop_name, attribute, default=25)-> None:
            """Obtiene atributo del perfil de cultivo"""
            for crop, profile in crop_profiles.items():
                if crop.lower() in crop_name.lower():
                    return profile.get(attribute, default)
            return default
    
        df["optimal_temp"] = df.Item.apply(lambda x: get_crop_profile(x, "optimal_temp", 25))
        df["temp_tolerance"] = df.Item.apply(lambda x: get_crop_profile(x, "temp_tolerance", 4))

        df["temp_stress"] = (df["Value_temperature"] - df["optimal_temp"])**2
        df["temp_dentro_optimo"] = np.abs((df["Value_temperature"] - df["optimal_temp"]) <= df["temp_tolerance"]).astype(int)

        for lag in [1,2]:
            df[f"temp_lag{lag}"] = df.Value_temperature.shift(lag)

        df['stress_category'] = pd.cut(df['temp_stress'], 
                    bins=[0, 1, 5, 10, float('inf')],
                    labels=['Bajo', 'Moderado', 'Alto', 'Extremo'],
                    include_lowest=True)
        
        cut_points = np.linspace(0, 1,4)
        quantile_values = df["Value_temperature"].dropna().quantile(cut_points)
        labels = ['Frío'
                , 'Moderado'
                , 'Caliente']
        
        df["temp_category"] = pd.cut(
            df["Value_temperature"], 
            bins=quantile_values, 
            labels=labels, 
            include_lowest=True
        )

        self.df = df
        self.logger.info(f"Generadascaracterísticas climáticas")


    def generate_interaction_features(self)-> None:
        """Genera características de interacción entre variables"""
        self.logger.info("Generando características de interacción...")

        df = self.df.copy()
        df['rainfall_temp_interaction'] = df['Value_rainfall']*df['Value_temperature']
        df['rainfall_pest_interaction'] = df['Value_rainfall']*df['Value_pesticides']
        df['pest_temp_interaction'] = df['Value_pesticides']*df['Value_temperature']
        self.logger.info(f"Generadas características de interacción")

        self.df =  df

    
    def create_label_encoded_features(self):
        """Crea características codificadas con LabelEncoder"""
        self.logger.info("Creando características codificadas...")

        df = self.df.copy()
        area_encoder = LabelEncoder()
        df['Area_encoded'] = area_encoder.fit_transform(df.Area)
        self.logger.info("Encoding de Area generado")

        item_encoder = LabelEncoder()
        df['Item_encoded'] = item_encoder.fit_transform(df.Item)
        self.logger.info("Encoding de ITem generado")

        df['Area_Item'] = df.Area.astype(str) + '-' + df.Item.astype(str)
        area_item_encoder = LabelEncoder()
        df['Area_Item'] = area_item_encoder.fit_transform(df.Area_Item)
        self.logger.info("Encoding de Area-item generado")

        pesticides_quartile_encoder = LabelEncoder()
        df['pesticide_quartile_global'] = pesticides_quartile_encoder.fit_transform(df.pesticide_quartile_global)
        self.logger.info("Encoding de pesticides quartiles generado")

        stress_encoder = LabelEncoder()
        df['stress_category'] = stress_encoder.fit_transform(df.stress_category)
        self.logger.info("Encoding de stress_category generado")

        temp_category_encoder = LabelEncoder()
        df['temp_category']  = temp_category_encoder.fit_transform(df.temp_category)
        self.logger.info("Encoding de temp_category generado")

        encoders = {
            'area_encoder': area_encoder,
            'item_encoder': item_encoder,
            'area_item_encoder': area_item_encoder,
            'pesticide_quartile_encoder': pesticides_quartile_encoder,
            'stress_category_encoder': stress_encoder,
            'temp_category_encoder': temp_category_encoder
        }

        self.df = df
        self.encoders = encoders
        self.logger.info(f"Creados {len(encoders)} encoders para características categóricas")


    def generate_features(self, n_years: int = 5) -> None:
        """
        Ejecuta todo el pipeline de generación de características
        
        Args:
            n_years: Número de años para ventana móvil en eficiencia de pesticidas
        """
        initial_shape = self.df.shape
        initial_columns = set(self.df.columns)

        self.generate_pesticides_quartiles()
        self.generate_pesticide_efficiency_feature(n_years= n_years)
        self.generate_climate_features()
        self.create_label_encoded_features()

        final_shape = self.df.shape
        final_columns = set(self.df.columns)
        new_columns = final_columns - initial_columns

        self.logger.info("=== RESUMEN DE GENERACIÓN DE CARACTERÍSTICAS ===")
        self.logger.info(f"Shape inicial: {initial_shape}")
        self.logger.info(f"Shape final: {final_shape}")
        self.logger.info(f"Nuevas características creadas: {len(new_columns)}")
        self.logger.info(f"Características nuevas: {sorted(list(new_columns))}")
        
        if self.encoders:
            self.logger.info(f"Encoders disponibles: {list(self.encoders.keys())}")

    def save_data(self) -> None:
        """ Guarda los datos con los features procesados"""
        output_path = Path(self.settings['output_path']) / 'data_processed.csv'
        self.df.to_csv(output_path, index=False)
        self.logger.info(f"Datos guardados exitosamente en: {output_path}")
        self.logger.info(f"Archivo final: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
                