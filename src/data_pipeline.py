import pandas as pd
from fuzzywuzzy import fuzz, process
import re
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any

class Preprocesser(object):
    # Loggings y documentacion basica hecha con IA
    def __init__(self, settings: Dict[str, Any]):
        """
        Inicializa el preprocessor con configuración y logging
        
        Args:
            settings: Diccionario con configuración, debe contener 'data_path'
        """
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('preprocesser.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.data_path = settings['data_path']

        self.logger.info(f"Preprocesser inicializado con data_path: {self.data_path}")
        
        # Inicializar DataFrames como None
        self.pesticides: Optional[pd.DataFrame] = None
        self.rainfall: Optional[pd.DataFrame] = None
        self.temp: Optional[pd.DataFrame] = None
        self.yield_df: Optional[pd.DataFrame] = None
        self.dfs_dict: Optional[Dict[str, pd.DataFrame]] = None
        self.df_final: Optional[pd.DataFrame] = None

    def read_data(self) -> None:
        """Lee todos los archivos CSV y valida su estructura"""
        self.logger.info("Iniciando lectura de datos...")
        # Leer archivos
        pesticides_path = Path(self.data_path) / 'pesticides.csv'
        rainfall_path = Path(self.data_path) / 'rainfall.csv'
        temp_path = Path(self.data_path) / 'temp.csv'
        yield_path = Path(self.data_path) / 'yield.csv'

        self.pesticides = pd.read_csv(pesticides_path)
        self.rainfall = pd.read_csv(rainfall_path, na_values=['..'])
        self.temp = pd.read_csv(temp_path)
        self.yield_df = pd.read_csv(yield_path)

        self.logger.info("Archivos leídos exitosamente")


    def rename_columns(self) -> None:
        """Renombra columnas para estandarizar nombres"""
        self.logger.info("Renombrando columnas...")
        # Renombrar columnas de rainfall
        original_cols_rainfall = self.rainfall.columns.tolist()
        self.rainfall = self.rainfall.rename(columns={
            ' Area': 'Area', 
            "average_rain_fall_mm_per_year": "Value"
        })
        self.logger.info(f"Rainfall columns: {original_cols_rainfall} -> {self.rainfall.columns.tolist()}")
        
        # Renombrar columnas de temperatura
        original_cols_temp = self.temp.columns.tolist()
        self.temp = self.temp.rename(columns={
            "country": "Area", 
            "year": "Year", 
            'avg_temp': 'Value'
        })
        self.logger.info(f"Temperature columns: {original_cols_temp} -> {self.temp.columns.tolist()}")

    def select_columns(self)-> None:
        """Selecciona solo las columnas necesarias"""
        self.logger.info("Seleccionando columnas relevantes...")

        columns_to_have = ['Area', 'Year', 'Value']
        self.pesticides = self.pesticides[columns_to_have]
        self.rainfall   = self.rainfall[columns_to_have]
        self.temp       = self.temp[columns_to_have]
        self.yield_df   = self.yield_df[columns_to_have + ['Item']]

    def remove_duplicates(self):
        """Remueve duplicados tomando la media por grupo"""
        self.logger.info('Removiendo duplicados')
        self.temp = self.temp.groupby(['Area', 'Year']).mean().reset_index()
        self.rainfall = self.rainfall.groupby(['Area', 'Year']).mean().reset_index()
        self.pesticides = self.pesticides.groupby(['Area', 'Year']).mean().reset_index()
        self.yield_df = self.yield_df.groupby(['Area', 'Item', 'Year']).mean().reset_index()

        
    def generate_data_dict(self)-> None:
        """Genera diccionario con los DataFrames procesados"""
        self.logger.info("Generando diccionario de datos...")

        self.dfs_dict = {
            'rainfall': self.rainfall,
            'pesticides': self.pesticides,
            'temperature': self.temp      
        }

    def find_best_fuzzy_matches(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                              field_na: str) -> Tuple[Dict[str, str], List[str]]:
        """
        Encuentra coincidencias fuzzy entre países de dos DataFrames
        
        Args:
            df1: DataFrame base
            df2: DataFrame con datos a matchear
            field_na: Campo con valores nulos a matchear
            
        Returns:
            Tupla con (asignaciones, países no encontrados)
        """
        self.logger.info(f"Iniciando fuzzy matching para campo: {field_na}")
        lista_no_found = list(df1[df1[field_na].isna()].Area.unique())
        list_area_df2 = list(df2.Area.unique())

        
        self.logger.info(f"Países sin datos en {field_na}: {len(lista_no_found)}")
        self.logger.info(f"Países disponibles en df2: {len(list_area_df2)}")

        def normalizar_nombre_pais(nombre):
            """Normaliza nombres de países para mejor matching"""
            nombre = nombre.strip().lower()
            nombre = re.sub(r'[^\w\s]', ' ', nombre)
            nombre = ' '.join(nombre.split())

            reemplazos = {
                'united states of america': 'united states',
                'united states': 'usa',
                'united kingdom': 'uk',
                'republic of korea': 'south korea',
                'korea, republic of': 'south korea',
                'china mainland': 'china',
                "venezuela bolivarian republic of": "venezuela",
                "bolivia plurinational state of": "bolivia",
                "democratic people's republic of korea": "korea",
                "united republic of tanzania": "tanzania",
                ' republic': '',
                ' federation': '',
                ' democratic': '',
                ' peoples': '',
                ' socialist': '',
                'republic of ': '',
                'democratic republic of ': '',
                ' rep ': ' republic ',
                ' dem ': ' democratic ',
                ' fed ': ' federation '}
            
            for buscar, reemplazar in reemplazos.items():
                    nombre = nombre.replace(buscar, reemplazar)
            nombre = ' '.join(nombre.split())
            return nombre
        
        lista_df1_norm = {pais: normalizar_nombre_pais(pais) for pais in lista_no_found}
        lista_df2_norm = {pais: normalizar_nombre_pais(pais) for pais in list_area_df2}

        asignaciones = {}
        no_encontrados  = []

        for pais_df1_orig, pais_df1_norm in lista_df1_norm.items():
            mejor_match = process.extractOne(
                pais_df1_norm, 
                list(lista_df2_norm.values()), 
                scorer=fuzz.ratio
            )
            if mejor_match and mejor_match[1] >= 80:
                # Encontrar el país original correspondiente
                pais_df2_match = [k for k, v in lista_df2_norm.items() if v == mejor_match[0]][0]
                asignaciones[pais_df1_orig] = pais_df2_match
                self.logger.debug(f"Match encontrado: {pais_df1_orig} -> {pais_df2_match} (score: {mejor_match[1]})")

            else:
                no_encontrados.append(pais_df1_orig)

        self.logger.info(f"Fuzzy matching completado. Matches: {len(asignaciones)}, No encontrados: {len(no_encontrados)}")

        return asignaciones, no_encontrados

    def rename_areas_using_fuzzy_matching(self, df: pd.DataFrame, 
                                        asignaciones: Dict[str, str]) -> pd.DataFrame:
        """
        Renombra áreas usando las asignaciones del fuzzy matching
        
        Args:
            df: DataFrame a procesar
            asignaciones: Diccionario de asignaciones
            
        Returns:
            DataFrame con áreas renombradas
        """
        # Removemos algunos paises que no son correctos. Sacamos China porque en temp esta China y CHina mainland a la vez
        to_drop = ["Nigeria","Iceland", "Gambia", "Czechia", "Dominican Republic", "China", "Dominica"] 
        asignaciones_invertidas = {v: k for k, v in asignaciones.items()}
        for pais in to_drop:
            asignaciones_invertidas.pop(pais, None)
        df_renamed = df.copy()
        df_renamed["Area"] = df_renamed.Area.apply(lambda x: asignaciones_invertidas.get(x, x))
        return df

    def join_dfs(self, df1: pd.DataFrame, df2: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """Une dos DataFrames por Area y Year"""

        result = df1.merge(df2, on=['Area', 'Year'], how='left', suffixes=('', suffix))
        self.logger.info(f"DataFrames unidos. Shape resultante: {result.shape}")
        return result


    def filter_by_years(self, df: pd.DataFrame, min_year: int, max_year: int) -> pd.DataFrame:
        """Filtra DataFrame por rango de años"""
        df_filtered = df[df.Year >= min_year]
        self.logger.info(f"Filtrando por años: {min_year} <= year <= {max_year}")
        df_filtered = df_filtered[df_filtered.Year <= max_year] 
        self.logger.info(f"Después del filtrado: {df_filtered.shape[0]} filas")

        return df_filtered

    def fill_na_with_last_n_means(self, df: pd.DataFrame, col: str, n: int = 5) -> pd.DataFrame:
        """Rellena valores nulos con media móvil de últimos n valores"""

        result = df.copy()
        rolling_mean = df[col].rolling(
            window = n,
            min_periods = 1
        ).mean()

        result[col] = df[col].fillna(rolling_mean)
        return result
              
    def process_field(self, df1: pd.DataFrame, df2: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """
        Procesa un campo específico uniendo DataFrames y aplicando limpieza
        
        Args:
            df1: DataFrame base
            df2: DataFrame a unir
            suffix: Sufijo para las columnas del segundo DataFrame
            
        Returns:
            DataFrame procesado
        """
        self.logger.info(f"Procesando campo con subfijo: {suffix}")
        
        try:
            self.logger.info(f"Iniciando process_field con suffix: {suffix}")
            self.logger.info(f"Filas df1: {len(df1)}, Filas df2: {len(df2)}")
            
            # Asegurar que df2 no tenga duplicados Area-Year
            df2_clean = df2.groupby(['Area', 'Year'])['Value'].mean().reset_index()
            self.logger.info(f"df2 después de limpiar duplicados: {len(df2_clean)}")
            
            df = self.join_dfs(df1, df2_clean, suffix=suffix)
            self.logger.info(f"Filas después del merge: {len(df)}")
            
            df = self.fill_na_with_last_n_means(df, col=f'Value{suffix}', n=10)
            df = df.drop_duplicates()
            self.logger.info(f"Filas después de drop_duplicates: {len(df)}")
            
            min_year = np.max([df1.Year.min(), df2.Year.min()])
            max_year = df1.Year.max()
            
            df = self.filter_by_years(df, min_year=min_year, max_year=max_year)
            self.logger.info(f"Filas después del filtro de años: {len(df)}")
            
            df = self.fill_na_with_last_n_means(df, col=f'Value{suffix}', n=5)
            
            df_no_na = df[~df[f'Value{suffix}'].isna()]
            asignaciones, no_encontrados = self.find_best_fuzzy_matches(df, df2_clean, f"Value{suffix}")
            
            self.logger.info(f'Países encontrados: {len(asignaciones)}')
            self.logger.info(f'Países no encontrados: {len(no_encontrados)}')
            
            if asignaciones:
                df_na_areas_renamed = self.rename_areas_using_fuzzy_matching(
                    df2_clean[df2_clean.Area.isin(asignaciones.keys())], 
                    asignaciones
                )
                
                df_na_areas_renamed_rejoined = df.drop(columns=[f'Value{suffix}'])[
                    df.Area.isin(asignaciones.values())
                ].merge(df_na_areas_renamed, on=["Area", "Year"], suffixes=["", suffix])
                
                if len(df_na_areas_renamed_rejoined) > 0:
                    df_areas_renamed = pd.concat([df_no_na, df_na_areas_renamed_rejoined])
                    df_areas_renamed = df_areas_renamed.drop_duplicates()  # Asegurar no duplicados
                else:
                    df_areas_renamed = df_no_na
            else:
                df_areas_renamed = df_no_na
                
            df_areas_renamed[f'Value{suffix}'] = df_areas_renamed.groupby("Year")[f'Value{suffix}'].transform(
                lambda x: x.fillna(x.median())
            )
            self.logger.info(f'Filas finales: {len(df_areas_renamed)}')
            #self.logger.info('Years considered:[', df_areas_renamed.Year.min(), ',', df_areas_renamed.Year.max(), ']')
            return df_areas_renamed
        
        except Exception as e:
            self.logger.error(f"Error al procesar campo {suffix}: {str(e)}")
            raise


    def process_all_df(self, df: pd.DataFrame, dfs_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Procesa todos los DataFrames y genera el resultado final
        
        Args:
            df: DataFrame base (yield)
            dfs_dict: Diccionario con DataFrames a procesar
        """
        required_keys = ['rainfall', 'pesticides', 'temperature']
        missing_keys = [key for key in required_keys if key not in dfs_dict]
        if missing_keys:
            raise KeyError(f"dfs_dict debe contener las claves: {missing_keys}")
        self.logger.info("Iniciando procesamiento de todos los DataFrames...")

        try:
            # Procesamiento de rainfall
            self.logger.info('----Processing Rainfall df---')
            df_rainfall = dfs_dict['rainfall']
            df_final = self.process_field(df, df_rainfall, '_rainfall')

            # Procesamiento de pesticides
            self.logger.info('----Processing Pesticides df---')
            df_pesticides = dfs_dict['pesticides']
            df_final = self.process_field(df_final, df_pesticides, '_pesticides')

            # Procesamiento de temperature
            self.logger.info('----Processing Temperature df---')
            df_temp = dfs_dict['temperature']
            df_final = self.process_field(df_final, df_temp, '_temperature')

            self.df_final = df_final
            self.logger.info(f"Procesamiento completado. Shape final: {self.df_final.shape}")

        except Exception as e:
                self.logger.error(f"Error en procesamiento general: {str(e)}")
                raise
    

    def save_data(self) -> None:
        """Guarda los datos procesados"""
        try:
            output_path = Path(self.data_path) / 'data_processed.csv'
            self.df_final.to_csv(output_path, index=False)
            self.logger.info(f"Datos guardados exitosamente en: {output_path}")
            self.logger.info(f"Archivo final: {self.df_final.shape[0]} filas, {self.df_final.shape[1]} columnas")
            
        except Exception as e:
            self.logger.error(f"Error al guardar datos: {str(e)}")
            raise self.df_final.to_csv(self.data_path + '/data_processed,csv')
