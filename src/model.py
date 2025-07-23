import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import logging
import joblib
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

class Models(object):
    """
    Clase para entrenamiento, evaluación y visualización de modelos de predicción de cultivos
    """
    def __init__(self, df, settings):
        """
        Inicializa la clase Models con logging y configuración
        
        Args:
            df: DataFrame con los datos para entrenamiento
            settings: Diccionario con configuraciones del proyecto
        """
               # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('models.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.df = df.copy()
        self.settings = settings
        self.meta_info = None
        self.X = None
        self.y = None
        self.comparison_df = None
        self.results = None
        self.scaler = None
        self.encoders = None
        self.train_mask = None
        self.test_mask = None
        self.val_mask = None
        self.logger.info(f"Models inicializado con DataFrame de shape: {self.df.shape}")
        self.logger.info(f"Columnas disponibles: {list(self.df.columns)}")
        # Suprimir warnings de sklearn
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

    def model_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, train: bool = True) -> Dict[str, float]:
        """
        Calcula métricas de evaluación del modelo
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            train: Si es True, agrega prefijo 'train_', sino 'test_'
            
        Returns:
            Diccionario con métricas calculadas
        """
        prefix = 'train_' if train else 'test_'

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true-y_pred)/(y_true+ 1e-8)))*100
        bias = np.mean(y_pred - y_true)
        metrics = {
            f'{prefix}MAE':mae,
            f'{prefix}RMSE': rmse,
            f'{prefix}R2': r2,
            f'{prefix}MAPE': mape,
            f'{prefix}Bias': bias
        }
        return metrics
    
    def train_test_split_year(self, df, test_years, validation_years):
        df = self.df.copy()
        df = df.sort_values(['Area', 'Item', 'Year'])
        years = sorted(df.Year.unique())
        max_year = max(years)
        min_year = min(years) + 3 # Por lags y otros features que vienen de tiempos

        test_start = max_year - test_years +1
        val_start = test_start - validation_years

        train_mask = (df.Year < val_start) & (df.Year >= min_year)
        val_mask = (df.Year >= val_start) & (df.Year < test_start)
        test_mask = df.Year >= test_start
        # Log de información sobre la división
        train_count = train_mask.sum()
        val_count = val_mask.sum()
        test_count = test_mask.sum()
        
        self.logger.info(f"División completada:")
        self.logger.info(f"  - Entrenamiento: {train_count:,} registros ({min_year}-{val_start-1})")
        self.logger.info(f"  - Validación: {val_count:,} registros ({val_start}-{test_start-1})")
        self.logger.info(f"  - Test: {test_count:,} registros ({test_start}-{max_year})")

        self.train_mask = train_mask
        self.val_mask   = val_mask
        self.test_mask  = test_mask
        return train_mask, val_mask, test_mask
    
    
    def preprocess_features(self, target_col: str = 'Value') -> None:
        """
        Preprocesa las características eliminando valores infinitos
        
        Args:
            target_col: Nombre de la columna objetivo
        """
        self.logger.info("Iniciando preprocesamiento de características...")

        df = self.df.copy()

        initial_shape = self.df.shape
        exclude_cols = [target_col, 'Year']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Reemplazar infinitos con NaN
        inf_count = np.isinf(self.df[numeric_cols]).sum().sum()
        if inf_count > 0:
            self.logger.warning(f"Se encontraron {inf_count} valores infinitos, reemplazando con NaN")
            self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        final_shape = self.df.shape
        self.logger.info(f"Preprocesamiento completado: {initial_shape} -> {final_shape}")
    
    def prepare_model_data(self, df: pd.DataFrame) -> None:
        """
        Prepara los datos para entrenamiento separando características y objetivo
        
        Args:
            df: DataFrame con los datos
        """
        self.logger.info("Preparando datos para modelado...")
        df = self.df.copy()
        feature_cols = [col for col in df.columns if not col in ['Value', 'Year', 'Area', 'Item']]

        self.logger.info(f"Características seleccionadas: {len(feature_cols)} columnas")
        self.logger.debug(f"Lista de características: {feature_cols}")   

        X = df.copy()
        self.y = df['Value'].copy()
        null_counts = X.isnull().sum()
        null_cols = null_counts[null_counts > 0]

        if len(null_cols) > 0:
            self.logger.warning(f"Columnas con valores faltantes:")
            for col, count in null_cols.items():
                pct = (count / len(X)) * 100
                self.logger.warning(f"  - {col}: {count:,} valores ({pct:.1f}%)")

        self.meta_info = X[['Year', 'Area', 'Item']].copy()

        X = X[feature_cols].copy()
        print(X.head())
        self.X = X.fillna(X.median())
        self.logger.info(f"Datos preparados: X shape={self.X.shape}, y shape={self.y.shape}")

    def hyperparam_tuning_random(self, X_train, y_train):
        """Random search hiperparameter tunning"""
        self.logger.info("Iniciando random search...")

        params = {
            'n_estimators': [10, 50,100,200,300],
            'max_depth': [5,15,25,None],
            'min_samples_split': [2, 5, 10 , 20],
            'max_features': ['sqrt', 'log2', None, 0.7]
        }

        rf = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            rf,
            param_distributions= params,
            cv = 3,
            scoring= 'r2',
            random_state= 42
        )
        random_search.fit(X_train, y_train)
        self.logger.info(f"Mejores parámetros: {random_search.best_params_}")
        self.logger.info(f"Mejor score: {random_search.best_score_:.3f}")
        return random_search.best_estimator_


    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series, tune_rf: bool = False) -> None:
        """
        Entrena múltiples modelos y evalúa su rendimiento
        
        Args:
            X_train: Características de entrenamiento
            X_test: Características de test
            y_train: Objetivo de entrenamiento
            y_test: Objetivo de test
        """
        self.logger.info("Iniciando entrenamiento de modelos...")

        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha = .5),
            'Lasso': Lasso(alpha = .5),
        }

        if tune_rf:
            self.logger.info("🔧 Aplicando hyperparameter tuning a Random Forest...")

            rf_tuned = self.hyperparam_tuning_random(X_train, y_train)
            models['Random Forest'] = rf_tuned
        else:
            models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, 
            random_state=42
            )
        self.logger.info(f"Modelos a entrenar: {list(models.keys())}")

        scaler = RobustScaler()  # Mejor para datos con outliers
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        results = {}
        self.logger.info("Escalado de características completado")

        for name, model in models.items():
            self.logger.info(f"Entrenando modelo: {name}")
            start_time = datetime.now()

            if name in ['Linear Regression', 'Ridge', 'Lasso']:
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

            train_metrics = self.model_metrics(y_train, y_pred_train, train= True)
            test_metrics  = self.model_metrics(y_test, y_pred_test, train= False)

            ## HECHO CON IA (un poco)
            # Combinar métricas

            all_metrics = {**train_metrics, **test_metrics}
            
            
            if name in ['Linear Regression', 'Ridge', 'Lasso']:
                results[name] = {
                    'model': model,
                    'metrics': all_metrics,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test,
                    'X_train': X_train_scaled,
                    'X_test': X_test_scaled,
                    'y_train': y_train,
                    'y_test': y_test

                }
            else:
                results[name] = {
                    'model': model,
                    'metrics': all_metrics,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }
            training_time = (datetime.now() - start_time).total_seconds()
                
            self.logger.info(f"  ✅ {name} completado en {training_time:.2f}s")
            self.logger.info(f"     Train R²: {train_metrics['train_R2']:.3f} | Test R²: {test_metrics['test_R2']:.3f}")
            self.logger.info(f"     Test RMSE: {test_metrics['test_RMSE']:.2f} | Test MAPE: {test_metrics['test_MAPE']:.1f}%")
        self.results = results
        self.scaler = scaler
        self.logger.info(f"Entrenamiento completado. {len(results)} modelos exitosos.")

    def run_pipeline(self) -> None:
        """
        Ejecuta el pipeline completo de modelado
        """
        self.logger.info("=== INICIANDO PIPELINE DE MODELADO ===")
        start_time = datetime.now()
        df = self.df.copy()
        self.prepare_model_data(df)

        X = self.X
        y = self.y
        meta_info = self.meta_info

        train_mask, val_mask, test_mask = self.train_test_split_year(
            df, 
                test_years=5, 
                validation_years=5
            )
        # Sin validacion por ahora
        train_mask = train_mask | val_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        self.train_models(X_train, X_test, y_train, y_test, tune_rf= self.settings['tune_rf'])
        total_time = (datetime.now() - start_time).total_seconds()

        self.logger.info(f"=== PIPELINE COMPLETADO EN {total_time:.2f}s ===")

        #return results, scaler, (X_train, X_test, y_train, y_test), meta_info, (train_mask, test_mask)
    
    def compare_model_results(self) -> None:
        """
        Compara los resultados de todos los modelos entrenados
        """
        self.logger.info("Comparando resultados de modelos...")
        comparison_df = []

        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_df.append({
                'Model': model_name,
                'Train_R2': metrics['train_R2'],
                'Test_R2': metrics['test_R2'],
                'Test_RMSE': metrics['test_RMSE'],
                'Test_MAPE': metrics['test_MAPE'],
                'Bias_pct': metrics['test_Bias'],
            })
        self.comparison_df = pd.DataFrame(comparison_df).sort_values('Test_R2', ascending=False)
        self.logger.info("=== COMPARACIÓN DE MODELOS ===")
        self.logger.info("\n" + self.comparison_df.to_string(index=False))
        best_model = self.comparison_df.iloc[0]['Model']
        best_r2 = self.comparison_df.iloc[0]['Test_R2']
        self.logger.info(f"\n🏆 Mejor modelo: {best_model} (Test R² = {best_r2:.3f})")

    def  save_data(self) -> None:
        """
        Guarda los resultados del modelo y comparaciones
        """
        self.logger.info("Guardando resultados del modelo...")
        output_path = Path(self.settings['models_path'])

        comparison_file = output_path / 'model_comparison.csv'
        self.comparison_df.to_csv(comparison_file, index=False)
        self.logger.info(f"Comparación guardada en: {comparison_file}")

        best_model_name = self.comparison_df.iloc[0]['Model']
        best_model = self.results[best_model_name]['model']
        
        model_file = output_path / 'best_model.pkl'
        joblib.dump(best_model, model_file)
        self.logger.info(f"Mejor modelo ({best_model_name}) guardado en: {model_file}")
        metadata = {
                    'model_name': best_model_name,
                    'feature_columns': list(self.X.columns) if self.X is not None else None,
                    'encoders': self.encoders,
                    'scaler': self.scaler,
                    'metrics': self.results[best_model_name]['metrics'],
                    'training_date': datetime.now().isoformat()
                }
                
        metadata_file = output_path / 'model_metadata.pkl'
        joblib.dump(metadata, metadata_file)
        self.logger.info(f"Metadatos guardados en: {metadata_file}")

    def set_encoders(self, encoders: Dict) -> None:
            """
            Establece los encoders para decodificación en visualizaciones
            
            Args:
                encoders: Diccionario con encoders de características categóricas
            """
            self.encoders = encoders
            self.logger.info(f"Encoders establecidos: {list(encoders.keys())}")

    def create_global_crop_production_plots(self, model_name: str = 'Random Forest', 
                                          top_crops: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Crea gráficos de producción global agregada por cultivo y año.
        FUNCION HECHA ENTERAMENTE CON IA
        
        Args:
            model_name: Nombre del modelo a usar para las predicciones
            top_crops: Lista de cultivos a mostrar. Si None, usa los 10 con más datos
            
        Returns:
            DataFrame con datos agregados
        """
        self.logger.info(f"Creando gráficos de producción global para modelo: {model_name}")
        
        if not self.results or model_name not in self.results:
            self.logger.error(f"Modelo {model_name} no encontrado en resultados")
            return pd.DataFrame()
        
        if self.encoders is None:
            self.logger.warning("No hay encoders disponibles. Asumiendo datos sin codificar.")
            self.encoders = {}
        
        try:
            X_train = self.results[model_name]['X_train']
            X_test = self.results[model_name]['X_test']
            y_train = self.results[model_name]['y_train']
            y_test = self.results[model_name]['y_test']
            
            # Usar las máscaras originales combinadas
            train_mask_combined = self.train_mask | self.val_mask
            test_mask = self.test_mask
            
            train_meta = self.meta_info[train_mask_combined].copy()
            test_meta = self.meta_info[test_mask].copy()
            
            # Verificar longitudes
            assert len(train_meta) == len(X_train), f"Train meta length {len(train_meta)} != X_train length {len(X_train)}"
            assert len(test_meta) == len(X_test), f"Test meta length {len(test_meta)} != X_test length {len(X_test)}"
            
            # Asignar datos
            train_meta['actual'] = y_train.values
            train_meta['type'] = 'train'
            train_meta['predicted'] = np.nan
            
            test_meta['actual'] = y_test.values
            test_meta['type'] = 'test'
            test_meta['predicted'] = self.results[model_name]['y_pred_test']
            
            full_data = pd.concat([train_meta, test_meta], ignore_index=True)
            
            # Decodificar Item si está encoded
            if 'Item' in self.encoders and 'Item' in full_data.columns:
                item_codes = full_data['Item'].astype(int)
                full_data['Item_decoded'] = self.encoders['Item'].inverse_transform(item_codes)
                item_col = 'Item_decoded'
            else:
                item_col = 'Item'
            
            # Decodificar Area si está encoded
            if 'Area' in self.encoders and 'Area' in full_data.columns:
                area_codes = full_data['Area'].astype(int)
                full_data['Area_decoded'] = self.encoders['Area'].inverse_transform(area_codes)
                area_col = 'Area_decoded'
            else:
                area_col = 'Area'
            
            # Seleccionar cultivos a mostrar
            if top_crops is None:
                crop_test_counts = full_data[full_data['type'] == 'test'][item_col].value_counts()
                top_crops = crop_test_counts.head(10).index.tolist()
            
            self.logger.info(f"Analizando cultivos: {top_crops}")
            
            # Agregar datos por cultivo y año
            aggregated_data = []
            
            for crop in top_crops:
                crop_data = full_data[full_data[item_col] == crop].copy()
                
                for data_type in ['train', 'test']:
                    type_data = crop_data[crop_data['type'] == data_type]
                    
                    if len(type_data) > 0:
                        yearly_sums = type_data.groupby('Year').agg({
                            'actual': 'sum',
                            'predicted': 'sum' if data_type == 'test' else lambda x: np.nan
                        }).reset_index()
                        
                        yearly_sums['crop'] = crop
                        yearly_sums['type'] = data_type
                        yearly_sums['countries_count'] = type_data.groupby('Year')[area_col].nunique().values
                        
                        aggregated_data.append(yearly_sums)
            
            agg_df = pd.concat(aggregated_data, ignore_index=True)
            
            # Crear plots
            n_crops = len(top_crops)
            n_cols = min(3, n_crops)
            n_rows = (n_crops + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
            if n_crops == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, crop in enumerate(top_crops):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                crop_agg = agg_df[agg_df['crop'] == crop].copy()
                train_agg = crop_agg[crop_agg['type'] == 'train'].sort_values('Year')
                test_agg = crop_agg[crop_agg['type'] == 'test'].sort_values('Year')
                
                # Plot datos de entrenamiento
                if len(train_agg) > 0:
                    ax.plot(train_agg['Year'], train_agg['actual'], 
                           'o-', color='blue', linewidth=2, markersize=6, alpha=0.7,
                           label='Train - Global Production')
                
                # Plot datos de test
                if len(test_agg) > 0:
                    ax.plot(test_agg['Year'], test_agg['actual'], 
                           'o-', color='red', linewidth=3, markersize=8,
                           label='Test - Real Global Production')
                    
                    ax.plot(test_agg['Year'], test_agg['predicted'], 
                           '^--', color='orange', linewidth=2.5, markersize=8,
                           label='Test - Predicted Global Production')
                    
                    ax.fill_between(test_agg['Year'], 
                                   test_agg['actual'], 
                                   test_agg['predicted'],
                                   alpha=0.2, color='gray', label='Prediction Error')
                
                # Línea de separación train/test
                if len(test_agg) > 0 and len(train_agg) > 0:
                    split_year = test_agg['Year'].min()
                    ax.axvline(x=split_year, color='gray', linestyle=':', alpha=0.7, linewidth=2,
                              label='Train/Test Split')
                
                ax.set_xlabel('Year', fontweight='bold')
                ax.set_ylabel('Global Production (Sum of All Countries)', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9, loc='best')
                
                # Formatear eje Y
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
                
                # Rotar etiquetas si hay muchos años
                years_range = pd.concat([train_agg['Year'], test_agg['Year']]).unique() if len(train_agg) > 0 and len(test_agg) > 0 else []
                if len(years_range) > 10:
                    ax.tick_params(axis='x', rotation=45)
            
            # Ocultar subplots vacíos
            for i in range(n_crops, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                if n_rows > 1:
                    axes[row, col].set_visible(False)
                elif n_cols > 1:
                    axes[col].set_visible(False)
            
            plt.suptitle(f'Global Production Aggregation by Crop - {model_name}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()

            # savefig
            output_path = Path(self.settings['data_path'])
            fig_path = Path(output_path).parent.parent / 'figuras/predictions_plot.png'
            fig.savefig(fig_path)
            self.logger.info(f'Saving predictions plot with model {model_name}') in {fig_path}
            
            #return agg_df
            
        except Exception as e:
            self.logger.error(f"Error creando gráficos de producción: {e}")
            raise


