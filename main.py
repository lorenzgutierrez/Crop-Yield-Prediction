import sys
sys.path.append('src')
from config import get_settings
from data_pipeline import Preprocesser
from model import Models
from feature_engineering import FeatureGenerator
import logging


def setup_logging():
    """Configura logging básico"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_pipeline():
    """Ejecuta el pipeline completo"""
    
    # Configurar logging
    logger = setup_logging()
    logger.info(f"🌾 Iniciando pipeline de predicción de cultivos")
    
    # Obtener configuración
    settings = get_settings()
    logger.info(f"📁 Ruta de datos: {settings['data_path']}")
    # 1. PIPELINE DE DATOS
    logger.info("📥 Iniciando procesamiento de datos...")
    preprocesser = Preprocesser(settings=settings)
    preprocesser.read_data()
    preprocesser.rename_columns()
    preprocesser.select_columns()
    preprocesser.remove_duplicates()
    preprocesser.generate_data_dict()
    preprocesser.process_all_df(preprocesser.yield_df, preprocesser.dfs_dict)
    
    df = preprocesser.df_final
    preprocesser.save_data()
    logger.info("✅ Datos procesados exitosamente")
    
    # 2. GENERACIÓN DE CARACTERÍSTICAS
    logger.info("🔧 Generando características...")
    feat_generator = FeatureGenerator(df, settings)
    feat_generator.generate_features()
    feat_generator.save_data()
    
    df = feat_generator.df
    encoders_dict = feat_generator.encoders
    logger.info("✅ Características generadas exitosamente")
    
    # 3. ENTRENAMIENTO DE MODELOS
    logger.info("🤖 Entrenando modelos...")
    model_orchestrator = Models(df, settings)
    model_orchestrator.run_pipeline()
    model_orchestrator.compare_model_results()
    model_orchestrator.set_encoders(encoders_dict)
    model_orchestrator.create_global_crop_production_plots()
    model_orchestrator.save_data()
    logger.info("✅ Modelos entrenados exitosamente")

run_pipeline()
