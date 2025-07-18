import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import os
import uuid
import logging
import pickle
import base64
from datetime import datetime

from app.logic.datos import codificar_datos

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCacheManager:
    """Gestor de caché para modelos entrenados con compatibilidad para versiones"""
    
    def __init__(self, cache_dir=None):
        try:
            # Configuración de rutas ABSOLUTAS
            if cache_dir is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.cache_dir = os.path.join(base_dir, 'data_cache')
            else:
                self.cache_dir = os.path.abspath(cache_dir)
            
            logger.info(f"Ruta de caché configurada: {self.cache_dir}")
            os.makedirs(self.cache_dir, exist_ok=True)
            self.current_model_path = self._find_latest_model()
            
        except Exception as e:
            logger.error(f"Error inicializando ModelCacheManager: {str(e)}")
            raise

    def _find_latest_model(self):
        """Busca el modelo más reciente (sin borrar otros)"""
        try:
            if not os.path.exists(self.cache_dir):
                return None
                
            model_files = [f for f in os.listdir(self.cache_dir) 
                         if f.endswith('.pkl') and f.startswith(('entrenamiento_', 'modelo_'))]
            
            if not model_files:
                return None
                
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.cache_dir, x)), reverse=True)
            latest_model = os.path.join(self.cache_dir, model_files[0])
            
            if os.path.getsize(latest_model) > 0:
                return latest_model
            return None
            
        except Exception as e:
            logger.error(f"Error buscando modelo: {str(e)}")
            return None

    def has_model(self):
        """Verifica si hay modelos disponibles"""
        return self._find_latest_model() is not None

    def load_model(self):
        """Carga modelos nuevos (v2.0) y antiguos (v1.0)"""
        if not self.current_model_path:
            self.current_model_path = self._find_latest_model()
            if not self.current_model_path:
                raise ValueError("No hay modelos válidos en caché")
        
        try:
            with open(self.current_model_path, 'rb') as f:
                model_data = pickle.load(f)
            
             # Debug: Inspeccionar estructura
            print("\n[DEBUG] Contenido del modelo:")
            for key, value in model_data.items():
             print(f"{key}: {type(value)}")
            # Detección automática de versión
            if 'version' in model_data:  # Modelo nuevo (v2.0+)
                required_keys = ['model', 'scaler', 'features', 'metrics', 'training_date']
                if all(key in model_data for key in required_keys):
                    logger.info("Modelo nuevo (v2.0) cargado")
                    return model_data
            
            # Modelo antiguo (v1.0)
            elif all(key in model_data for key in ['model', 'scaler', 'features']):
                logger.info("Modelo antiguo (v1.0) detectado - Convirtiendo...")
                model = model_data['model']
                scaler = model_data['scaler']
                
                if isinstance(model, str):  # Si está serializado en base64
                    model = pickle.loads(base64.b64decode(model.encode('utf-8')))
                    scaler = pickle.loads(base64.b64decode(model_data['scaler'].encode('utf-8')))
                
                return {
                    'model': model,
                    'scaler': scaler,
                    'features': model_data['features'],
                    'metrics': model_data.get('metrics', {}),
                    'training_date': model_data.get('training_date', datetime.now().isoformat()),
                    'version': '1.0'
                }
            
            raise ValueError("Formato de modelo desconocido")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise

    def save_model(self, trained_model):
        """Guarda el modelo asegurando la estructura correcta"""
        required_keys = ['model', 'scaler', 'features', 'metrics', 'training_date']
        missing_keys = [k for k in required_keys if k not in trained_model]
        if missing_keys:
            raise ValueError(f"Faltan claves en el modelo: {missing_keys}")

        try:
            model_path = os.path.join(self.cache_dir, f"modelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(trained_model, f)
            logger.info(f"Modelo guardado en: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error al guardar: {str(e)}")
            raise

def train_model(training_data, cache_manager=None):
    """
    Entrena un modelo de regresión logística y lo guarda automáticamente
    Args:
        training_data (pd.DataFrame): Datos de entrenamiento
        cache_manager (ModelCacheManager): Opcional. Si no se proporciona, se crea uno nuevo
    Returns:
        dict: Modelo entrenado y componentes
    """
    try:
        logger.info("Iniciando entrenamiento del modelo")

        # 1. Preprocesamiento
        df_encoded = codificar_datos(training_data, es_prediccion=False)
        X = df_encoded.drop(columns=['abandono', 'nombre'], errors='ignore')
        y = df_encoded['abandono']
        features = list(X.columns)

        # 2. Escalado
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. Entrenamiento
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs'
        )

        # 4. Validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')

        # 5. Entrenamiento final
        model.fit(X_scaled, y)

        # 6. Métricas
        y_pred = model.predict(X_scaled)
        metrics = {
            'accuracy': round(accuracy_score(y, y_pred), 3),
            'precision': round(precision_score(y, y_pred), 3),
            'recall': round(recall_score(y, y_pred), 3),
            'f1': round(f1_score(y, y_pred), 3),
            'cv_accuracy': [round(score, 3) for score in cv_scores],
            'cv_mean_accuracy': round(cv_scores.mean(), 3)
        }

        # Crear estructura del modelo
        trained_model = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metrics': metrics,
            'training_date': datetime.now().isoformat(),
            'version': '2.0'
        }

        logger.info("Modelo entrenado con éxito")
        
        # Guardar usando el cache_manager
        if cache_manager is None:
            cache_manager = ModelCacheManager()
        
        cache_manager.save_model(trained_model)
        logger.info("Modelo entrenado y guardado correctamente")
        
        return trained_model

    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}")
        raise

def predict(model_data, input_data):
    """
    Realiza predicciones con el modelo
    Args:
        model_data (dict): Modelo entrenado
        input_data (pd.DataFrame): Datos para predecir
    Returns:
        dict: Resultados y métricas
    """
    try:
        # Validación de estructura
        required = ['model', 'scaler', 'features']
        if not all(key in model_data for key in required):
            raise ValueError(f"Faltan componentes: {required}")
        
        # Preprocesamiento
        df_encoded = codificar_datos(input_data, es_prediccion=True)
        X = df_encoded.drop(columns=['nombre'], errors='ignore')
        
        # Dentro de predict(), reemplaza esta parte:
        if list(X.columns) != model_data['features']:
            # Mensaje detallado
            missing = set(model_data['features']) - set(X.columns)
            extra = set(X.columns) - set(model_data['features'])
            error_msg = f"""
            Error de coincidencia en features:
            - Faltan: {list(missing) if missing else 'Ninguna'}
            - Sobran: {list(extra) if extra else 'Ninguna'}
            - Esperadas: {model_data['features']}
            - Recibidas: {list(X.columns)}
            """
            raise ValueError(error_msg)
        
        # Predicción
        X_scaled = model_data['scaler'].transform(X)
        predictions = model_data['model'].predict(X_scaled)
        probabilities = model_data['model'].predict_proba(X_scaled)[:, 1]

        return {
            'success': True,
            'predictions': df_encoded.assign(
                prediccion=predictions,
                probabilidad=probabilities
            ).to_dict('records'),
            'metrics': model_data.get('metrics', {}),
            'prediction_date': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'debug_info': {
                'model_keys': list(model_data.keys()) if isinstance(model_data, dict) else str(type(model_data)),
                'input_columns': list(input_data.columns) if hasattr(input_data, 'columns') else None
            }
        }


# Función auxiliar (debes implementarla)
def codificar_datos(df, es_prediccion=False):
    """
    Preprocesamiento robusto para datos de deserción estudiantil:
    - Limpieza de NaN
    - Codificación de variables categóricas
    - Conversión de tipos
    """
    df = df.copy()
    
    # 1. Limpieza de valores faltantes
    # Opción 1: Eliminar filas con NaN (si tienes suficientes datos)
    df = df.dropna()
    
     # 1. Normalizar nombres de columnas
    column_mapping = {
        'Te sientes motivadoa a seguir estudiando': 'motivacion',
        'Asistencia promedio en % asistencia a clases': 'asistencia',
        'Te sientes estresadoa con tus estudios actualmente': 'estres',
        'Tienes acceso a recursos escolares internet libros computadora': 'acceso_recursos'
    }
    df = df.rename(columns=column_mapping)
    
    # 2. Codificar variables categóricas (ej: sexo, nivel_escolar)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col != 'nombre':  # Excluir columna de nombres
            df[col] = pd.factorize(df[col])[0]  # Convertir a códigos numéricos
    
    # 3. Convertir la variable objetivo (si es entrenamiento)
    if not es_prediccion and 'abandono' in df.columns:
        df['abandono'] = df['abandono'].astype(int)
    
    # Debug: Verificar que no queden NaN o tipos incorrectos
    print("\n✅ Debug - Resumen post-procesamiento:")
    print(f"- NaN restantes: {df.isnull().sum().sum()}")
    print(f"- Tipos de datos:\n{df.dtypes}")
    print(f"- Columnas categóricas convertidas: {list(cat_cols)}")
    
    return df

if __name__ == "__main__":
    """Ejemplo de uso"""
    try:
        # 1. Inicializar
        cache = ModelCacheManager()
        
        # 2. Simular datos
        train_data = pd.DataFrame({
            'edad': [25, 30, 35, 40],
            'promedio': [80, 85, 75, 90],
            'abandono': [0, 1, 0, 1]
        })
        
        # 3. Entrenar y guardar
        modelo = train_model(train_data)
        cache.save_model(modelo)
        
        # 4. Cargar y predecir
        modelo_cargado = cache.load_model()
        test_data = pd.DataFrame({
            'edad': [28, 33],
            'promedio': [82, 88]
        })
        
        resultados = predict(modelo_cargado, test_data)
        print("Predicción exitosa:", resultados['success'])
        
    except Exception as e:
        print(f"Error en pruebas: {str(e)}")