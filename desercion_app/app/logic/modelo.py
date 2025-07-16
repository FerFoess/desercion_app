import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import os
import uuid
import logging
import pickle
import base64
from datetime import datetime
import fnmatch

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCacheManager:
    """Gestor de caché para modelos entrenados con todas las correcciones"""
    
    def __init__(self, cache_dir=None):
        try:
            # Configuración de rutas ABSOLUTAS
            if cache_dir is None:
                # Retrocedemos 2 niveles desde el directorio actual (app/logic/) para llegar a DESERCION_APP/
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.cache_dir = os.path.join(base_dir, 'data_cache')
            else:
                self.cache_dir = os.path.abspath(cache_dir)
            
            print(f"\n[DEBUG] Ruta base calculada: {base_dir}")
            print(f"[DEBUG] Ruta completa de caché: {self.cache_dir}")
            print(f"[DEBUG] ¿Existe el directorio? {os.path.exists(self.cache_dir)}")
            
            os.makedirs(self.cache_dir, exist_ok=True)
            self.current_model_path = self._find_latest_model()
            
            if self.current_model_path:
                print(f"[DEBUG] Modelo actual configurado: {self.current_model_path}")
            else:
                print("[WARNING] No se encontraron modelos en caché")
                
        except Exception as e:
            print(f"[ERROR] Error inicializando ModelCacheManager: {str(e)}")
            raise

    def _find_latest_model(self):
        """Busca el modelo más reciente con criterios mejorados"""
        try:
            if not os.path.exists(self.cache_dir):
                print(f"[ERROR] ¡El directorio no existe! {self.cache_dir}")
                return None
                
            # Busca archivos que empiecen con 'entrenamiento_' y terminen en '.pkl'
            model_files = [f for f in os.listdir(self.cache_dir) 
                         if f.endswith('.pkl') and f.startswith('entrenamiento_')]
            print(f"[DEBUG] Archivos .pkl encontrados: {model_files}")
            
            if not model_files:
                return None
                
            # Ordena por fecha de modificación (el más reciente primero)
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.cache_dir, x)), reverse=True)
            latest_model = os.path.join(self.cache_dir, model_files[0])
            
            # Verifica que el archivo no esté vacío
            if os.path.getsize(latest_model) == 0:
                print(f"[WARNING] ¡El modelo {latest_model} está vacío!")
                return None
                
            return latest_model
            
        except Exception as e:
            print(f"[ERROR] Error en _find_latest_model: {str(e)}")
            return None

    def load_model(self):
        """Carga el modelo con validación mejorada y definición correcta de trained_model"""
        if not self.current_model_path:
            self.current_model_path = self._find_latest_model()
            if not self.current_model_path:
                raise ValueError("No hay modelos válidos en el directorio de caché")
        
        print(f"\n[DEBUG] Intentando cargar: {self.current_model_path}")
        print(f"[DEBUG] Tamaño del archivo: {os.path.getsize(self.current_model_path)} bytes")
        
        try:
            with open(self.current_model_path, 'rb') as f:
                model_data = pickle.load(f)
                print("[DEBUG] Claves del modelo cargado:", model_data.keys())
                
            # Validar estructura del modelo
            required_keys = ['model', 'scaler', 'features', 'metrics', 'training_date']
            if not all(key in model_data for key in required_keys):
                raise ValueError("Estructura del modelo en caché inválida")
            
            # Deserializar componentes - ¡Aquí definimos correctamente trained_model!
            trained_model = {
                'metrics': model_data['metrics'],
                'training_date': model_data['training_date'],
                'features': model_data['features'],
                'model': pickle.loads(base64.b64decode(model_data['model'].encode('utf-8'))),
                'scaler': pickle.loads(base64.b64decode(model_data['scaler'].encode('utf-8')))
            }
            
            print("[DEBUG] Modelo deserializado correctamente")
            return trained_model
            
        except Exception as e:
            print(f"[ERROR] ¡El modelo podría estar corrupto! Error: {str(e)}")
            # Intenta eliminar el modelo corrupto
            try:
                os.remove(self.current_model_path)
                print(f"[CLEANUP] Modelo corrupto eliminado: {self.current_model_path}")
            except:
                pass
            self.current_model_path = None
            raise

    def save_model(self, trained_model):
        """Serializa y guarda el modelo en caché"""
        try:
            # Generar nombre de archivo único con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_file = os.path.join(self.cache_dir, f"entrenamiento_{timestamp}_{uuid.uuid4().hex[:6]}.pkl")
            
            # Serializar componentes del modelo
            model_data = {
                'metrics': trained_model['metrics'],
                'training_date': trained_model['training_date'],
                'features': trained_model['features'],
                'model': base64.b64encode(pickle.dumps(trained_model['model'])).decode('utf-8'),
                'scaler': base64.b64encode(pickle.dumps(trained_model['scaler'])).decode('utf-8'),
                'version': '1.0'
            }
            
            # Guardar en archivo
            with open(cache_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Limpiar modelo anterior si existe
            if self.current_model_path and os.path.exists(self.current_model_path):
                os.remove(self.current_model_path)
            
            self.current_model_path = cache_file
            logger.info(f"Modelo guardado en caché: {cache_file}")
            return cache_file
            
        except Exception as e:
            logger.error(f"Error al guardar modelo en caché: {str(e)}")
            raise

def encode_data(df):
    """Codificación de datos categóricos"""
    # Implementa tu lógica de codificación aquí
    return df

def train_model(training_data):
    """
    Entrena un modelo de regresión logística
    
    Args:
        training_data (pd.DataFrame): Datos de entrenamiento
        
    Returns:
        dict: Modelo entrenado y componentes
    """
    try:
        logger.info("Iniciando entrenamiento del modelo")
        
        # 1. Preprocesamiento de datos
        df_encoded = encode_data(training_data)
        X = df_encoded.drop(columns=['abandono', 'nombre'], errors='ignore')
        y = df_encoded['abandono']
        features = list(X.columns)
        
        # 2. Escalado de características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 3. Entrenamiento del modelo
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
        
        # 6. Cálculo de métricas
        y_pred = model.predict(X_scaled)
        metrics = {
            'accuracy': round(accuracy_score(y, y_pred), 3),
            'precision': round(precision_score(y, y_pred), 3),
            'recall': round(recall_score(y, y_pred), 3),
            'f1': round(f1_score(y, y_pred), 3),
            'cv_accuracy': [round(score, 3) for score in cv_scores],
            'cv_mean_accuracy': round(cv_scores.mean(), 3)
        }
        
        logger.info(f"Métricas del modelo: {metrics}")
        
        return {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metrics': metrics,
            'training_date': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en entrenamiento del modelo: {str(e)}")
        raise

def predict(model_data, input_data):
    """
    Realiza predicciones con el modelo entrenado
    
    Args:
        model_data (dict): Modelo entrenado y componentes
        input_data (pd.DataFrame): Datos para predicción
        
    Returns:
        dict: Resultados de la predicción
    """
    try:
        logger.info("Iniciando predicción")
        
        # 1. Validar componentes del modelo
        required_components = ['model', 'scaler', 'features']
        for comp in required_components:
            if comp not in model_data:
                raise ValueError(f"Falta componente requerido: {comp}")
        
        # 2. Preprocesamiento de datos
        df_encoded = encode_data(input_data)
        X = df_encoded.drop(columns=['nombre'], errors='ignore')
        
        # 3. Validar características
        if list(X.columns) != model_data['features']:
            raise ValueError(
                "Las características no coinciden con las de entrenamiento. "
                f"Esperadas: {model_data['features']}, Obtenidas: {list(X.columns)}"
            )
        
        # 4. Escalado y predicción
        X_scaled = model_data['scaler'].transform(X)
        predictions = model_data['model'].predict(X_scaled)
        probabilities = model_data['model'].predict_proba(X_scaled)[:, 1]
        
        # 5. Preparar resultados
        results = df_encoded.copy()
        results['prediccion'] = predictions
        results['probabilidad'] = probabilities
        
        return {
            'success': True,
            'predictions': results.to_dict('records'),
            'metrics': model_data['metrics'],
            'prediction_date': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

# Ejemplo de uso seguro
if __name__ == "__main__":
    print("\n=== PRUEBA DEL SISTEMA DE MODELOS ===")
    
    try:
        # 1. Inicializar el gestor
        cache_manager = ModelCacheManager()
        print("\n[TEST] ModelCacheManager inicializado correctamente")
        
        # 2. Simular datos de entrenamiento (ejemplo)
        print("\n[TEST] Simulando entrenamiento...")
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'abandono': [0, 1, 0, 1, 0]
        }
        train_df = pd.DataFrame(data)
        
        # 3. Entrenar y guardar modelo
        trained_model = train_model(train_df)
        model_path = cache_manager.save_model(trained_model)
        print(f"\n[TEST] Modelo guardado en: {model_path}")
        
        # 4. Cargar modelo
        loaded_model = cache_manager.load_model()
        print("\n[TEST] Modelo cargado correctamente")
        print("Features:", loaded_model['features'])
        print("Métricas:", loaded_model['metrics'])
        
        # 5. Simular predicción
        print("\n[TEST] Simulando predicción...")
        test_data = pd.DataFrame({
            'feature1': [2, 3],
            'feature2': [4, 3]
        })
        
        prediction = predict(loaded_model, test_data)
        print("\n[TEST] Resultados de predicción:", prediction)
        
    except Exception as e:
        print(f"\n[ERROR] Error en las pruebas: {str(e)}")
    
    print("\n=== PRUEBAS COMPLETADAS ===")