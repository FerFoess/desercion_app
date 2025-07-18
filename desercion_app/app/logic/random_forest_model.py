import os
import logging
import pickle
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.logic.datos import codificar_datos
from app.logic.cache import ModelCacheManager

logger = logging.getLogger(__name__)

def train_random_forest_model(training_data, cache_manager=None):
    try:
        logger.info("Entrenando modelo Random Forest")

        df_encoded = codificar_datos(training_data, es_prediccion=False)
        X = df_encoded.drop(columns=['abandono', 'nombre'], errors='ignore')
        y = df_encoded['abandono']
        features = list(X.columns)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        model.fit(X_scaled, y)

        y_pred = model.predict(X_scaled)
        metrics = {
            'accuracy': round(accuracy_score(y, y_pred), 3),
            'precision': round(precision_score(y, y_pred), 3),
            'recall': round(recall_score(y, y_pred), 3),
            'f1': round(f1_score(y, y_pred), 3),
            'cv_mean_accuracy': round(cv_scores.mean(), 3)
        }

        feature_importances = model.feature_importances_

        trained_model = {
    'model': model,
    'scaler': scaler,
    'features': features,
    'metrics': metrics,
    'feature_importances': feature_importances.tolist(),  # agregar esta línea
    'training_date': datetime.now().isoformat(),
    'version': '2.0'
}

        if cache_manager is None:
            cache_manager = ModelCacheManager()
        
        cache_manager.save_model(trained_model)
        logger.info("Random Forest entrenado y guardado")
        return trained_model

    except Exception as e:
        logger.error(f"Error entrenando Random Forest: {str(e)}")
        raise

def predict_random_forest(model_data, input_data):
    try:
        df_encoded = codificar_datos(input_data, es_prediccion=True)
        X = df_encoded.drop(columns=['nombre'], errors='ignore')

        if list(X.columns) != model_data['features']:
            raise ValueError("Las columnas de entrada no coinciden con las del modelo entrenado")
        
        X_scaled = model_data['scaler'].transform(X)
        predictions = model_data['model'].predict(X_scaled)
        probabilities = model_data['model'].predict_proba(X_scaled)[:, 1]

        df_encoded['prediccion'] = predictions
        df_encoded['probabilidad'] = probabilities

        return {
            'success': True,
            'predictions': df_encoded.to_dict('records'),
            'metrics': model_data.get('metrics', {}),
            'prediction_date': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error en predicción RF: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
