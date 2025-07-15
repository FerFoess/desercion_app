import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

from .datos import codificar_datos


def entrenar_y_predecir(ruta_entrenamiento, ruta_prediccion, limite_datos=None):
    # Leer archivos
    df_train = pd.read_csv(ruta_entrenamiento)
    df_test = pd.read_csv(ruta_prediccion)

    # Codificar
    df_train = codificar_datos(df_train)
    df_test = codificar_datos(df_test)

    # Aplicar límite
    if limite_datos:
        df_test = df_test.head(limite_datos)

    if 'abandono' not in df_train.columns:
        raise ValueError("El archivo de entrenamiento debe tener la columna 'abandono'")

    # Separar variables y target
    X_train = df_train.drop(columns=['abandono', 'nombre'])
    y_train = df_train['abandono']
    X_test = df_test.drop(columns=['nombre'])

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predicción
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    df_test['abandono_predicho'] = y_pred
    df_test['probabilidad_abandono'] = y_pred_proba

    # Etiquetas
    df_test['etiqueta_riesgo'] = df_test['abandono_predicho'].apply(
        lambda x: 'Alto riesgo de abandono' if x == 1 else 'Riesgo medio/bajo'
    )

    # Imprimir las primeras 10 predicciones para depuración
    print("Primeras 10 predicciones:", y_pred[:10])
    print("Primeros 10 valores de probabilidad de abandono:", y_pred_proba[:10])

    # Calcular métricas con datos de entrenamiento (puedes cambiar a validación si tienes)
    y_train_pred = model.predict(X_train_scaled)
    metricas = {
        'accuracy': round(accuracy_score(y_train, y_train_pred), 3),
        'precision': round(precision_score(y_train, y_train_pred), 3),
        'recall': round(recall_score(y_train, y_train_pred), 3),
        'f1_score': round(f1_score(y_train, y_train_pred), 3),
    }

    # Imprimir métricas para depuración
    print("Métricas calculadas:", metricas)

    # Generar gráficos ROC y Precisión-Recall usando test
    grafico_roc, grafico_prec_rec = generar_graficos_modelo(y_test=None, y_pred_proba=y_pred_proba)

    return {
        'metricas': metricas,
        'datos_predicciones': df_test.reset_index(drop=True),
        'grafico_roc': grafico_roc,
        'grafico_prec_rec': grafico_prec_rec,
        'etiqueta_alto_riesgo': 1  # Etiqueta para "alto riesgo" = 1
    }

def generar_graficos_modelo(y_test, y_pred_proba):
    # Crear carpeta estática si no existe
    os.makedirs("desercion_app/app/static", exist_ok=True)

    # Como no tienes y_test en predicción (solo tienes para entrenamiento), puedes pasar None
    # Para gráfico ROC, si no tienes y_test real, este gráfico no se puede generar bien.
    # Aquí asumiremos que solo generamos gráficos basados en predicciones (ejemplo simplificado)

    # Gráfico 1: ROC Curve (si y_test disponible)
    graf1_path = None
    if y_test is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.tight_layout()
        graf1 = "roc_curve.png"
        graf1_path = os.path.join("desercion_app", "app", "static", graf1)
        plt.savefig(graf1_path)
        plt.close()

    # Gráfico 2: Precisión-Recall Curve (si y_test disponible)
    graf2_path = None
    if y_test is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        graf2 = "prec_rec_curve.png"
        graf2_path = os.path.join("desercion_app", "app", "static", graf2)
        plt.savefig(graf2_path)
        plt.close()

    # Si no hay y_test, se pueden generar gráficos alternativos como conteos o promedios (puedes usar el método anterior para eso)

    return graf1_path, graf2_path
