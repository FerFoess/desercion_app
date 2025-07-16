import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os

from .datos import codificar_datos


def entrenar_y_predecir(ruta_entrenamiento, ruta_prediccion, limite_datos=None, limite_prediccion=None):
    # Leer archivos
    df_train = pd.read_csv(ruta_entrenamiento)
    df_test_original = pd.read_csv(ruta_prediccion)  # Guardamos el original completo

    # Aplicar límites
    if limite_datos is not None:
        df_train = df_train.head(limite_datos)
    if limite_prediccion is not None:
        df_test_original = df_test_original.head(limite_prediccion)

    # Codificar ambos
    df_train = codificar_datos(df_train)
    df_test_codificado = codificar_datos(df_test_original.copy())

    # Validar columna 'abandono'
    if 'abandono' not in df_train.columns:
        raise ValueError("El archivo de entrenamiento debe tener la columna 'abandono'.")

    # Entrenamiento
    X_train = df_train.drop(columns=['abandono', 'nombre'], errors='ignore')
    y_train = df_train['abandono']
    X_test = df_test_codificado.drop(columns=['nombre'], errors='ignore')

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predicción
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # ✅ Combinar datos originales con codificados y predicciones
    df_pred_final = df_test_original.copy()

    # Agregar columnas codificadas que no existan en el original
    for col in df_test_codificado.columns:
        if col != 'nombre' and col not in df_pred_final.columns:
            df_pred_final[col] = df_test_codificado[col]

    # Añadir predicciones
    df_pred_final['abandono_predicho'] = y_pred
    df_pred_final['probabilidad_abandono'] = y_pred_proba
    df_pred_final['etiqueta_riesgo'] = df_pred_final['abandono_predicho'].apply(
        lambda x: 'Alto riesgo de abandono' if x == 1 else 'Riesgo medio/bajo'
    )

    # Agregar 'abandono' si existe
    if 'abandono' in df_test_codificado.columns:
        df_pred_final['abandono'] = df_test_codificado['abandono']

    # Métricas del entrenamiento
    y_train_pred = model.predict(X_train_scaled)
    metricas = {
        'accuracy': round(accuracy_score(y_train, y_train_pred), 3),
        'precision': round(precision_score(y_train, y_train_pred), 3),
        'recall': round(recall_score(y_train, y_train_pred), 3),
        'f1_score': round(f1_score(y_train, y_train_pred), 3),
    }

    # Coeficientes
    variables = X_train.columns
    coeficientes = model.coef_[0]
    modelo_coeficientes = dict(sorted(
        zip(variables, coeficientes),
        key=lambda item: abs(item[1]), reverse=True
    ))

    # Gráficos ROC y Prec-Rec
    grafico_roc, grafico_prec_rec = generar_graficos_modelo(
        y_test=None, y_pred_proba=y_pred_proba
    )

    return {
        'metricas': metricas,
        'datos_predicciones': df_pred_final.reset_index(drop=True),
        'grafico_roc': grafico_roc,
        'grafico_prec_rec': grafico_prec_rec,
        'etiqueta_alto_riesgo': 1,
        'modelo_coeficientes': modelo_coeficientes
    }


def generar_graficos_modelo(y_test, y_pred_proba):
    os.makedirs("desercion_app/app/static", exist_ok=True)

    graf1_path = None
    graf2_path = None

    if y_test is not None:
        # ROC Curve
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
        graf1_path = os.path.join("desercion_app", "app", "static", "roc_curve.png")
        plt.savefig(graf1_path)
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        graf2_path = os.path.join("desercion_app", "app", "static", "prec_rec_curve.png")
        plt.savefig(graf2_path)
        plt.close()

    return graf1_path, graf2_path
