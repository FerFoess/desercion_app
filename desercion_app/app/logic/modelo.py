import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    # Asegurar columna "abandono" en entrenamiento
    if 'abandono' not in df_train.columns:
        raise ValueError("El archivo de entrenamiento debe tener la columna 'abandono'")

    # Separar X y y
    X_train = df_train.drop(columns=['abandono', 'nombre'])
    y_train = df_train['abandono']
    X_test = df_test.drop(columns=['nombre'])

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Predicción
    y_pred = model.predict(X_test_scaled)
    df_test['abandono_predicho'] = y_pred

    # Etiquetas
    df_test['etiqueta_riesgo'] = df_test['abandono_predicho'].apply(
        lambda x: 'Alto riesgo de abandono' if x == 1 else 'Riesgo medio/bajo'
    )

    # Gráficos
    graf1, graf2 = generar_graficos(df_test)

    return {
        'datos_clusterizados': df_test.reset_index(drop=True),
        'resumen_clusters': df_test.groupby('abandono_predicho').mean().reset_index(),
        'etiqueta_alto_riesgo': 1,  # En regresión 1 = abandono
        'grafica1': graf1,
        'grafica2': graf2
    }


def generar_graficos(df):
    os.makedirs("desercion_app/app/static", exist_ok=True)

    # Gráfico 1: Conteo de estudiantes por predicción
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='abandono_predicho', palette='Set2')
    plt.title("Conteo de estudiantes según predicción de abandono")
    plt.tight_layout()
    graf1 = "grafico1.png"
    ruta1 = os.path.join("desercion_app", "app", "static", graf1)
    plt.savefig(ruta1)
    plt.close()

    # Gráfico 2: Promedio de motivación por predicción
    if 'motivacion' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df, x='abandono_predicho', y='motivacion', palette='Set3')
        plt.title("Motivación promedio según predicción")
        plt.tight_layout()
        graf2 = "grafico2.png"
        ruta2 = os.path.join("desercion_app", "app", "static", graf2)
        plt.savefig(ruta2)
        plt.close()
    else:
        graf2 = None

    return graf1, graf2
