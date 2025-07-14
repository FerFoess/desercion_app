import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .datos import codificar_datos
import matplotlib.pyplot as plt
import seaborn as sns
import os


def procesar_clusterizacion(csv_path, variables=None, n_clusters=3, limite_datos=None):
    import warnings
    warnings.filterwarnings("ignore")

    # Leer y codificar
    df = pd.read_csv(csv_path)
    df = codificar_datos(df)

    if limite_datos is not None:
        df = df.head(limite_datos)

    # 🔒 Variables fijas para análisis de riesgo (interpretación)
    variables_fijas = [
        'promedio', 'reprobo_materia', 'materias_reprobadas',
        'motivacion', 'asistencia', 'dificultad_materias', 'horas_estudio',
        'estres', 'condicion_medica',
        'vive_con_tutores', 'apoyo_familiar', 'conflictos_casa',
        'trabaja', 'trabaja_apoyo', 'economia_dificulta', 'acceso_recursos',
        'interes_terminar', 'orientacion', 'conoce_apoyos',
        'edad', 'sexo'
    ]

    # Verificación de columnas existentes (por si el CSV no tiene todas)
    variables_utiles = [var for var in variables_fijas if var in df.columns]

    # Usar todo para el clustering
    X = df.select_dtypes(include=['number'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters

    # Interpretación con las variables útiles
    resumen = df.groupby('cluster')[variables_utiles].mean()

    # Determinar alto riesgo
    suma_medias = resumen.sum(axis=1)
    cluster_alto_riesgo = suma_medias.idxmin()

    df['etiqueta_riesgo'] = df['cluster'].apply(
        lambda c: "Alto riesgo de abandono" if c == cluster_alto_riesgo else "Riesgo medio/bajo"
    )

    graf1, graf2 = generar_graficos(df)

    return {
        "resumen_clusters": resumen,
        "datos_clusterizados": df,
        "centroides": pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns),
        "etiqueta_alto_riesgo": cluster_alto_riesgo,
        "grafica1": graf1,
        "grafica2": graf2
    }


def generar_graficos(df):
    os.makedirs("desercion_app/app/static", exist_ok=True)

    # Gráfico 1: Conteo de estudiantes por cluster
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='cluster', palette='Set2')
    plt.title("Cantidad de estudiantes por clúster")
    plt.tight_layout()
    graf1 = "grafico1.png"
    ruta1 = os.path.join("desercion_app", "app", "static", graf1)
    plt.savefig(ruta1)
    plt.close()

    # Gráfico 2: Promedio de una variable (ej. motivación) por clúster
    if 'motivacion' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df, x='cluster', y='motivacion', ci=None, palette='Set3')
        plt.title("Motivación promedio por clúster")
        plt.tight_layout()
        graf2 = "grafico2.png"
        ruta2 = os.path.join("desercion_app", "app", "static", graf2)
        plt.savefig(ruta2)
        plt.close()
    else:
        graf2 = None

    return graf1, graf2

