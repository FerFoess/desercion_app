import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .datos import codificar_datos

def procesar_clusterizacion(csv_path, variables, n_clusters=3):
    # Leer y codificar datos
    df = pd.read_csv(csv_path)
    df = codificar_datos(df)

    # Validar que las variables estén en el DataFrame
    if not set(variables).issubset(set(df.columns)):
        faltantes = set(variables) - set(df.columns)
        raise ValueError(f"Variables no encontradas en el dataset: {faltantes}")

    # Seleccionar variables para clustering
    X = df[variables]

    # Escalar variables para igualar pesos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Añadir cluster al DataFrame
    df['cluster'] = clusters

    # Resumen estadístico por cluster para interpretación
    resumen = df.groupby('cluster')[variables].mean()

    # Etiquetar clusters: asumimos cluster con menor suma de medias = alto riesgo
    suma_medias = resumen.sum(axis=1)
    cluster_alto_riesgo = suma_medias.idxmin()

    def etiqueta_riesgo(row):
        if row['cluster'] == cluster_alto_riesgo:
            return "Alto riesgo de abandono"
        else:
            return "Riesgo medio/bajo"

    df['etiqueta_riesgo'] = df.apply(etiqueta_riesgo, axis=1)

    # Devolver resultados útiles para la app
    return {
        "resumen_clusters": resumen,
        "datos_clusterizados": df,
        "centroides": pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=variables),
        "etiqueta_alto_riesgo": cluster_alto_riesgo
    }
