import pandas as pd
from sklearn.impute import SimpleImputer

def codificar_datos(df, es_prediccion=False):
    # Limpiar espacios y saltos de línea en los nombres de columna
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\n', ' ', regex=True)

    # Renombrar las columnas para que coincidan con los nombres esperados por el modelo
    renombrar_columnas = {
        "Te sientes motivadoa a seguir estudiando": "motivacion",
        "Asistencia promedio en % asistencia a clases": "asistencia",
        "Te sientes estresadoa con tus estudios actualmente": "estres",
        "Tienes acceso a recursos escolares internet libros computadora": "acceso_recursos"
    }

    # Eliminar columna 'abandono' si es predicción
    if es_prediccion:
        df = df.drop(columns=["abandono"], errors="ignore")

    # Renombrar columnas
    df = df.rename(columns=renombrar_columnas)

    # Imputación de valores nulos
    columnas_numericas = df.select_dtypes(include=["float64", "int64"]).columns
    columnas_no_numericas = df.select_dtypes(exclude=["float64", "int64"]).columns

    imputer_numerico = SimpleImputer(strategy="mean")
    df[columnas_numericas] = imputer_numerico.fit_transform(df[columnas_numericas])

    imputer_no_numerico = SimpleImputer(strategy="most_frequent")
    df[columnas_no_numericas] = imputer_no_numerico.fit_transform(df[columnas_no_numericas])

    return df
