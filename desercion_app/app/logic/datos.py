import pandas as pd
from sklearn.impute import SimpleImputer

def codificar_datos(df, es_prediccion=False):
    # Limpiar espacios y saltos de línea en los nombres de columna
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\n', ' ', regex=True)

    # Renombrar las columnas para que coincidan con los nombres del CSV que me proporcionaste
    renombrar_columnas = {
        "Te sientes motivadoa a seguir estudiando": "motivacion",
        "Asistencia promedio en % asistencia a clases": "asistencia",
        "Te sientes estresadoa con tus estudios actualmente": "estres",
        "Tienes acceso a recursos escolares internet libros computadora": "acceso_recursos"
    }

    # Si es el archivo de predicción, eliminamos la columna "abandono"
    if es_prediccion:
        df = df.drop(columns=["abandono"], errors="ignore")  # Asegurarnos de que no esté presente

    # Renombrar las columnas de df
    df = df.rename(columns=renombrar_columnas)

    # Identificar columnas numéricas y no numéricas
    columnas_numericas = df.select_dtypes(include=["float64", "int64"]).columns
    columnas_no_numericas = df.select_dtypes(exclude=["float64", "int64"]).columns

    # Imputar los valores nulos en las columnas numéricas con la media
    imputer_numerico = SimpleImputer(strategy="mean")  # Rellenar con la media
    df[columnas_numericas] = imputer_numerico.fit_transform(df[columnas_numericas])

    # Imputar los valores nulos en las columnas no numéricas con el valor más frecuente (modo)
    imputer_no_numerico = SimpleImputer(strategy="most_frequent")  # Rellenar con el valor más frecuente
    df[columnas_no_numericas] = imputer_no_numerico.fit_transform(df[columnas_no_numericas])

    # Verificar si las columnas ahora coinciden con las que espera el modelo
    expected_columns = ["edad", "sexo", "nivel_escolar", "promedio", "reprobo_materia", "materias_reprobadas", 
                        "motivacion", "asistencia", "dificultad_materias", "horas_estudio", "estres", 
                        "considera_abandonar", "condicion_medica", "vive_con_tutores", "apoyo_familiar", 
                        "conflictos_casa", "trabaja", "trabaja_apoyo", "economia_dificulta", "acceso_recursos", 
                        "interes_terminar", "orientacion", "conoce_apoyos", "abandono"]

    # Comprobamos si las columnas coinciden con las que el modelo espera
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"Las siguientes columnas están faltando en el DataFrame: {missing_cols}")
    else:
        print("Las columnas coinciden correctamente.")

    return df
