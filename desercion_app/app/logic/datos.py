import pandas as pd

def codificar_datos(df):
    # Quitar espacios en los nombres de columna
    df.columns = df.columns.str.strip()

    # Renombrar columnas largas a nombres cortos para el modelo
    renombrar_columnas = {
        "Edad": "edad",
        "Sexo": "sexo",
        "¿Cuál es tu promedio general actual?": "promedio",
        "¿Has reprobado alguna materia en el último año?": "reprobo_materia",
        "¿Cuántas materias has reprobado?": "materias_reprobadas",
        "¿Te sientes motivado/a a seguir estudiando?": "motivacion",
        "Asistencia promedio (en %) de asistencia a clases": "asistencia",
        "¿Tienes dificultades con alguna materia en especial?": "dificultad_materias",
        "¿Cuántas horas estudias fuera de clases por semana?": "horas_estudio",
        "¿Te sientes estresado/a con tus estudios actualmente?": "estres",
        "¿Tienes alguna discapacidad o condición médica relevante?": "condicion_medica",
        "¿Vives con tus padres o tutores?": "vive_con_tutores",
        "¿Tus padres o tutores apoyan tu educación?": "apoyo_familiar",
        "¿Hay conflictos frecuentes en tu casa que afecten tu concentración?": "conflictos_casa",
        "¿Tienes un empleo actualmente?": "trabaja",
        "¿Trabajas para ayudar económicamente en casa?": "trabaja_apoyo",
        "¿Consideras que la economía familiar dificulta que sigas estudiando?": "economia_dificulta",
        "¿Tienes acceso a recursos escolares (internet, libros, computadora)?": "acceso_recursos",
        "¿Qué tanto te interesa terminar tus estudios?": "interes_terminar",
        "¿Recibes orientación académica o psicológica en tu institución?": "orientacion",
        "¿Conoces los apoyos o becas que tu escuela ofrece?": "conoce_apoyos"
    }

    # Renombrar columnas
    df = df.rename(columns=renombrar_columnas)

    # Mapeos de valores
    mapeos = {
        "edad": {
            "10-13": 1, "14-17": 2, "18-21": 3,
            "22-26": 4, "27-30": 5, "31-35": 6
        },
        "sexo": {"F": 0, "M": 1, "Otra": 2},
        "reprobo_materia": {"Sí": 1, "Si": 1, "No": 0},
        "motivacion": {
            "Nada": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
            "Mucho": 7, "Solo algunas veces": 3, "A veces": 4, "Muchísimo": 7
        },
        "asistencia": {"25%": 1, "50%": 2, "75%": 3, "100%": 4},
        "dificultad_materias": {"Sí": 1, "Si": 1, "No": 0},
        "horas_estudio": {
            "0-5": 1, "5-10": 2, "10-15": 3, "15-20": 4, "20-mas": 5,
            "0": 1, "5": 2, "10": 3, "15": 4, "20": 5
        },
        "estres": {"No": 0, "A veces": 1, "Sí": 2, "Si": 2},
        "condicion_medica": {"Sí": 1, "Si": 1, "No": 0},
        "vive_con_tutores": {"Sí": 1, "Si": 1, "No": 0},
        "apoyo_familiar": {
            "Nada interesados": 1, "1": 2, "2": 3, "3": 4,
            "4": 5, "5": 6, "Muy interesados": 7,
            "No": 2, "Sí": 6, "Si": 6
        },
        "conflictos_casa": {"Sí": 1, "Si": 1, "No": 0},
        "trabaja": {"Sí": 1, "Si": 1, "No": 0},
        "trabaja_apoyo": {"Sí": 1, "Si": 1, "No": 0},
        "economia_dificulta": {"Sí": 1, "Si": 1, "No": 0},
        "acceso_recursos": {"No": 0, "Parcialmente": 1, "Sí": 2, "Si": 2},
        "interes_terminar": {"Nada": 1, "Poco": 2, "Mucho": 3},
        "orientacion": {"Sí": 1, "Si": 1, "No": 0},
        "conoce_apoyos": {"Sí": 1, "Si": 1, "No": 0}
    }

    for columna, mapa in mapeos.items():
        if columna in df.columns:
            df[columna] = df[columna].map(mapa)

    # Convertir numéricos
    for col in ["materias_reprobadas", "promedio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    return df
