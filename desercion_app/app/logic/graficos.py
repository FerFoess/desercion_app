from flask import current_app
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from matplotlib.ticker import PercentFormatter
import numpy as np

def generar_graficos_brutos(df):
    print(df)
    # --- PASO 1: Renombrar columnas para estandarizar ---
    df = df.rename(columns={
        'considera_abandonar': 'abandono',  # Columna clave para gráficos
        'dificultad_materias': 'dificultad_academica',
        'conflictos_casa': 'problemas_familiares'  # Ejemplo adicional
    })
    
    # --- PASO 2: Asegurar que las columnas críticas existan ---
    columnas_requeridas = {
        'edad': 'numérica',
        'asistencia': 'numérica',
        'abandono': 'binaria (0/1)',
        'promedio': 'numérica',
        'motivacion': 'numérica',
        'estres': 'numérica',
        'economia_dificulta': 'binaria (0/1)'
    }
    
    for col, tipo in columnas_requeridas.items():
        if col not in df.columns:
            raise ValueError(f"Columna requerida faltante: '{col}' (tipo: {tipo})")

    # --- PASO 3: Configuración inicial (original) ---
    plt.style.use('seaborn')
    img_dir = os.path.join('app', 'static', 'img', 'graficos')
    os.makedirs(img_dir, exist_ok=True)
    
    # Limpieza de gráficos antiguos
    for f in os.listdir(img_dir):
        os.remove(os.path.join(img_dir, f))
    
    graficos = []

    # --- PASO 4: Generación de gráficos (adaptada a tus columnas) ---
    try:
        # 1. Gráfico de Edades (siempre disponible)
        plt.figure(figsize=(10, 6))
        sns.histplot(df['edad'], bins=12, kde=True, color='skyblue')
        plt.title('Distribución de Edades')
        graficos.append(guardar_grafico('edad'))
        
        # 2. Gráfico de Asistencia vs Abandono
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='abandono', y='asistencia', data=df)
        plt.title('Asistencia vs Intención de Abandono')
        graficos.append(guardar_grafico('asistencia_abandono'))
        
        # 3. Gráfico de Pastel - Abandono
        plt.figure(figsize=(8, 8))
        df['abandono'].value_counts().plot.pie(
            autopct='%1.1f%%',
            labels=['No abandonará', 'Sí abandonará']
        )
        plt.title('Intención de Abandono Escolar')
        graficos.append(guardar_grafico('abandono_pastel'))
        
        # 4. Factores de Riesgo (personalizado para tus columnas)
        factores = ['motivacion', 'estres', 'economia_dificulta', 'dificultad_academica']
        plt.figure(figsize=(12, 6))
        df[factores].mean().sort_values().plot.barh(color='darkorange')
        plt.title('Factores de Riesgo Promedio')
        graficos.append(guardar_grafico('factores_riesgo'))
        
    except Exception as e:
        print(f"Error generando gráficos: {str(e)}")
        raise  # Opcional: eliminar en producción

    return graficos

def guardar_grafico(prefix):
    """Función auxiliar para guardar gráficos y devolver metadatos"""
    nombre_archivo = f"{prefix}_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(os.path.join(img_dir, nombre_archivo), bbox_inches='tight', dpi=100)
    plt.close()
    return {'titulo': prefix.replace('_', ' ').title(), 'nombre_archivo': f'graficos/{nombre_archivo}'}

def obtener_graficos_guardados():
    graficos = []
    img_dir = os.path.join('app', 'static', 'img', 'graficos')
    if os.path.exists(img_dir):
        archivos = sorted(os.listdir(img_dir))
        nombres_legibles = {
            'edad': 'Distribución de Edades',
            'asistencia': 'Asistencia vs Abandono',
            'correlacion': 'Correlación entre Variables',
            'abandono': 'Distribución de Abandono',
            'rendimiento': 'Rendimiento vs Asistencia',
            'factores': 'Factores de Riesgo'
        }
        
        for f in archivos:
            if f.endswith('.png'):
                key = f.split('_')[0]
                titulo = nombres_legibles.get(key, key.capitalize())
                graficos.append({
                    'titulo': titulo,
                    'nombre_archivo': f'graficos/{f}'
                })
    return graficos