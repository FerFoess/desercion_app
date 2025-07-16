from flask import current_app
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from matplotlib.ticker import PercentFormatter
import numpy as np

def generar_graficos_brutos(df):
    graficos = []
    
    # Mapeo de columnas específicas de tu CSV
    column_mapping = {
        'abandono': 'considera_abandonar',
        'dificultad_academica': 'dificultad_materias',
        'problemas_familiares': 'conflictos_casa'
    }
    
    try:
        # 1. Gráfico de Edades (siempre funciona)
        plt.figure(figsize=(10, 6))
        sns.histplot(df['edad'], bins=12, kde=True, color='skyblue')
        plt.title('Distribución de Edades')
        graficos.append(guardar_grafico(plt, 'edad'))
        plt.close()
        
        # 2. Gráfico Asistencia vs Abandono (con columna mapeada)
        if 'asistencia' in df.columns and column_mapping['abandono'] in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                x=column_mapping['abandono'], 
                y='asistencia', 
                data=df
            )
            plt.title('Asistencia vs Intención de Abandono')
            graficos.append(guardar_grafico(plt, 'asistencia_abandono'))
            plt.close()
        
        # 3. Gráfico de Correlación (con columnas numéricas)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
            plt.title('Correlación entre Variables')
            graficos.append(guardar_grafico(plt, 'correlacion'))
            plt.close()
        
        # 4. Gráfico Factores de Riesgo (personalizado para tus columnas)
        risk_factors = [
            'motivacion',
            'estres',
            'economia_dificulta',
            column_mapping['dificultad_academica'],
            column_mapping['problemas_familiares']
        ]
        existing_factors = [f for f in risk_factors if f in df.columns]
        
        if existing_factors:
            plt.figure(figsize=(12, 6))
            df[existing_factors].mean().sort_values().plot.barh(color='darkorange')
            plt.title('Factores de Riesgo Promedio')
            graficos.append(guardar_grafico(plt, 'factores_riesgo'))
            plt.close()
            
    except Exception as e:
        current_app.logger.error(f"Error generando gráficos: {str(e)}", exc_info=True)
    
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