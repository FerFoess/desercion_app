import json
import os
import uuid
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configuración importante para servidor
import matplotlib.pyplot as plt
import seaborn as sns
from flask import (
    Blueprint, render_template, request, redirect, 
    url_for, flash, send_file, session, current_app
)
from werkzeug.utils import secure_filename
from .logic.modelo import ModelCacheManager, train_model, predict

# Configuración
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
ITEMS_PER_PAGE = 20

bp = Blueprint('main', __name__)

# Inicializar el gestor de caché de modelos
model_cache = ModelCacheManager()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def guardar_datos_cache(df, tipo='entrenamiento'):
    """Guarda un DataFrame en caché en disco"""
    try:
        os.makedirs('data_cache', exist_ok=True)
        cache_file = os.path.join('data_cache', f'{tipo}_{uuid.uuid4().hex[:8]}.pkl')
        df.to_pickle(cache_file)
        session[f'cache_file_{tipo}'] = cache_file
        session[f'nombre_archivo_{tipo}'] = secure_filename(f"{tipo}_{uuid.uuid4().hex[:4]}.csv")
        return cache_file
    except Exception as e:
        current_app.logger.error(f"Error guardando datos en caché: {str(e)}")
        raise

def cargar_datos_cache(tipo='entrenamiento'):
    """Carga un DataFrame desde caché"""
    try:
        cache_key = f'cache_file_{tipo}'
        if cache_key in session:
            cache_file = session[cache_key]
            if os.path.exists(cache_file):
                return pd.read_pickle(cache_file)
        return None
    except Exception as e:
        current_app.logger.error(f"Error cargando datos desde caché: {str(e)}")
        raise

def limpiar_cache(tipo='all'):
    """Elimina archivos de caché"""
    try:
        if tipo == 'all':
            keys = [k for k in session.keys() if k.startswith('cache_file_')]
            for key in keys:
                try:
                    if os.path.exists(session[key]):
                        os.remove(session[key])
                except:
                    pass
                session.pop(key, None)
                session.pop(f'nombre_archivo_{key.split("_")[-1]}', None)
        else:
            cache_key = f'cache_file_{tipo}'
            if cache_key in session:
                try:
                    if os.path.exists(session[cache_key]):
                        os.remove(session[cache_key])
                except:
                    pass
                session.pop(cache_key, None)
                session.pop(f'nombre_archivo_{tipo}', None)
    except Exception as e:
        current_app.logger.error(f"Error limpiando caché: {str(e)}")

def guardar_grafico(plt, prefix):
    """
    Guarda el gráfico generado y devuelve el path relativo desde static.
    """
    # Crear el directorio si no existe: static/img/graficos
    img_dir = os.path.join(current_app.static_folder, 'img', 'graficos')
    os.makedirs(img_dir, exist_ok=True)

    # Generar un nombre único
    nombre_archivo = f"{prefix}_{uuid.uuid4().hex[:6]}.png"

    # Ruta completa para guardar
    filepath = os.path.join(img_dir, nombre_archivo)

    # Guardar y cerrar el gráfico
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()

    # Retornar solo la ruta relativa desde static, por ejemplo:
    # 'img/graficos/train_edad_xxxxxx.png'
    return f'img/graficos/{nombre_archivo}'


def limpiar_graficos_anteriores(tipo='all'):
    """Elimina gráficos antiguos del tipo especificado"""
    img_dir = os.path.join(current_app.static_folder, 'img', 'graficos')
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.endswith('.png'):
                if tipo == 'all' or f.startswith(tipo):
                    try:
                        os.remove(os.path.join(img_dir, f))
                        current_app.logger.info(f"Gráfico eliminado: {f}")
                    except Exception as e:
                        current_app.logger.warning(f"No se pudo eliminar {f}: {str(e)}")

def generar_graficos_brutos(df, prefix='train'):
    graficos = []

    try:
        # 1. Gráfico de Edades
        if 'edad' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['edad'], bins=12, kde=True, color='skyblue')
            plt.title(f'Distribución de Edades ({prefix})')
            graficos.append({
                'titulo': 'Entrenamiento - Edad',
                'nombre_archivo': guardar_grafico(plt, f'{prefix}_edad'),
                'descripcion': 'Este gráfico muestra la distribución de edades de los estudiantes en el conjunto de datos de entrenamiento.'
            })

        # 2. Asistencia vs Abandono
        if 'asistencia' in df.columns and 'abandono' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='abandono', y='asistencia', data=df)
            plt.title(f'Asistencia vs Abandono ({prefix})')
            graficos.append({
                'titulo': 'Entrenamiento - Asistencia',
                'nombre_archivo': guardar_grafico(plt, f'{prefix}_asistencia_abandono'),
                'descripcion': 'Compara la asistencia media de quienes abandonaron frente a quienes no lo hicieron.'
            })

        # 3. Correlación entre variables
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f'Correlación entre Variables ({prefix})')
            graficos.append({
                'titulo': 'Entrenamiento - Correlación',
                'nombre_archivo': guardar_grafico(plt, f'{prefix}_correlacion'),
                'descripcion': 'Mapa de calor que muestra cómo se relacionan numéricamente las variables.'
            })

        # 4. Factores de Riesgo
        risk_factors = ['motivacion', 'estres', 'economia_dificulta', 'dificultad_materias', 'conflictos_casa']
        existing_factors = [f for f in risk_factors if f in df.columns]
        if existing_factors:
            plt.figure(figsize=(12, 6))
            df[existing_factors].mean().sort_values().plot.barh(color='darkorange')
            plt.title(f'Factores de Riesgo Promedio ({prefix})')
            plt.xlabel('Valor Promedio')
            graficos.append({
                'titulo': 'Entrenamiento - Factores',
                'nombre_archivo': guardar_grafico(plt, f'{prefix}_factores_riesgo'),
                'descripcion': 'Promedio de diversos factores de riesgo que podrían influir en el abandono.'
            })

        # 5. Abandono por Nivel Escolar
        if 'nivel_escolar' in df.columns and 'abandono' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x='nivel_escolar', hue='abandono', data=df)
            plt.title(f'Abandono por Nivel Escolar ({prefix})')
            plt.xticks(rotation=45)
            graficos.append({
                'titulo': 'Entrenamiento - Abandono',
                'nombre_archivo': guardar_grafico(plt, f'{prefix}_abandono_nivel'),
                'descripcion': 'Muestra cuántos estudiantes abandonaron según su nivel escolar.'
            })

        # 6. Motivación vs Promedio
        if 'motivacion' in df.columns and 'promedio' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='motivacion', y='promedio', data=df, alpha=0.6)
            plt.title(f'Relación entre Motivación y Promedio ({prefix})')
            graficos.append({
                'titulo': 'Entrenamiento - Motivación',
                'nombre_archivo': guardar_grafico(plt, f'{prefix}_motivacion_promedio'),
                'descripcion': 'Observa si existe relación entre el nivel de motivación y el promedio académico.'
            })

    except Exception as e:
        current_app.logger.error(f"Error generando gráficos {prefix}: {str(e)}", exc_info=True)

    return graficos


def obtener_graficos_guardados(prefix=None):
    graficos = []
    img_dir = os.path.join(current_app.static_folder, 'img', 'graficos')
    
    if os.path.exists(img_dir):
        # Mapeo de prefijos a títulos legibles
        nombres_legibles = {
            'train': 'Entrenamiento',
            'pred': 'Predicción',
            'eval': 'Evaluación'
        }
        
        # Obtener archivos ordenados por fecha de modificación (más recientes primero)
        archivos = sorted(os.listdir(img_dir), 
                         key=lambda x: os.path.getmtime(os.path.join(img_dir, x)), 
                         reverse=True)
        
        # Procesar solo archivos PNG
        for f in archivos:
            if f.endswith('.png'):
                if prefix is None or f.startswith(prefix):
                    file_prefix = f.split('_')[0]
                    titulo = f"{nombres_legibles.get(file_prefix, file_prefix)} - {f.split('_')[1].capitalize()}"
                    
                    graficos.append({
                        'titulo': titulo,
                        'nombre_archivo': f'img/graficos/{f}',
                        'tipo': file_prefix
                    })
    
    return graficos

@bp.route('/', methods=['GET', 'POST'])
def index():
    pagina = request.args.get('pagina', 1, type=int)
    por_pagina = ITEMS_PER_PAGE

    if request.method == 'POST':
        if 'archivo_entrenamiento' not in request.files:
            flash('No se seleccionó archivo', 'error')
            return redirect(url_for('main.index'))

        archivo = request.files['archivo_entrenamiento']
        if archivo.filename == '':
            flash('No se seleccionó archivo', 'error')
            return redirect(url_for('main.index'))

        if archivo and allowed_file(archivo.filename):
            try:
                if archivo.filename.endswith('.csv'):
                    df = pd.read_csv(archivo)
                else:
                    df = pd.read_excel(archivo)

                if 'abandono' not in df.columns:
                    flash('El archivo debe contener una columna "abandono"', 'error')
                    return redirect(url_for('main.index'))

                guardar_datos_cache(df, 'entrenamiento')
                limpiar_graficos_anteriores('train')
                graficos = generar_graficos_brutos(df, 'train')

                # 👉 Guardamos en sesión
                session['graficos_train'] = json.dumps(graficos)

                flash('Archivo de entrenamiento cargado correctamente', 'success')

                total_paginas = max(1, math.ceil(len(df) / por_pagina))
                datos_paginados = df.iloc[0:por_pagina].to_dict('records')

                return render_template('index.html',
                                       datos_paginados=datos_paginados,
                                       columnas=df.columns.tolist(),
                                       pagina_actual=1,
                                       total_paginas=total_paginas,
                                       graficos_brutos=graficos,
                                       nombre_archivo=session.get('nombre_archivo_entrenamiento'),
                                       modelo_entrenado='modelo_entrenado' in session)
            except Exception as e:
                current_app.logger.error(f"Error al procesar archivo: {str(e)}", exc_info=True)
                flash(f'Error: {str(e)}', 'error')
                return redirect(url_for('main.index'))

    # GET
    df = cargar_datos_cache('entrenamiento')
    datos_paginados = []
    columnas = []
    total_paginas = 1
    graficos = []

    if df is not None:
        total_paginas = max(1, math.ceil(len(df) / por_pagina))
        pagina = max(1, min(pagina, total_paginas))
        inicio = (pagina - 1) * por_pagina
        datos_paginados = df.iloc[inicio:inicio+por_pagina].to_dict('records')
        columnas = df.columns.tolist()

        # 👉 Recuperamos gráficos desde sesión
        if 'graficos_train' in session:
            graficos = json.loads(session['graficos_train'])
        else:
            graficos = obtener_graficos_guardados('train')

    return render_template('index.html',
                           datos_paginados=datos_paginados,
                           columnas=columnas,
                           pagina_actual=pagina,
                           total_paginas=total_paginas,
                           graficos_brutos=graficos,
                           nombre_archivo=session.get('nombre_archivo_entrenamiento'),
                           modelo_entrenado='modelo_entrenado' in session)

@bp.route('/limpiar', methods=['POST'])
def limpiar_datos():
    # Limpiar todo
    limpiar_cache('all')
    limpiar_graficos_anteriores('all')
    session.pop('modelo_entrenado', None)
    session.pop('resultados_prediccion', None)
    flash('Datos, gráficos y modelo reseteados correctamente', 'success')
    return redirect(url_for('main.index'))

@bp.route('/entrenar', methods=['POST'])
def entrenar():
    if 'cache_file_entrenamiento' not in session:
        flash('Primero sube un archivo CSV para entrenamiento', 'error')
        return redirect(url_for('main.index'))
    
    try:
        df = cargar_datos_cache('entrenamiento')
        
        if df is None:
            flash('No se pudieron cargar los datos de entrenamiento', 'error')
            return redirect(url_for('main.index'))
        
        # Entrenar el modelo
        modelo_entrenado = train_model(df)
        
        # Guardar el modelo en caché
        model_cache.save_model(modelo_entrenado)
        
        # Guardar en sesión y forzar el guardado
        session['modelo_entrenado'] = True
        session.modified = True  # ← Esto es importante
        
        print("✅ Modelo entrenado y sesión actualizada:", session['modelo_entrenado'])
        print("✅ Modelo entrenado y sesión actualizada:", session['modelo_entrenado'])

        
        flash('Modelo entrenado exitosamente', 'success')
        return redirect(url_for('main.mostrar_metricas'))
        
    except Exception as e:
        current_app.logger.error(f"Error entrenando modelo: {str(e)}", exc_info=True)
        flash(f'Error entrenando modelo: {str(e)}', 'error')
        return redirect(url_for('main.index'))

@bp.route('/metricas')
def mostrar_metricas():
    if 'modelo_entrenado' not in session:
        flash('Primero entrena el modelo', 'error')
        return redirect(url_for('main.index'))
    
    try:
        # Cargar modelo para obtener métricas
        modelo_cargado = model_cache.load_model()
        
        # Obtener gráficos de evaluación
        graficos_evaluacion = obtener_graficos_guardados('eval')
        
        return render_template('resultados_metrica.html',
                             metricas=modelo_cargado['metrics'],
                             graficos_evaluacion=graficos_evaluacion,
                             fecha_entrenamiento=modelo_cargado['training_date'])
    except Exception as e:
        current_app.logger.error(f"Error mostrando métricas: {str(e)}", exc_info=True)
        flash('Error al cargar las métricas del modelo', 'error')
        return redirect(url_for('main.index'))

@bp.route('/predecir', methods=['GET', 'POST'])
def predecir():
    if request.method == 'POST':
        # Verificar si el archivo fue enviado
        if 'archivo_prediccion' not in request.files:
            flash('No se seleccionó archivo', 'error')
            return redirect(url_for('main.predecir'))
            
        archivo = request.files['archivo_prediccion']
        
        # Verificar si se seleccionó un archivo
        if archivo.filename == '':
            flash('No se seleccionó archivo', 'error')
            return redirect(url_for('main.predecir'))
            
        if archivo and allowed_file(archivo.filename):
            try:
                # Leer archivo según extensión
                if archivo.filename.endswith('.csv'):
                    df = pd.read_csv(archivo)
                else:
                    df = pd.read_excel(archivo)
                
                # Guardar en caché
                guardar_datos_cache(df, 'prediccion')
                
                # Generar gráficos exploratorios
                limpiar_graficos_anteriores('pred')
                generar_graficos_brutos(df, 'pred')
                
                flash('Archivo para predicción cargado correctamente', 'success')
                return redirect(url_for('main.predecir'))
            except Exception as e:
                current_app.logger.error(f"Error procesando archivo de predicción: {str(e)}", exc_info=True)
                flash(f'Error: {str(e)}', 'error')
                return redirect(url_for('main.predecir'))
    
    df_pred = cargar_datos_cache('prediccion')
    
    # Verificar si hay modelo entrenado (usa get para evitar KeyError)
    modelo_entrenado = session.get('modelo_entrenado', False)
    
    # Debug: Imprime el estado para diagnóstico
    print(f"DEBUG - Modelo entrenado: {modelo_entrenado}, Datos cargados: {df_pred is not None}")
    print("DEBUG: modelo_entrenado =", session.get('modelo_entrenado'))
    print("DEBUG: modelo_entrenado =", session.get('modelo_entrenado'))
    print("DEBUG: archivo de predicción cargado =", df_pred.shape if df_pred is not None else 'No cargado')


    # Pasar los datos a la plantilla
    return render_template('prediccion.html',
        datos_prediccion=df_pred.to_dict('records') if df_pred is not None else [],
        columnas_prediccion=df_pred.columns.tolist() if df_pred is not None else [],
        graficos_prediccion=obtener_graficos_guardados('pred'),
        modelo_entrenado=modelo_entrenado
    )

@bp.route('/ejecutar_prediccion', methods=['POST'])
def ejecutar_prediccion():
    if 'cache_file_prediccion' not in session:
        flash('Primero sube un archivo para predicción', 'error')
        return redirect(url_for('main.predecir'))

    try:
        # Cargar datos de predicción desde caché
        df_pred = cargar_datos_cache('prediccion')
        if df_pred is None:
            flash('Error al cargar datos de predicción', 'error')
            return redirect(url_for('main.predecir'))

        # Verificar si hay un modelo entrenado
        if 'modelo_entrenado' not in session:
            flash('Primero debes entrenar un modelo', 'error')
            return redirect(url_for('main.index'))

        # Cargar el modelo entrenado
        modelo_cargado = model_cache.load_model()
        if modelo_cargado is None:
            flash('Error al cargar el modelo entrenado', 'error')
            return redirect(url_for('main.index'))
        
        # Hacer la predicción
        resultados = predict(modelo_cargado, df_pred)
        
        if not resultados['success']:
            flash(f'Error en predicción: {resultados["error"]}', 'error')
            return redirect(url_for('main.predecir'))
        
        # Guardar resultados en la sesión
        session['resultados_prediccion'] = resultados
        
        flash('Predicción ejecutada correctamente', 'success')
        return redirect(url_for('main.mostrar_resultados'))

    except Exception as e:
        current_app.logger.error(f"Error en predicción: {str(e)}", exc_info=True)
        flash(f'Error al predecir: {str(e)}', 'error')
        return redirect(url_for('main.predecir')) 

@bp.route('/resultados')
def mostrar_resultados():
    # Verificar si hay resultados
    if 'resultados_prediccion' not in session:
        flash('No hay resultados de predicción disponibles', 'error')
        return redirect(url_for('main.predecir'))

    try:
        resultados = session['resultados_prediccion']
        
        # Convertir a DataFrame para gráficos
        df_resultados = pd.DataFrame(resultados['predictions'])
        
        # Generar gráficos (opcional)
        graficos_resultados = []
        if 'probabilidad' in df_resultados.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df_resultados['probabilidad'], bins=20, kde=True)
            plt.title('Distribución de Probabilidades de Abandono')
            graficos_resultados.append(guardar_grafico(plt, 'res_probabilidades'))

        return render_template('resultados_prediccion.html',
                             predicciones=resultados['predictions'],
                             fecha_prediccion=resultados['prediction_date'],
                             graficos_resultados=graficos_resultados,
                             metricas=resultados.get('metrics', None))

    except Exception as e:
        current_app.logger.error(f"Error mostrando resultados: {str(e)}", exc_info=True)
        flash('Error al mostrar resultados', 'error')
        return redirect(url_for('main.predecir'))

@bp.route('/exportar/<tipo>')
def exportar_resultados(tipo):
    if 'resultados_prediccion' not in session:
        flash('No hay resultados para exportar', 'error')
        return redirect(url_for('main.predecir'))
    
    try:
        resultados = session['resultados_prediccion']
        df = pd.DataFrame(resultados['predictions'])
        
        if tipo == 'csv':
            output_path = os.path.join(current_app.static_folder, 'temp', 'resultados_prediccion.csv')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            return send_file(output_path, as_attachment=True)
        
        elif tipo == 'excel':
            output_path = os.path.join(current_app.static_folder, 'temp', 'resultados_prediccion.xlsx')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_excel(output_path, index=False)
            return send_file(output_path, as_attachment=True)
        
        else:
            flash('Formato de exportación no válido', 'error')
            return redirect(url_for('main.mostrar_resultados'))
    
    except Exception as e:
        current_app.logger.error(f"Error exportando resultados: {str(e)}", exc_info=True)
        flash(f'Error al exportar resultados: {str(e)}', 'error')
        return redirect(url_for('main.mostrar_resultados'))