import datetime
import json
import os
import pickle
import uuid
import math
import glob
import pandas as pd
import numpy as np
import matplotlib
from pip import main
matplotlib.use('Agg')  # Configuración importante para servidor
import matplotlib.pyplot as plt
import seaborn as sns
from flask import (
    Blueprint, jsonify, render_template, request, redirect, 
    url_for, flash, send_file, session, current_app
)
from werkzeug.utils import secure_filename
from .logic.modelo import ModelCacheManager, train_model, predict
from datetime import datetime 

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
    
def verificar_permisos():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cache_dir = os.path.join(base_dir, 'data_cache')

    
    if not os.path.exists(cache_dir):
        current_app.logger.error(f"Directorio no existe: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
    
    if not os.access(cache_dir, os.R_OK):
        current_app.logger.error(f"No hay permisos de lectura en: {cache_dir}")
        raise PermissionError(f"No se puede leer el directorio {cache_dir}")
    
@bp.route('/verificar_ruta')
def verificar_ruta():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cache_dir = os.path.join(base_dir, 'data_cache')


    
    return jsonify({
        'base_dir': base_dir,
        'cache_dir': cache_dir,
        'existe': os.path.exists(cache_dir),
        'archivos': os.listdir(cache_dir) if os.path.exists(cache_dir) else []
    })    
    
"""----- CAMBIOS PARA GUARDAR LOS MODELOS DE ENTRENAMIENTO EN HISTORICOS"""
def obtener_historial_entrenamientos():
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cache_dir = os.path.join(base_dir, 'data_cache')
        
        if not os.path.exists(cache_dir):
            current_app.logger.error(f"Directorio no existe: {cache_dir}")
            return []
        
        archivos = []
        for f in os.listdir(cache_dir):
            if f.lower().startswith('entrenamiento_') and f.lower().endswith('.pkl'):
                archivo_path = os.path.join(cache_dir, f)
                if os.path.isfile(archivo_path):
                    archivos.append(archivo_path)
        
        historial = []
        for archivo in archivos:
            try:
                nombre = os.path.basename(archivo)
                stat = os.stat(archivo)
                fecha_creacion = datetime.datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                tamano = stat.st_size
                historial.append({
                    'nombre': nombre,      # ✅ solo el nombre
                    'fecha': fecha_creacion,
                    'tamano': tamano
                })
            except Exception as e:
                current_app.logger.error(f"Error procesando {archivo}: {str(e)}")
        
        # Ordenar por fecha descendente
        historial.sort(key=lambda x: x['fecha'], reverse=True)
        return historial
        
    except Exception as e:
        current_app.logger.error(f"Error en obtener_historial: {str(e)}", exc_info=True)
        return []
    
@bp.route('/cargar_modelo', methods=['POST'])
def cargar_modelo():
    try:
        nombre_modelo = request.form.get('nombre_modelo')
        if not nombre_modelo:
            return jsonify({'success': False, 'message': 'Nombre de modelo no proporcionado'})
        
        base_dir = os.path.abspath(os.path.dirname(__file__))
        ruta_modelo = os.path.join(base_dir, '..', 'data_cache', nombre_modelo)
        
        if not os.path.exists(ruta_modelo):
            return jsonify({'success': False, 'message': 'Archivo de modelo no encontrado'})
        
        session['cache_file_entrenamiento'] = ruta_modelo
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': 'Modelo cargado correctamente',
            'nombre_modelo': nombre_modelo
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@bp.route('/eliminar_modelo', methods=['POST'])
def eliminar_modelo():
    """Elimina un modelo del historial"""
    try:
        nombre_modelo = request.form.get('nombre_modelo')
        if nombre_modelo:
            ruta_modelo = os.path.join('data_cache', nombre_modelo)
            if os.path.exists(ruta_modelo):
                # Si es el modelo actualmente cargado, limpiar la sesión
                if 'cache_file_entrenamiento' in session and session['cache_file_entrenamiento'] == ruta_modelo:
                    limpiar_cache('entrenamiento')
                os.remove(ruta_modelo)
                return jsonify({'success': True, 'message': 'Modelo eliminado correctamente'})
        return jsonify({'success': False, 'message': 'Error al eliminar el modelo'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@bp.route('/obtener_historial', methods=['GET'])
def obtener_historial():
    """Devuelve el historial de modelos en formato JSON"""
    try:
        historial = obtener_historial_entrenamientos()
        return jsonify({
            'success': True,
            'historial': historial,
            'count': len(historial)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'historial': [],
            'count': 0
        })
        
@bp.route('/debug_archivos')
def debug_archivos():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cache_dir = os.path.join(base_dir, 'data_cache')

    
    return jsonify({
        'base_dir': base_dir,
        'cache_dir': cache_dir,
        'existe': os.path.exists(cache_dir),
        'archivos': os.listdir(cache_dir) if os.path.exists(cache_dir) else []
    })
    
@bp.route('/debug_directorios')
def debug_directorios():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cache_dir = os.path.join(base_dir, 'data_cache')


    
    return jsonify({
        'base_dir': base_dir,
        'cache_dir': cache_dir,
        'existe_cache': os.path.exists(cache_dir),
        'archivos_en_cache': os.listdir(cache_dir) if os.path.exists(cache_dir) else []
    })

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
    """Guarda gráficos en img/graficos con estructura consistente"""
    img_dir = os.path.join(current_app.static_folder, 'img')
    os.makedirs(img_dir, exist_ok=True)
    
    nombre_archivo = f"{prefix}_{uuid.uuid4().hex[:6]}.png"
    ruta_completa = os.path.join(img_dir, nombre_archivo)
    
    plt.savefig(ruta_completa, bbox_inches='tight', dpi=150)
    plt.close()
    
    return f'img/{nombre_archivo}'

def limpiar_graficos_anteriores(tipo='all'):
    """Elimina gráficos antiguos manteniendo estructura"""
    img_dir = os.path.join(current_app.static_folder, 'img')
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.endswith('.png'):
                if tipo == 'all' or f.startswith(f"{tipo}_"):
                    try:
                        os.remove(os.path.join(img_dir, f))
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
    img_dir = os.path.join(current_app.static_folder, 'img' )
    
    if os.path.exists(img_dir):
        # Mapeo de prefijos a títulos legibles
        nombres_legibles = {
            'train': 'Entrenamiento',
            'pred': 'Predicción',
            'prev': 'Preevaluación',
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
                        'nombre_archivo': f'img/{f}',
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

                # 2. ⭐⭐ NUEVO: Entrenar y guardar modelo automáticamente ⭐⭐
                try:
                    modelo = train_model(df, model_cache)  # Usa model_cache.save_model()
                    session['modelo_entrenado'] = True
                    flash('Modelo entrenado exitosamente', 'success')
                except Exception as e:
                    flash(f'Error entrenando modelo: {str(e)}', 'warning')

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
        # 1. Cargar datos
        df = cargar_datos_cache('entrenamiento')
        if df is None:
            flash('Error al cargar datos de entrenamiento', 'error')
            return redirect(url_for('main.index'))
        
        # 2. Entrenar modelo (ESTA ES LA PARTE CLAVE)
        modelo_entrenado = train_model(df, model_cache)  # <- Asegúrate de pasar model_cache
        
        # 3. Guardar en sesión
        session['modelo_entrenado'] = True
        session.modified = True
        
        flash('Modelo entrenado y guardado correctamente', 'success')
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
    pagina = request.args.get('pagina', 1, type=int)
    por_pagina = ITEMS_PER_PAGE  # Usa el mismo valor que tienes para index (ejemplo: 10 o 20)

    if request.method == 'POST':
        if 'archivo_prediccion' not in request.files:
            flash('No se seleccionó archivo', 'error')
            print("[DEBUG] archivo_prediccion NO está en request.files")
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
                limpiar_graficos_anteriores('prev')
                generar_graficos_brutos(df, 'prev')
                
                flash('Archivo para predicción cargado correctamente', 'success')
                return redirect(url_for('main.predecir'))
            except Exception as e:
                current_app.logger.error(f"Error procesando archivo de predicción: {str(e)}", exc_info=True)
                flash(f'Error: {str(e)}', 'error')
                return redirect(url_for('main.predecir'))
    
    df_pred = cargar_datos_cache('prediccion')
    modelo_entrenado = model_cache.has_model()

    datos_paginados = []
    columnas = []
    total_paginas = 1

    if df_pred is not None and not df_pred.empty:
        columnas = df_pred.columns.tolist()
        total_paginas = max(1, math.ceil(len(df_pred) / por_pagina))
        pagina = max(1, min(pagina, total_paginas))
        inicio = (pagina - 1) * por_pagina
        datos_paginados = df_pred.iloc[inicio:inicio+por_pagina].to_dict('records')

        print(f"[DEBUG] Mostrando página {pagina} de {total_paginas}, filas {inicio}-{inicio+por_pagina}")
    else:
        print("[DEBUG] No hay datos cargados aún para predicción.")

    # Recuperar gráficos desde sesión o disco
    if 'graficos_pred' in session:
        graficos = json.loads(session['graficos_pred'])
    else:
        graficos = obtener_graficos_guardados('pred')
        
    

    # --- Renderizar plantilla ---
    return render_template('prediccion.html',
        datos_prediccion=df_pred.to_dict('records') if df_pred is not None else [],
        columnas_prediccion=df_pred.columns.tolist() if df_pred is not None else [],
        graficos_prediccion=obtener_graficos_guardados('prev'),
        modelo_entrenado=modelo_entrenado
    )

@bp.route('/ejecutar_prediccion', methods=['POST'])
def ejecutar_prediccion():
    try:
        # 1. Verificar archivo de predicción
        df_pred = cargar_datos_cache('prediccion')
        if df_pred is None:
            flash('Primero sube un archivo para predicción', 'error')
            return redirect(url_for('main.predecir'))

        # 2. Verificar modelo entrenado usando el caché
        if not model_cache.has_model():  # 👈 Nuevo: Verifica en caché en lugar de sesión
            flash('Primero debes entrenar un modelo', 'error')
            return redirect(url_for('main.index'))

        # 3. Cargar modelo desde caché
        modelo_cargado = model_cache.load_model()
        if modelo_cargado is None:
            flash('Error al cargar el modelo entrenado', 'error')
            return redirect(url_for('main.index'))
        
        # 4. Ejecutar predicción
        resultados = predict(modelo_cargado, df_pred)
    
        if not resultados['success']:
            flash(f'Error en predicción: {resultados["error"]}', 'error')
            return redirect(url_for('main.predecir'))
        
        # 5. Guardar resultados (aquí puedes mantener la sesión o moverlo a caché)
        cache_file = os.path.join('data_cache', f'resultados_{uuid.uuid4().hex[:8]}.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(resultados, f)

        session['cache_resultados'] = cache_file
        
        flash('Predicción ejecutada correctamente', 'success')
        return redirect(url_for('main.mostrar_metricas'))

    except Exception as e:
        current_app.logger.error(f"Error en predicción: {str(e)}", exc_info=True)
        flash(f'Error al predecir: {str(e)}', 'error')
        return redirect(url_for('main.predecir'))
    

@bp.route('/resultados')
def mostrar_resultados():
    try:
        # 1. Verificar caché en sesión
        if 'cache_resultados' not in session:
            flash('No hay resultados disponibles. Ejecuta una predicción primero.', 'error')
            return redirect(url_for('main.predecir'))

        # 2. Cargar resultados de predicción
        cache_file = session['cache_resultados']
        if not os.path.exists(cache_file):
            session.pop('cache_resultados', None)
            flash('Los resultados han expirado. Por favor, genera una nueva predicción.', 'error')
            return redirect(url_for('main.predecir'))

        with open(cache_file, 'rb') as f:
            resultados = pickle.load(f)

        # 3. Obtener ruta CORRECTA (al mismo nivel que app/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Sube a desercion_app/
        cache_dir = os.path.join(base_dir, 'data_cache')  # Ahora apunta directamente a data_cache/
        
        current_app.logger.debug(f"Buscando modelos en: {cache_dir}")

        # 4. Buscar modelos disponibles
        try:
            listado_modelos = sorted(
                [f for f in os.listdir(cache_dir) if f.startswith('modelo_') and f.endswith('.pkl')],
                key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)),
                reverse=True
            )
        except FileNotFoundError:
            flash('No se encontró el directorio de modelos. Contacta al administrador.', 'error')
            return redirect(url_for('main.index'))

        if not listado_modelos:
            flash('No hay modelos disponibles. Entrena un modelo primero.', 'error')
            return redirect(url_for('main.index'))

        modelo_path = os.path.join(cache_dir, listado_modelos[0])
        current_app.logger.info(f'Usando modelo: {listado_modelos[0]}')

        # 5. Cargar modelo y extraer coeficientes
        with open(modelo_path, 'rb') as f:
            modelo_data = pickle.load(f)

        modelo = modelo_data['model']
        features = modelo_data['features']
        
        if hasattr(modelo, 'coef_'):
            coeficientes = list(zip(features, modelo.coef_[0])) if modelo.coef_.ndim > 1 else list(zip(features, modelo.coef_))
            coeficientes = sorted(coeficientes, key=lambda x: abs(x[1]), reverse=True)
        else:
            coeficientes = []
            current_app.logger.warning('Modelo no tiene coeficientes')

        # 6. Preparar datos para la plantilla
        context = {
            'predicciones': resultados.get('predictions', []),
            'coeficientes': coeficientes,
            'fecha_prediccion': resultados.get('prediction_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'nombre_modelo': listado_modelos[0],
            'metricas': resultados.get('metrics', {}),
            'graficos': resultados.get('graficos', {})
        }

        return render_template('resultados_predicciones.html', **context)

    except Exception as e:
        current_app.logger.error(f'Error en mostrar_resultados: {str(e)}', exc_info=True)
        flash('Ocurrió un error al mostrar los resultados. Por favor, inténtalo nuevamente.', 'error')
        return redirect(url_for('main.index'))


@bp.route('/graficos-prediccion')
def mostrar_graficos_prediccion():
    try:
        if 'cache_resultados' not in session:
            flash('Primero debes generar predicciones', 'error')
            return redirect(url_for('main.predecir'))
        
        cache_file = session['cache_resultados']
        
        # Verificación adicional del archivo
        if not os.path.exists(cache_file):
            flash('Los datos de predicción han expirado', 'error')
            return redirect(url_for('main.predecir'))

        with open(cache_file, 'rb') as f:
            resultados = pickle.load(f)
        
        
        if 'predictions' not in resultados:
            flash('Datos de predicción no tienen el formato esperado', 'error')
            return redirect(url_for('main.predecir'))

        df_predicciones = pd.DataFrame(resultados['predictions'])
        
        # Debug: Verifica las columnas del DataFrame
        current_app.logger.debug(f"Columnas del DataFrame: {str(df_predicciones.columns.tolist())}")
        
        graficos = generar_graficos_prediccion(df_predicciones)
        
        if not graficos:
            flash('No se pudieron generar gráficos con los datos disponibles', 'warning')
            return redirect(url_for('main.predecir'))
        
        return render_template('resultados_graficos.html',
                            graficos_brutos=graficos,
                            tipo='prediccion')
            
    except Exception as e:
        current_app.logger.error(f"Error completo mostrando gráficos: {str(e)}", exc_info=True)
        flash('Error al generar gráficos de predicción. Detalles en logs.', 'error')
        return redirect(url_for('main.predecir'))



def generar_graficos_prediccion(df):
    """Genera exactamente 6 gráficos de predicción, evitando duplicados"""
    graficos = []
    
    # Verificar si el DataFrame tiene datos
    if df.empty:
        print("DEBUG: DataFrame vacío, no se generarán gráficos.")
        return graficos
    try:
        img_dir = os.path.join(current_app.static_folder, 'img' )
        os.makedirs(img_dir, exist_ok=True)
        
        # Por una de estas opciones:
        plt.style.use('seaborn-v0_8')  # Estilo moderno equivalente
        # O:
        plt.style.use('ggplot')        # Alternativa popular
        # O:
        sns.set_theme() 
        colores = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948']
        
        # Lista de gráficos a generar
        graficos_config = [
            {
                'nombre': 'probabilidades',
                'generador': lambda: sns.histplot(df['probabilidad'], bins=20, kde=True, color=colores[0]),
                'titulo': 'Distribución de Probabilidades',
                'descripcion': 'Distribución de probabilidades de abandono',
                'requiere': ['probabilidad']
            },
            {
                'nombre': 'asistencia_probabilidad',
                'generador': lambda: sns.scatterplot(x='asistencia', y='probabilidad', data=df, hue='prediccion', palette=colores[1:3]),
                'titulo': 'Asistencia vs Probabilidad',
                'descripcion': 'Relación entre asistencia y probabilidad de abandono',
                'requiere': ['asistencia', 'probabilidad', 'prediccion']
            },
            {
                'nombre': 'edad_probabilidad',
                'generador': lambda: sns.boxplot(x='edad', y='probabilidad', data=df, color=colores[2]),
                'titulo': 'Probabilidad por Edad',
                'descripcion': 'Distribución de probabilidades por edad',
                'requiere': ['edad', 'probabilidad']
            },
            {
                'nombre': 'sexo_probabilidad',
                'generador': lambda: sns.barplot(x='sexo', y='probabilidad', data=df, ci=None, palette=[colores[4], colores[5]]),
                'titulo': 'Probabilidad por Sexo',
                'descripcion': 'Comparación de probabilidades entre sexos',
                'requiere': ['sexo', 'probabilidad']
            },
            {
                'nombre': 'motivacion_probabilidad',
                'generador': lambda: sns.regplot(x='motivacion', y='probabilidad', data=df, scatter_kws={'alpha':0.3}),
                'titulo': 'Motivación vs Probabilidad',
                'descripcion': 'Relación entre motivación y probabilidad de abandono',
                'requiere': ['motivacion', 'probabilidad']
            },
            {
                'nombre': 'factores_riesgo',
                'generador': lambda: df[['estres', 'economia_dificulta', 'dificultad_materias']].mean().plot.bar(color=colores),
                'titulo': 'Factores de Riesgo Promedio',
                'descripcion': 'Factores que influyen en el abandono',
                'requiere': ['estres', 'economia_dificulta', 'dificultad_materias']
            }
        ]
        for config in graficos_config[:6]:
            try:
                # Verificar columnas requeridas
                if not all(col in df.columns for col in config['requiere']):
                    current_app.logger.warning(f"Columnas faltantes para gráfico {config['nombre']}")
                    continue
                    
                nombre_archivo = f"pred_{config['nombre']}.png"
                ruta_completa = os.path.join(img_dir, nombre_archivo)
                
                plt.figure(figsize=(10, 6))
                config['generador']()
                plt.title(config['titulo'])
                
                # Guardar el gráfico
                plt.savefig(ruta_completa, bbox_inches='tight', dpi=150)
                plt.close()
                
                graficos.append({
                    'titulo': config['titulo'],
                    'nombre_archivo': f'img/{nombre_archivo}',
                    'descripcion': config['descripcion']
                })
                    
            except Exception as e:
                current_app.logger.error(f"Error generando gráfico {config['nombre']}: {str(e)}", exc_info=True)
                continue
        
    except Exception as e:
        current_app.logger.error(f"Error en generar_graficos_prediccion: {str(e)}", exc_info=True)
        
    return graficos[:6]

@bp.route('/exportar/<tipo>')
def exportar_resultados(tipo):
    print(f"DEBUG: Exportando resultados en formato {tipo}")
    if 'cache_resultados' not in session:
        flash('No hay resultados para exportar', 'error')
        return redirect(url_for('main.predecir'))
    
    cache_file = session['cache_resultados']
    if not os.path.exists(cache_file):
        flash('Los resultados han expirado o fueron eliminados', 'error')
        return redirect(url_for('main.predecir'))
    
    with open(cache_file, 'rb') as f:
        resultados = pickle.load(f)
    
    try:
        df = pd.DataFrame(resultados['predictions'])
        
        # Crear carpeta temporal dentro de static para exportar
        output_dir = os.path.join(current_app.static_folder, 'temp')
        os.makedirs(output_dir, exist_ok=True)

        if tipo == 'csv':
            output_path = os.path.join(output_dir, 'resultados_prediccion.csv')
            df.to_csv(output_path, index=False)
            return send_file(output_path, as_attachment=True)
        
        elif tipo == 'excel':
            output_path = os.path.join(output_dir, 'resultados_prediccion.xlsx')
            df.to_excel(output_path, index=False)
            return send_file(output_path, as_attachment=True)
        
        elif tipo == 'pdf':
            from fpdf import FPDF
            
            # Crear PDF
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Configuración inicial
            pdf.set_font('Arial', 'B', 16)
            
            # Agregar título
            pdf.cell(0, 10, 'Reporte de Resultados de Predicción', 0, 1, 'C')
            pdf.ln(10)
            
            # Obtener todas las imágenes de la carpeta gráficos
            graphics_dir = os.path.join(current_app.static_folder, 'img')
            all_images = [f for f in os.listdir(graphics_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            # Organizar imágenes por categorías
            categories = {
                'train': [],
                'pred': [],
                'prev': []
            }
            
            for img in all_images:
                if img.startswith('train_'):
                    categories['train'].append(img)
                elif img.startswith('pred_'):
                    categories['pred'].append(img)
                elif img.startswith('prev_'):
                    categories['prev'].append(img)
            
            # Sección 1: Train (Entrenamiento)
            if categories['train']:
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, '1. Gráficos de Entrenamiento (train)', 0, 1)
                pdf.set_font('Arial', '', 12)
                
                for img in sorted(categories['train']):
                    img_path = os.path.join(graphics_dir, img)
                    try:
                        pdf.image(img_path, x=10, w=190)
                        pdf.ln(5)
                    except:
                        current_app.logger.warning(f"No se pudo agregar imagen {img} al PDF")
                        continue
            
            # Sección 2: Pred (Predicciones)
            if categories['pred']:
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, '2. Gráficos de Predicción (pred)', 0, 1)
                pdf.set_font('Arial', '', 12)
                
                for img in sorted(categories['pred']):
                    img_path = os.path.join(graphics_dir, img)
                    try:
                        pdf.image(img_path, x=10, w=190)
                        pdf.ln(5)
                    except:
                        current_app.logger.warning(f"No se pudo agregar imagen {img} al PDF")
                        continue

            # Sección 3: Prev (Preevaluación)
            if categories['prev']:
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, '3. Gráficos de prealgoritmo (prev)', 0, 1)
                pdf.set_font('Arial', '', 12)

                for img in sorted(categories['prev']):
                    img_path = os.path.join(graphics_dir, img)
                    try:
                        pdf.image(img_path, x=10, w=190)
                        pdf.ln(5)
                    except:
                        current_app.logger.warning(f"No se pudo agregar imagen {img} al PDF")
                        continue
            
            # Guardar PDF
            output_path = os.path.join(output_dir, 'resultados_prediccion.pdf')
            pdf.output(output_path)
            
            return send_file(output_path, as_attachment=True)
        
        else:
            flash('Formato de exportación no válido', 'error')
            return redirect(url_for('main.mostrar_resultados'))
    
    except Exception as e:
        current_app.logger.error(f"Error exportando resultados: {str(e)}", exc_info=True)
        flash(f'Error al exportar resultados: {str(e)}', 'error')
        return redirect(url_for('main.mostrar_resultados'))



@bp.route('/explorador-datos')
def explorador_datos():
    try:
        # Verificación combinada más eficiente
        cache_file = session.get('cache_resultados')
        if not cache_file or not os.path.exists(cache_file):
            flash('Primero debes generar predicciones o los resultados han expirado', 'error')
            return redirect(url_for('main.predecir'))
            
        with open(cache_file, 'rb') as f:
            resultados = pickle.load(f)
        print(f"DEBUG: Resultados cargados en explorador_datos: {resultados}")
        if 'predictions' not in resultados:
            flash('Los resultados no tienen el formato correcto', 'error')
            return redirect(url_for('main.predecir'))
            
        # Pasar los resultados al template
        return render_template('resultados_datos.html', 
                            resultados=resultados['predictions'],
                            metadatos={
                                'fecha_prediccion': resultados.get('prediction_date', 'N/A'),
                                'modelo_usado': resultados.get('model_name', 'Desconocido')
                            })
            
    except Exception as e:
        current_app.logger.error(f"Error en explorador_datos: {str(e)}", exc_info=True)
        flash(f'Error técnico al cargar los resultados: {str(e)}', 'error')
        return redirect(url_for('main.index'))


@bp.route('/resultados/datos_json')
def datos_json():
    try:
        # Verificar si hay resultados en caché
        if 'cache_resultados' not in session:
            return jsonify({'error': 'No hay datos disponibles'}), 404
            
        cache_file = session['cache_resultados']
        
        # Verificar que el archivo existe físicamente
        if not os.path.exists(cache_file):
            return jsonify({'error': 'Archivo de resultados no encontrado'}), 404
            
        # Cargar y devolver los datos
        with open(cache_file, 'rb') as f:
            resultados = pickle.load(f)
        
        # Asegurar que tenemos predictions
        if 'predictions' not in resultados:
            return jsonify({'error': 'Formato de datos inválido'}), 500
            
        return jsonify(resultados['predictions'])
        
    except Exception as e:
        current_app.logger.error(f"Error en datos_json: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})
    
@bp.route('/graficos-random-forest')
def graficos_random_forest():
    try:
        if 'cache_resultados' not in session:
            flash("Primero debes generar predicciones", "error")
            return redirect(url_for('main.index'))

        cache_file = session['cache_resultados']
        if not os.path.exists(cache_file):
            flash("Archivo de resultados no encontrado", "error")
            return redirect(url_for('main.index'))

        with open(cache_file, 'rb') as f:
            resultados = pickle.load(f)

        df = pd.DataFrame(resultados['predictions'])

        # Verificar columnas necesarias
        required_cols = ['prediccion', 'probabilidad', 'edad']
        if not all(col in df.columns for col in required_cols):
            flash("Datos insuficientes para generar gráficos", "error")
            return redirect(url_for('main.index'))

        # Obtener importancia de características si es un modelo Random Forest
        feature_importances = []
        features = []
        
        if model_cache.has_model():
            modelo_data = model_cache.load_model()
            if hasattr(modelo_data['model'], 'feature_importances_'):
                features = modelo_data.get('features', [])
                feature_importances = modelo_data['model'].feature_importances_.tolist()

        return render_template(
            "graficos_rf.html",
            datos=df.to_dict(orient="records"),
            fecha=resultados.get("prediction_date", ""),
            feature_importances=feature_importances,
            features=features
        )

    except Exception as e:
        current_app.logger.error(f"Error en graficos_random_forest: {str(e)}", exc_info=True)
        flash(f"Error al generar gráficos: {str(e)}", "error")
        return redirect(url_for("main.index"))
    try:
        if 'cache_resultados' not in session:
            flash("Primero debes predecir con Random Forest", "error")
            return redirect(url_for('main.index'))

        cache_file = session['cache_resultados']
        if not os.path.exists(cache_file):
            flash("Archivo de resultados no encontrado", "error")
            return redirect(url_for('main.index'))

        with open(cache_file, 'rb') as f:
            resultados = pickle.load(f)

        df = pd.DataFrame(resultados['predictions'])

        if not all(col in df.columns for col in ['prediccion', 'probabilidad', 'edad']):
            flash("No hay columnas suficientes para graficar", "error")
            return redirect(url_for('main.index'))

        cache_manager = ModelCacheManager()
        modelo_entrenado = cache_manager.load_model()

        if modelo_entrenado is None:
            feature_importances = []
            features = []
            logger.warning("No se encontró modelo entrenado en caché para obtener importancias")
        else:
            feature_importances = modelo_entrenado.get('feature_importances', [])
            features = modelo_entrenado.get('features', [])

        return render_template(
            "graficos_rf.html",
            datos=df.to_dict(orient="records"),
            fecha=resultados.get("prediction_date", ""),
            feature_importances=feature_importances,
            features=features
        )

    except Exception as e:
        logger.error(f"Error en graficos_rf: {str(e)}")
        flash(f"Error técnico al generar gráficas: {str(e)}", "error")
        return redirect(url_for("main.index"))