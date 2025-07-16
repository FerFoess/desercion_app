import os
import json
import uuid
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import (
    Blueprint, render_template, request, redirect, 
    url_for, flash, send_file, session, current_app
)
from werkzeug.utils import secure_filename
from .logic.modelo import entrenar_y_predecir
from .utils import generate_pdf, generate_excel
from .logic.datos import codificar_datos
from .logic.graficos import generar_graficos_brutos, obtener_graficos_guardados
# En routes.py, reemplaza el almacenamiento en sesión por:
from flask import current_app
import os


def guardar_datos_csv(df):
    # Guardar el CSV en disco en lugar de la sesión
    os.makedirs('app/data_cache', exist_ok=True)
    cache_file = os.path.join('app/data_cache', f'temp_{uuid.uuid4().hex[:8]}.pkl')
    df.to_pickle(cache_file)  # Más eficiente que JSON
    session['cache_file'] = cache_file  # Solo guardamos la ruta
    session['nombre_archivo'] = archivo.filename

def cargar_datos_csv():
    if 'cache_file' in session:
        cache_file = session['cache_file']
        if os.path.exists(cache_file):
            return pd.read_pickle(cache_file)
    return None

bp = Blueprint('main', __name__)

# Configuración
ALLOWED_EXTENSIONS = {'csv'}
ITEMS_PER_PAGE = 20

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def guardar_grafico(fig, prefix="plot"):
    """Guarda gráficos en la carpeta static/img"""
    img_dir = os.path.join(current_app.static_folder, 'img')
    os.makedirs(img_dir, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex[:6]}.png"
    filepath = os.path.join(img_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return filename

def clean_old_files(directory, prefix):
    """Elimina archivos antiguos para evitar acumulación"""
    for f in os.listdir(directory):
        if f.startswith(prefix):
            try:
                os.remove(os.path.join(directory, f))
            except Exception as e:
                current_app.logger.error(f"Error eliminando {f}: {e}")



@bp.route('/', methods=['GET', 'POST'])
def index():
    current_app.logger.info("Inicio de la ruta index")
    pagina = request.args.get('pagina', 1, type=int)
    por_pagina = 10

    if request.method == 'POST':
        current_app.logger.debug("Método POST recibido")
        
        if 'archivo_entrenamiento' not in request.files:
            current_app.logger.warning("No se recibió archivo en el request")
            flash('No se seleccionó archivo', 'error')
            return redirect(url_for('main.index'))
            
        archivo = request.files['archivo_entrenamiento']
        current_app.logger.info(f"Archivo recibido: {archivo.filename}")
        
        if archivo.filename == '':
            current_app.logger.warning("Nombre de archivo vacío")
            flash('No se seleccionó archivo', 'error')
            return redirect(url_for('main.index'))
            
        if archivo and allowed_file(archivo.filename):
            try:
                df = pd.read_csv(archivo)
                current_app.logger.info("CSV leído correctamente")
                current_app.logger.debug(f"Columnas del CSV: {df.columns.tolist()}")
                current_app.logger.debug(f"Muestra de datos:\n{df.head(2).to_string()}")
                
                guardar_datos_csv(df)
                generar_graficos_brutos(df)
                
                flash('Archivo cargado correctamente', 'success')
                return redirect(url_for('main.index'))
            except Exception as e:
                current_app.logger.error(f"Error al procesar archivo: {str(e)}", exc_info=True)
                flash(f'Error: {str(e)}', 'error')
                return redirect(url_for('main.index'))

    # Cargar datos
    df = cargar_datos_csv()
    datos_paginados = []
    columnas = []
    total_paginas = 1
    
    if df is not None:
        current_app.logger.debug(f"Datos cargados desde caché. Columnas: {df.columns.tolist()}")
        total_paginas = max(1, math.ceil(len(df) / por_pagina))
        pagina = max(1, min(pagina, total_paginas))
        inicio = (pagina - 1) * por_pagina
        datos_paginados = df.iloc[inicio:inicio+por_pagina].to_dict('records')
        columnas = df.columns.tolist()
        current_app.logger.debug(f"Columnas a renderizar: {columnas}")

    return render_template('index.html',
                         datos_paginados=datos_paginados,
                         columnas=columnas,
                         pagina_actual=pagina,
                         total_paginas=total_paginas,
                         graficos_brutos=obtener_graficos_guardados(),
                         nombre_archivo=session.get('nombre_archivo'))

@bp.route('/limpiar', methods=['POST'])
def limpiar_datos():
    if 'cache_file' in session:
        try:
            os.remove(session['cache_file'])
        except:
            pass
        session.pop('cache_file', None)
    session.pop('nombre_archivo', None)
    flash('Datos reseteados', 'success')
    return redirect(url_for('main.index'))

@bp.route('/procesar', methods=['POST'])
def procesar():
    if 'train_data' not in session or 'predict_data' not in session:
        flash('Primero sube archivos válidos para entrenamiento y predicción', 'error')
        return redirect(url_for('main.index'))
    
    try:
        # Obtener parámetros del formulario
        limit = request.form.get('limit', 100, type=int)
        
        # Cargar datos
        df_train = pd.read_json(session['train_data'])
        df_pred = pd.read_json(session['predict_data'])
        
        # Procesamiento con el modelo
        resultados = entrenar_y_predecir(df_train, df_pred, limit)
        
        # Guardar resultados en sesión
        session['results'] = {
            'metrics': resultados.get('metrics', {}),
            'predictions': resultados.get('predictions', []),
            'coefficients': resultados.get('coefficients', {}),
            'feature_importance': resultados.get('feature_importance', {})
        }
        
        return redirect(url_for('main.show_metrics'))
        
    except Exception as e:
        current_app.logger.error(f"Error en procesamiento: {str(e)}")
        flash(f'Error en el procesamiento: {str(e)}', 'error')
        return redirect(url_for('main.index'))

@bp.route('/resultados/metricas')
def show_metrics():
    if 'results' not in session:
        flash('No hay resultados disponibles', 'error')
        return redirect(url_for('main.index'))
    
    return render_template('resultados_metrica.html',
                         metrics=session['results']['metrics'])

@bp.route('/resultados/predicciones')
def show_predictions():
    if 'results' not in session:
        flash('No hay resultados disponibles', 'error')
        return redirect(url_for('main.index'))
    
    # Interpretación de coeficientes
    interpretation = {
        'acceso_recursos': 'Media - Acceso a recursos mejora permanencia',
        'asistencia': 'Alta - Más asistencia, menor riesgo',
        'promedio': 'Alta - Promedio bajo aumenta riesgo',
        # ... agregar más interpretaciones
    }
    
    return render_template('resultados_predicciones.html',
                         predictions=session['results']['predictions'],
                         coefficients=session['results']['coefficients'],
                         interpretation=interpretation)

@bp.route('/resultados/graficos')
def show_graphs():
    if 'results' not in session:
        flash('No hay resultados disponibles', 'error')
        return redirect(url_for('main.index'))
    
    # Generar gráficos automáticos
    df = pd.DataFrame(session['results']['predictions'])
    auto_plots = []
    
    # Gráfico de distribución de probabilidades
    if 'probabilidad_abandono' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['probabilidad_abandono'], bins=20, kde=True, ax=ax)
        ax.set_title('Distribución de Probabilidades de Abandono')
        filename = guardar_grafico(fig, 'prob_dist')
        auto_plots.append({
            'title': 'Distribución de Probabilidades',
            'filename': filename
        })
    
    # Gráfico de importancia de características
    if session['results']['feature_importance']:
        features = list(session['results']['feature_importance'].keys())
        importance = list(session['results']['feature_importance'].values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importance, y=features, palette='viridis', ax=ax)
        ax.set_title('Importancia de Variables en la Predicción')
        filename = guardar_grafico(fig, 'feature_imp')
        auto_plots.append({
            'title': 'Importancia de Variables',
            'filename': filename
        })
    
    return render_template('resultados_graficos.html',
                         auto_plots=auto_plots,
                         columns=df.columns.tolist())

@bp.route('/exportar/pdf', methods=['POST'])
def export_pdf():
    if 'results' not in session:
        flash('No hay resultados para exportar', 'error')
        return redirect(url_for('main.index'))
    
    try:
        selected_items = request.form.getlist('export_items')
        output_path = generate_pdf(session['results'], selected_items)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        current_app.logger.error(f"Error generando PDF: {str(e)}")
        flash(f'Error al generar PDF: {str(e)}', 'error')
        return redirect(url_for('main.show_metrics'))

@bp.route('/exportar/excel', methods=['POST'])
def export_excel():
    if 'results' not in session:
        flash('No hay resultados para exportar', 'error')
        return redirect(url_for('main.index'))
    
    try:
        selected_items = request.form.getlist('export_items')
        output_path = generate_excel(session['results'], selected_items)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        current_app.logger.error(f"Error generando Excel: {str(e)}")
        flash(f'Error al generar Excel: {str(e)}', 'error')
        return redirect(url_for('main.show_metrics'))

@bp.route('/limpiar', methods=['POST'])
def clean_session():
    """Limpia los datos de la sesión"""
    session.clear()
    flash('Datos reiniciados correctamente', 'success')
    return redirect(url_for('main.index'))