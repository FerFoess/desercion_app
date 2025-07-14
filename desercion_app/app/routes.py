import os
import json
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, session
from .logic.modelo import procesar_clusterizacion
from .utils import generate_pdf, generate_excel

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        archivo = request.files.get('archivo_csv')
        if not archivo or not archivo.filename.endswith('.csv'):
            flash('Por favor sube un archivo CSV válido.')
            return redirect(url_for('main.index'))

        # Guardar archivo
        filename = archivo.filename
        filepath = os.path.join('desercion_app', 'app', 'data', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        archivo.save(filepath)

        # Leer el CSV y generar vista previa
        try:
            df = pd.read_csv(filepath)
            preview_html = df.head(30).to_html(classes='table table-striped table-bordered', index=False)
        except Exception as e:
            flash(f"No se pudo leer el archivo: {e}")
            return redirect(url_for('main.index'))

        # Guardar ruta del archivo en sesión para procesar después
        session['csv_path'] = filepath
        session['csv_name'] = filename

        return render_template('index.html', preview=preview_html, filename=filename)

    return render_template('index.html')


@bp.route('/procesar', methods=['POST'])
def procesar():
    csv_path = session.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        flash('Primero sube un archivo CSV válido.')
        return redirect(url_for('main.index'))

    # Leer el límite de datos
    try:
        limite = int(request.form.get('limite_datos', 100))
    except ValueError:
        limite = 100

    try:
        # Ya no pasamos variables: se usan variables fijas internamente
        resultados = procesar_clusterizacion(csv_path, variables=None, n_clusters=3, limite_datos=limite)
    except Exception as e:
        flash(f"Error al procesar la clusterización: {str(e)}")
        return redirect(url_for('main.index'))

    # Guardar resultados para exportaciones
    session['resultados'] = resultados["datos_clusterizados"].to_dict(orient='records')

    # Preparar resultados para tabla HTML
    resultados['resumen_clusters'] = resultados['resumen_clusters'].reset_index()
    resultados['datos_clusterizados'] = resultados['datos_clusterizados'].reset_index()

    return render_template('results.html', resultados=resultados)


@bp.route('/export/pdf', methods=['POST'])
def export_pdf():
    try:
        items = json.loads(request.form.get('items', '[]'))
        resultados = session.get('resultados')
        if not resultados:
            flash('No hay resultados para exportar.')
            return redirect(url_for('main.index'))

        output_path = generate_pdf(resultados, items)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        flash(f"No se pudo generar el PDF: {e}")
        return redirect(url_for('main.index'))


@bp.route('/export/excel', methods=['POST'])
def export_excel():
    try:
        items = json.loads(request.form.get('items', '[]'))
        resultados = session.get('resultados')
        if not resultados:
            flash('No hay resultados para exportar.')
            return redirect(url_for('main.index'))

        output_path = generate_excel(resultados, items)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        flash(f"No se pudo generar el Excel: {e}")
        return redirect(url_for('main.index'))
