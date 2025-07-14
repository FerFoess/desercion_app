import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, session
from .logic.modelo import procesar_clusterizacion
from .utils import generate_pdf, generate_excel
import json

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@bp.route('/procesar', methods=['POST'])
def procesar():
    archivo = request.files['archivo_csv']
    if not archivo or not archivo.filename.endswith('.csv'):
        flash('Por favor sube un archivo CSV válido.')
        return redirect(url_for('main.index'))

    filepath = os.path.join('desercion_app', 'app', 'data', archivo.filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    archivo.save(filepath)

    variables = request.form.getlist('variables')
    if not variables:
        flash('Selecciona al menos una variable.')
        return redirect(url_for('main.index'))

    try:
        resultados = procesar_clusterizacion(filepath, variables, n_clusters=3)
    except Exception as e:
        flash(f"Error al procesar la clusterización: {str(e)}")
        return redirect(url_for('main.index'))

    # Guardar en sesión solo los datos para exportar
    session['resultados'] = resultados["datos_clusterizados"].to_dict(orient='records')

    # Pasar DataFrames para mostrar en template
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

        pdf_buffer = generate_pdf(resultados, items)
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='reporte.pdf'
        )
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

        excel_buffer = generate_excel(resultados, items)
        return send_file(
            excel_buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='reporte.xlsx'
        )
    except Exception as e:
        flash(f"No se pudo generar el Excel: {e}")
        return redirect(url_for('main.index'))
