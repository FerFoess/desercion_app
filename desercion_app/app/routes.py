import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, session
from .logic.modelo import procesar_clusterizacion
from .utils import generate_pdf, generate_excel  # Simula funciones de exportación

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@bp.route('/export/pdf')
def export_pdf():
    try:
        output_path = generate_pdf(session.get('resultados', []))
        return send_file(output_path, as_attachment=True)
    except Exception:
        flash("No se pudo generar el PDF.")
        return redirect(url_for('main.index'))

@bp.route('/export/excel')
def export_excel():
    try:
        output_path = generate_excel(session.get('resultados', []))
        return send_file(output_path, as_attachment=True)
    except Exception:
        flash("No se pudo generar el archivo Excel.")
        return redirect(url_for('main.index'))

@bp.route('/procesar', methods=['POST'])
def procesar():
    archivo = request.files['archivo_csv']
    if not archivo or not archivo.filename.endswith('.csv'):
        flash('Por favor sube un archivo CSV válido.')
        return redirect(url_for('main.index'))

    filepath = os.path.join('app', 'data', archivo.filename)
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

    # Guardar solo los datos clusterizados para exportar
    session['resultados'] = resultados["datos_clusterizados"].to_dict(orient='records')

    # Convertir DataFrames a formato adecuado para render_template
    resultados['resumen_clusters'] = resultados['resumen_clusters'].reset_index()
    resultados['datos_clusterizados'] = resultados['datos_clusterizados'].reset_index()

    return render_template('results.html', resultados=resultados)
