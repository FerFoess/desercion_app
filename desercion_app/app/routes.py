import os
import json
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, session
from .logic.modelo import entrenar_y_predecir
from .utils import generate_pdf, generate_excel
from .logic.datos import codificar_datos

bp = Blueprint('main', __name__)
@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        archivo_entrenamiento = request.files.get('archivo_entrenamiento')
        archivo_prediccion = request.files.get('archivo_prediccion')

        if (not archivo_entrenamiento or not archivo_entrenamiento.filename.endswith('.csv') or
            not archivo_prediccion or not archivo_prediccion.filename.endswith('.csv')):
            flash('Por favor sube archivos CSV válidos para entrenamiento y predicción.')
            return redirect(url_for('main.index'))

        data_dir = os.path.join('app', 'data')
        os.makedirs(data_dir, exist_ok=True)

        ruta_entrenamiento = os.path.join(data_dir, archivo_entrenamiento.filename)
        ruta_prediccion = os.path.join(data_dir, archivo_prediccion.filename)

        archivo_entrenamiento.save(ruta_entrenamiento)
        archivo_prediccion.save(ruta_prediccion)

        try:
            # Leer los archivos CSV
            df_train = pd.read_csv(ruta_entrenamiento)
            df_pred = pd.read_csv(ruta_prediccion)

            # Codificar los datos (eliminamos NaN y renombramos las columnas)
            df_train = codificar_datos(df_train, es_prediccion=False)
            df_pred = codificar_datos(df_pred, es_prediccion=True)

            # Vista previa de los datos
            preview_train = df_train.head(30).to_html(classes='table table-striped table-bordered', index=False)
            preview_pred = df_pred.head(30).to_html(classes='table table-striped table-bordered', index=False)

        except Exception as e:
            flash(f"No se pudieron leer los archivos: {e}")
            return redirect(url_for('main.index'))

        session['csv_train_path'] = ruta_entrenamiento
        session['csv_train_name'] = archivo_entrenamiento.filename
        session['csv_predict_path'] = ruta_prediccion
        session['csv_predict_name'] = archivo_prediccion.filename

        return render_template('index.html', 
                               preview_train=preview_train, 
                               preview_pred=preview_pred,
                               filename_train=archivo_entrenamiento.filename,
                               filename_pred=archivo_prediccion.filename)

    return render_template('index.html')



@bp.route('/procesar', methods=['POST'])
def procesar():
    csv_train_path = session.get('csv_train_path')
    csv_predict_path = session.get('csv_predict_path')

    if (not csv_train_path or not os.path.exists(csv_train_path) or
        not csv_predict_path or not os.path.exists(csv_predict_path)):
        flash('Primero sube archivos CSV válidos para entrenamiento y predicción.')
        return redirect(url_for('main.index'))

    try:
        limite = int(request.form.get('limite_datos', 100))
    except ValueError:
        limite = 100

    try:
        resultados = entrenar_y_predecir(csv_train_path, csv_predict_path, limite_datos=limite)
        # Verificar si los resultados se están guardando correctamente
        print("Resultados generados:", resultados)

    except Exception as e:
        flash(f"Error al aplicar regresión logística: {str(e)}")
        return redirect(url_for('main.index'))

    # Guardamos todo el diccionario de resultados en la sesión
    resultados_sesion = {}

    for k, v in resultados.items():
        if isinstance(v, pd.DataFrame):
            resultados_sesion[k] = v.to_dict(orient='records')
        else:
            resultados_sesion[k] = v

    # Imprimir los resultados que se están guardando
    print("Resultados guardados en la sesión:", resultados_sesion)

    session['resultados'] = resultados_sesion

    return redirect(url_for('main.resultados_metrica'))


@bp.route('/resultados/metrica')
def resultados_metrica():
    resultados = session.get('resultados')
    if not resultados or 'metricas' not in resultados:
        flash("No hay resultados para mostrar.")
        return redirect(url_for('main.index'))
    return render_template('resultados_metrica.html', metricas=resultados['metricas'])

@bp.route('/resultados/predicciones')
def resultados_predicciones():
    resultados = session.get('resultados')
    if not resultados or 'datos_predicciones' not in resultados:
        flash("No hay resultados para mostrar.")
        return redirect(url_for('main.index'))
    # Reconstruimos dataframe para renderizar
    df_pred = pd.DataFrame(resultados['datos_predicciones'])
    return render_template('resultados_predicciones.html', datos_predicciones=df_pred)

@bp.route('/resultados/graficos')
def resultados_graficos():
    resultados = session.get('resultados')
    if not resultados or 'grafico_roc' not in resultados or 'grafico_prec_rec' not in resultados:
        flash("No hay resultados para mostrar.")
        return redirect(url_for('main.index'))
    return render_template('resultados_graficos.html', 
                           grafico_roc=resultados['grafico_roc'], 
                           grafico_prec_rec=resultados['grafico_prec_rec'])

@bp.route('/resultados/exportar')
def resultados_exportar():
    resultados = session.get('resultados')
    if not resultados:
        flash("No hay resultados para mostrar.")
        return redirect(url_for('main.index'))
    return render_template('resultados_exportar.html')

@bp.route('/resultados/datos')
def resultados_datos():
    resultados = session.get('resultados')
    if not resultados or 'datos_predicciones' not in resultados:
        flash("No hay resultados para mostrar.")
        return redirect(url_for('main.index'))
    df_pred = pd.DataFrame(resultados['datos_predicciones'])
    return render_template('resultados_datos.html', datos_predicciones=df_pred)

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
