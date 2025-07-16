import os
import json
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, session
from .logic.modelo import entrenar_y_predecir
from .utils import generate_pdf, generate_excel
from .logic.datos import codificar_datos


bp = Blueprint('main', __name__)

def guardar_grafico(figura, carpeta="app/static", prefijo="grafico"):
    os.makedirs(carpeta, exist_ok=True)
    nombre = f"{prefijo}_{uuid.uuid4().hex[:8]}.png"
    ruta = os.path.join(carpeta, nombre)
    figura.savefig(ruta, bbox_inches='tight')
    plt.close(figura)
    return nombre

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
            df_train = pd.read_csv(ruta_entrenamiento)
            df_pred = pd.read_csv(ruta_prediccion)

            df_train = codificar_datos(df_train, es_prediccion=False)
            df_pred = codificar_datos(df_pred, es_prediccion=True)

            preview_train = df_train.head(30).to_html(classes='table table-striped table-bordered', index=False)
            preview_pred = df_pred.head(30).to_html(classes='table table-striped table-bordered', index=False)

            session['columnas_pred'] = df_pred.columns.tolist()

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
    except Exception as e:
        flash(f"Error al aplicar regresión logística: {str(e)}")
        return redirect(url_for('main.index'))

    resultados_sesion = {}
    for k, v in resultados.items():
        if isinstance(v, pd.DataFrame):
            resultados_sesion[k] = v.to_dict(orient='records')
        else:
            resultados_sesion[k] = v

    session['resultados'] = resultados_sesion

    if 'datos_predicciones' in resultados_sesion:
        df_pred = pd.DataFrame(resultados_sesion['datos_predicciones'])
        session['columnas_pred'] = df_pred.columns.tolist()

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

    datos_predicciones = resultados['datos_predicciones']
    coef_dict = resultados.get('modelo_coeficientes', {})
    coeficientes = list(coef_dict.items()) if coef_dict else []

    # Interpretaciones alineadas con variables reales
    interpretacion = {
        'acceso_recursos': 'Media (acceso a recursos mejora permanencia)',
        'apoyo_familiar': 'Alta (menos apoyo familiar, mayor abandono)',
        'asistencia': 'Alta (más asistencia, menor riesgo)',
        'condicion_medica': 'Media (condiciones médicas pueden afectar)',
        'conflictos_casa': 'Alta (más conflictos, mayor abandono)',
        'conoce_apoyos': 'Alta (si conoce apoyos, menor abandono)',
        'considera_abandonar': 'Muy alta (indicador directo de intención)',
        'dificultad_materias': 'Media (más dificultad, mayor riesgo)',
        'economia_dificulta': 'Alta (problemas económicos aumentan abandono)',
        'edad': 'Media (edad más alta puede relacionarse con abandono)',
        'estres': 'Media (mayor estrés, más riesgo)',
        'horas_estudio': 'Alta (menos horas, más abandono)',
        'interes_terminar': 'Alta (menos interés, mayor abandono)',
        'materias_reprobadas': 'Baja (poca influencia directa)',
        'motivacion': 'Alta (menos motivación, más riesgo)',
        'nivel_escolar': 'Media (niveles bajos pueden aumentar abandono)',
        'orientacion': 'Media (falta de orientación influye)',
        'promedio': 'Alta (promedio bajo, más abandono)',
        'reprobo_materia': 'Alta (reprobar influye en abandono)',
        'sexo': 'Baja (influencia leve entre géneros)',
        'trabaja': 'Media (trabajar puede aumentar abandono)',
        'trabaja_apoyo': 'Baja (trabajo con apoyo afecta poco)',
        'vive_con_tutores': 'Baja (ligera influencia del entorno familiar)'
    }

    return render_template('resultados_predicciones.html',
                           datos_predicciones=datos_predicciones,
                           coeficientes=coeficientes,
                           interpretacion=interpretacion)

@bp.route('/resultados/graficos', methods=['GET', 'POST'])
def resultados_graficos():
    import glob
    import seaborn as sns
    from matplotlib import pyplot as plt

    resultados = session.get('resultados')

    if not resultados or 'datos_predicciones' not in resultados:
        flash("No hay datos disponibles para graficar.")
        return redirect(url_for('main.index'))

    columnas = session.get('columnas_pred', [])
    df = pd.DataFrame(resultados['datos_predicciones'])
    grafico_generado = None
    graficos_brutos = []

    static_dir = os.path.join('app', 'static')
    os.makedirs(static_dir, exist_ok=True)

    # 🧹 Borrar gráficos brutos anteriores
    for f in glob.glob(os.path.join(static_dir, "bruto_*.png")):
        try:
            os.remove(f)
        except Exception as e:
            print(f"No se pudo eliminar {f}: {e}")

    # 🧠 Diccionario para nombres legibles
    nombres_legibles = {
        'sexo': 'sexo',
        'nivel_escolar': 'trabaja',
        'edad': 'edad',
        'etiqueta_riesgo': 'etiqueta_riesgo',
        'abandono_predicho': 'abandono_predicho',
        'probabilidad_abandono': 'probabilidad_abandono',
        'promedio': 'promedio',
        'interes_terminar': 'interes_terminar',
        'motivacion': 'motivacion',
        'estres': 'Estrés Académico',
    }

    # 📊 Gráficos automáticos simples
    columnas_brutas = ['sexo', 'nivel_escolar', 'edad', 'etiqueta_riesgo', 'abandono_predicho']

    for col in columnas_brutas:
        if col in df.columns:
            try:
                plt.figure(figsize=(7, 5))
                df[col].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
                titulo = nombres_legibles.get(col, col.replace("_", " ").capitalize())
                plt.title(f"Distribución de {titulo}")
                plt.xlabel(titulo)
                plt.ylabel("Frecuencia")
                plt.xticks(rotation=45)
                plt.tight_layout()

                filename = f"bruto_{col}_{uuid.uuid4().hex[:6]}.png"
                ruta = os.path.join(static_dir, filename)
                plt.savefig(ruta)
                plt.close()

                graficos_brutos.append({'titulo': f"Distribución de {titulo}", 'archivo': filename})
            except Exception as e:
                print(f"Error al generar gráfico de {col}: {e}")

    # 📊 Gráfico: Distribución de probabilidad de abandono
    if 'probabilidad_abandono' in df.columns:
        try:
            plt.figure(figsize=(7, 5))
            sns.histplot(df['probabilidad_abandono'], bins=20, kde=True, color='orchid')
            plt.title("Distribución de Probabilidad de Abandono")
            plt.xlabel("Probabilidad")
            plt.ylabel("Frecuencia")
            plt.tight_layout()
            filename = f"bruto_probabilidad_{uuid.uuid4().hex[:6]}.png"
            plt.savefig(os.path.join(static_dir, filename))
            plt.close()
            graficos_brutos.append({'titulo': "Distribución de Probabilidad", 'archivo': filename})
        except Exception as e:
            print(f"Error al generar gráfico de probabilidad: {e}")

    # 📊 Gráficos de relación con abandono_predicho
    relaciones = ['promedio', 'interes_terminar', 'motivacion', 'estres']
    for col in relaciones:
        if col in df.columns and 'abandono_predicho' in df.columns:
            try:
                plt.figure(figsize=(7, 5))
                sns.boxplot(data=df, x='abandono_predicho', y=col, palette='pastel')
                titulo_y = nombres_legibles.get(col, col.replace("_", " ").capitalize())
                plt.title(f"{titulo_y} según Abandono Predicho")
                plt.xlabel("Abandono Predicho")
                plt.ylabel(titulo_y)
                plt.tight_layout()
                filename = f"bruto_{col}_vs_abandono_{uuid.uuid4().hex[:6]}.png"
                plt.savefig(os.path.join(static_dir, filename))
                plt.close()
                graficos_brutos.append({'titulo': f"{titulo_y} vs Abandono", 'archivo': filename})
            except Exception as e:
                print(f"Error al generar gráfico {col} vs abandono: {e}")

    # 📥 POST: Gráfico manual generado por usuario
    if request.method == 'POST':
        tipo = request.form.get('tipo')
        variables = request.form.getlist('variables')

        if tipo and len(variables) >= 1:
            filename = f'grafico_{tipo}_{uuid.uuid4().hex[:6]}.png'
            ruta = os.path.join(static_dir, filename)

            try:
                plt.figure(figsize=(8, 6))

                if tipo == 'histograma' and len(variables) == 1:
                    df[variables[0]].hist(color='mediumpurple', edgecolor='black')

                elif tipo == 'barras' and len(variables) == 2:
                    df.groupby(variables[0])[variables[1]].mean().plot(kind='bar', color='salmon')

                elif tipo == 'dispersion' and len(variables) == 2:
                    sns.scatterplot(data=df, x=variables[0], y=variables[1], hue=variables[1], palette='coolwarm')

                elif tipo == 'caja' and len(variables) == 1:
                    sns.boxplot(data=df, y=variables[0], color='lightgreen')

                else:
                    flash("Parámetros inválidos para el tipo de gráfico seleccionado.")
                    return redirect(url_for('main.resultados_graficos'))

                plt.title(f"Gráfico: {tipo.capitalize()}")
                plt.tight_layout()
                plt.savefig(ruta)
                plt.close()
                grafico_generado = filename

            except Exception as e:
                flash(f"Error al generar el gráfico: {e}")
                return redirect(url_for('main.resultados_graficos'))

    return render_template('resultados_graficos.html',
                           columnas=columnas,
                           grafico_generado=grafico_generado,
                           graficos_brutos=graficos_brutos)


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

@bp.route('/resultados/datos_json')
def resultados_datos_json():
    resultados = session.get('resultados')
    if not resultados or 'datos_predicciones' not in resultados:
        flash("No hay resultados para mostrar.")
        return redirect(url_for('main.index'))

    df_pred = pd.DataFrame(resultados['datos_predicciones'])
    datos_json = df_pred.to_dict(orient='records')
    return json.dumps(datos_json)
