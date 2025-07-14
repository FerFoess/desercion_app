# utils.py
import pandas as pd
from fpdf import FPDF
from io import BytesIO
import os

def generate_pdf(resultados, items):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "📊 Reporte de Deserción Escolar", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    if 'resumen' in items:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "✅ Resumen de Clústeres:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "Incluye resumen de clústeres seleccionado", ln=True)
        pdf.ln(5)

    if 'datos' in items:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "📄 Datos Clusterizados:", ln=True)
        pdf.set_font("Arial", '', 12)
        # Solo muestra primeros registros resumidos
        for row in resultados[:3]:
            txt = ', '.join(f"{k}: {v}" for k, v in row.items())
            pdf.multi_cell(0, 10, txt)
        pdf.ln(5)

    if 'grafico1' in items:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "📈 Gráfico 1:", ln=True)
        grafico1_path = os.path.join('desercion_app', 'app', 'static', 'grafica1.png')
        if os.path.exists(grafico1_path):
            pdf.image(grafico1_path, w=100)
        pdf.ln(5)

    if 'grafico2' in items:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "📊 Gráfico 2:", ln=True)
        grafico2_path = os.path.join('desercion_app', 'app', 'static', 'grafica2.png')
        if os.path.exists(grafico2_path):
            pdf.image(grafico2_path, w=100)
        pdf.ln(5)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

def generate_excel(resultados, items):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df = pd.DataFrame(resultados)

        # Hoja con datos completos
        if 'datos' in items:
            df.to_excel(writer, sheet_name='Datos Clusterizados', index=False)

        # Hoja resumen (simulado)
        if 'resumen' in items:
            resumen = df.describe().reset_index()
            resumen.to_excel(writer, sheet_name='Resumen', index=False)

        # Hoja gráficos (puede incluir nombres de archivos o algo simbólico)
        if 'grafico_tabla' in items:
            graf = pd.DataFrame({'Gráfico': ['Gráfico 1', 'Gráfico 2']})
            graf.to_excel(writer, sheet_name='Graficos', index=False)
    output.seek(0)
    return output
