import pandas as pd
from fpdf import FPDF
import os

def generate_pdf(resultados, items):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Reporte de Deserción Escolar", ln=True)

    # Añadir información según items seleccionados
    if 'resumen' in items:
        pdf.cell(200, 10, txt="Incluye resumen de clústeres", ln=True)
    if 'datos' in items:
        pdf.cell(200, 10, txt="Incluye datos clusterizados", ln=True)
    if 'grafico1' in items:
        pdf.cell(200, 10, txt="Incluye gráfico 1", ln=True)
        # Agrega imagen (debe estar en static)
        grafico1_path = os.path.join('app', 'static', 'grafica1.png')
        if os.path.exists(grafico1_path):
            pdf.image(grafico1_path, w=100)
    if 'grafico2' in items:
        pdf.cell(200, 10, txt="Incluye gráfico 2", ln=True)
        grafico2_path = os.path.join('app', 'static', 'grafica2.png')
        if os.path.exists(grafico2_path):
            pdf.image(grafico2_path, w=100)

    # Ejemplo: agregar datos reales (solo primeros 3 registros)
    if resultados and isinstance(resultados, list):
        pdf.cell(200, 10, txt="Ejemplo de datos:", ln=True)
        for idx, row in enumerate(resultados[:3]):
            pdf.cell(200, 10, txt=str(row), ln=True)

    output_dir = os.path.join('desercion_app', 'app', 'static', 'temp')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'reporte.pdf')
    pdf.output(output_path)

    return output_path

def generate_excel(resultados, items):
    df = pd.DataFrame(resultados)

    # Filtrar columnas si quieres según items (ejemplo)
    # if items:
    #     columnas = [col for col in df.columns if col in items]
    #     df = df[columnas]

    output_dir = os.path.join('desercion_app', 'app', 'static', 'temp')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'datos.xlsx')
    df.to_excel(output_path, index=False)

    return output_path