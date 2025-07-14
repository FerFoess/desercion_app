from fpdf import FPDF
import pandas as pd
import os
from datetime import datetime

EXPORT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def generate_pdf(datos, items):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Resultados de la Clusterización", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Secciones incluidas: " + ", ".join(items), ln=True)

    # Crear carpeta si no existe
    os.makedirs(EXPORT_FOLDER, exist_ok=True)

    # Ruta final del archivo
    filename = f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = os.path.join(EXPORT_FOLDER, filename)
    pdf.output(output_path)

    return output_path

def generate_excel(datos, items):
    df = pd.DataFrame(datos)

    os.makedirs(EXPORT_FOLDER, exist_ok=True)

    filename = f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    output_path = os.path.join(EXPORT_FOLDER, filename)
    df.to_excel(output_path, index=False)

    return output_path
