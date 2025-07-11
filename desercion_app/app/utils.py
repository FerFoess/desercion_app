import pandas as pd
from fpdf import FPDF

def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Reporte de Deserción Escolar", ln=True)
    pdf.output("reporte.pdf")

def generate_excel():
    df = pd.DataFrame({'Nombre': ['Juan'], 'Edad': [18]})
    df.to_excel('datos.xlsx', index=False)
