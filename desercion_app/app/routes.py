from flask import Blueprint, render_template, request
from .forms import StudentForm
from .models import predict_dropout
from .utils import generate_pdf, generate_excel

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET', 'POST'])
def form():
    form = StudentForm()
    if form.validate_on_submit():
        data = form.data
        prediction, graph_path, table = predict_dropout(data)
        return render_template('results.html', prediction=prediction, graph=graph_path, table=table)
    return render_template('index.html')

@bp.route('/export/pdf')
def export_pdf():
    generate_pdf()
    return "PDF generado"

@bp.route('/export/excel')
def export_excel():
    generate_excel()
    return "Excel generado"
