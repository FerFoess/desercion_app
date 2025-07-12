from flask import Blueprint, render_template, request
import pandas as pd

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET', 'POST'])
def index():
    data = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_excel(file)
            data = df
    return render_template('index.html', data=data)
