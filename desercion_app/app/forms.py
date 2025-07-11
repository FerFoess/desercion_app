from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired

class StudentForm(FlaskForm):
    edad = IntegerField('Edad', validators=[DataRequired()])
    promedio = IntegerField('Promedio', validators=[DataRequired()])
    ingreso_familiar = SelectField('Ingreso familiar', choices=[('bajo', 'Bajo'), ('medio', 'Medio'), ('alto', 'Alto')])
    submit = SubmitField('Evaluar')
