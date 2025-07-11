import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def predict_dropout(data):
    X = pd.DataFrame([[data['edad'], data['promedio']]], columns=['edad', 'promedio'])
    model = LogisticRegression()
    model.fit([[18, 7], [22, 5]], [0, 1])  # Entrenamiento ficticio
    prediction = model.predict_proba(X)[0][1]

    plt.figure()
    plt.bar(['Probabilidad de Abandono'], [prediction])
    path = 'app/static/prediccion.png'
    plt.savefig(path)

    return prediction, path, X.to_html()
