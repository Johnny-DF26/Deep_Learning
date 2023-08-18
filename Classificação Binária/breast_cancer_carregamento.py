import numpy as np
import pandas as pd
from keras.models import model_from_json


with open('classificador_breast.json', 'r') as f:
    arquivo = f.read()

classificador = model_from_json(arquivo)
classificador.load_weights('classificador_breast.h5')


novo2 = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                   0.20, 0.05, 1098, 0.87, 4508, 145.2, 0.005, 0.04, 0.05, 0.015,
                   0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])

previsao = classificador.predict(novo2) > 0.5

previsores = pd.read_csv('base_dados/previsores_breast_cancer.csv')
classe = pd.read_csv('base_dados/classe_breast_cancer.csv')

classificador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
resultado = classificador.evaluate(previsores, classe)










