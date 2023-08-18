import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

previsores = pd.read_csv('base_dados/previsores_breast_cancer')
classe = pd.read_csv('base_dados/classe_breast_cancer')


def cria_rede():
    classificador = Sequential()
    classificador.add(Input(shape=30))
    classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    classificador.add(Dropout(rate=0.2))
    classificador.add(Dense(units=1, activation='sigmoid'))
    classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])    
    return classificador

classificador = KerasClassifier(model=cria_rede, batch_size=10, epochs=100)
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe,
                             scoring='accuracy', cv=10)


media = resultados.mean()
desvio = resultados.std()




