
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Layer, Dropout, Input, Dense
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


url = 'C:/Users/johnn/OneDrive/Documentos/Deep Learning/Deep Learning com Python de A a Z - O Curso Completo/base_dados/wdbc.data'
cancer = datasets.load_breast_cancer()
dados = pd.read_csv(url, header=None)


entradas = dados.iloc[:,2:]
entradas.columns = cancer.feature_names
saidas = dados.iloc[:,1]

X = dados.iloc[:,2:].values
y = dados.iloc[:,1].replace('M', 1).replace('B', 0).values


# Divisão da base de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Rede Neural
classificador = Sequential()
classificador.add(Input(shape=30, name='Camada_entrada'))
classificador.add(Dense(units=16, activation='relu', kernel_initializer='uniform', name='Camada_oculta0'))
classificador.add(Dropout(rate=0.2, name='Camada_dropout0'))
classificador.add(Dense(units=1, activation='sigmoid', name='Camada_saida'))

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(X_train, y_train, batch_size=10, epochs=100)
previsoes = classificador.predict(X_test) > 0.7
previsoes = previsoes.astype(int)

# Métricas de avaliação 
accuracia1 = classificador.evaluate(X_test, y_test)
accuracia2 = accuracy_score(y_test, previsoes)

grafico = ConfusionMatrixDisplay(confusion_matrix(y_test, previsoes))
grafico.plot()

