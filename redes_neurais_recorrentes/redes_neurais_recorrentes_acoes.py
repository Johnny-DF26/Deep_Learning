import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import MinMaxScaler


base = pd.read_csv('HCTR11.SA.csv')
base_teste = base.iloc[-20:, :]
base_treinamento = base.iloc[:-20, :]

base_treinamento = base_treinamento.dropna()

base_treinamento = base_treinamento.iloc[:,1:2].values
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizado = normalizador.fit_transform(base_treinamento)

previsores = []
preco_real = []

for i in range(90, 909):
    previsores.append(base_treinamento_normalizado[i-90:i, 0])
    preco_real.append(base_treinamento_normalizado[i,0])

previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

regressor = Sequential(
    [
         layers.Input(shape=(previsores.shape[1],1)),
         layers.LSTM(units=100, return_sequences=True),
         layers.Dropout(0.3),
         layers.LSTM(units=50, return_sequences=True),
         layers.Dropout(0.3),
         layers.LSTM(units=50),
         layers.Dropout(0.3),
         layers.Dense(units=1, activation='linear')
    ])
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size=32, epochs=100)

preco_real_teste = base_teste.iloc[:, 1:2].values

base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
base_completa = base_completa.iloc[:-1]
entradas = base_completa[len(base_completa) - len(base_completa)-90: ].values

entradas = entradas.reshape(-1,1)
entradas = normalizador.transform(entradas)
X_teste = regressor.predict(entradas)










