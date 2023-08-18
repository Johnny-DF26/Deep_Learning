from sklearn.datasets import load_iris
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

previsores, classe = load_iris(return_X_y=True, as_frame=True)
classe_dummy = np_utils.to_categorical(classe)

X_treino, X_teste, y_treino, y_teste = train_test_split(previsores, classe_dummy,
                                                        test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units=4, activation='relu', input_shape=(4,)))
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dense(units=3, activation='softmax'))
classificador.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

classificador.fit(previsores, classe_dummy, batch_size=10, epochs=1000)

resultado = classificador.evaluate(X_teste, y_teste)
previsoes = (classificador.predict(X_teste) > 0.5).astype(int)

classe_teste2 = [np.argmax(t) for t in y_teste]
previsoes2 = [np.argmax(t) for t in previsoes]
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


matriz = confusion_matrix(classe_teste2, previsoes2)




