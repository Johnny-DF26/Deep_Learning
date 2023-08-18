from sklearn.datasets import load_iris
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score

previsores, classe = load_iris(return_X_y=True, as_frame=True)
classe_dummy = np_utils.to_categorical(classe)

def criaRede():
    classificador = Sequential()
    classificador.add(Dense(units=4, activation='relu', input_shape=(4,)))
    classificador.add(Dense(units=3, activation='softmax'))
    classificador.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics='categorical_accuracy')
    return classificador

classificador = KerasClassifier(build_fn=criaRede, batch_size=10, epochs=1000)

cross = cross_val_score(estimator=classificador, X=previsores, y=classe,
                        scoring='accuracy', cv=5)

media = cross.mean()



