import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


previsores = pd.read_csv('base_dados/previsores_breast_cancer')
classe = pd.read_csv('base_dados/classe_breast_cancer')

def criaRede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Input(shape=30))
    classificador.add(Dense(units= neurons, activation=activation,
                            kernel_initializer=kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=1, activation='sigmoid'))
    classificador.compile(optimizer=optimizer, loss=loos, metrics=['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn= criaRede)
parametros = {'batch_size': range(10,30), 'epochs': range(50,110), 'optimizer': ['adam', 'sgd'],
              'loos': ['binary_crossentropy', 'hinge'], 'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'], 'neurons': range(8,16)}


grid_random = RandomizedSearchCV(estimator= classificador, param_distributions= parametros, 
                                 n_iter=10, scoring='accuracy', cv=10)

grid_random.fit(X=previsores, y=classe)
melhores_parametros = grid_random.best_params_
melhor_score = grid_random.best_score_

