import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


previsores = pd.read_csv('base_dados/previsores_breast_cancer.csv')
classe = pd.read_csv('base_dados/classe_breast_cancer.csv')


def criaRede(neuronios, ativacao, kernel_initializer, otimizador, perda):
    classificador = Sequential()
    classificador.add(Input(shape=30))
    classificador.add(Dense(units=neuronios, activation=ativacao, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=1, activation='sigmoid'))
    classificador.compile(optimizer=otimizador, loss=perda, metrics=['accuracy'])
    
    return classificador


classificador = KerasClassifier(build_fn=criaRede)
param = {'neuronios': range(15,20), 'batch_size': range(8, 20), 'epochs': range(100,300),
         'ativacao': ['relu', 'tanh', 'selu'], 'kernel_initializer': ['random_normal', 'random_uniform'],
         'otimizador': ['Adamax', 'adam', 'sgd'], 'perda': ['hinge', 'binary_crossentropy']}
random = RandomizedSearchCV(estimator=classificador, param_distributions=param,
                            n_iter=5, scoring='accuracy', cv=10)

random.fit(previsores, classe)
melhor_parametro = random.best_params_
melhor_score = random.best_score_




















