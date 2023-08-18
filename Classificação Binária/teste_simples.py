
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
import matplotlib.pyplot as plt

previsores = pd.read_csv('base_dados/previsores_breast_cancer.csv')
classe = pd.read_csv('base_dados/classe_breast_cancer.csv')

X_treino, X_teste, y_treino, y_teste = train_test_split(previsores, classe, test_size=0.2)

def criaRede(X, y, epocas=100):
    classificador = Sequential()
    classificador.add(Input(shape=30, name='camada_entrada'))
    classificador.add(Dense(units=(30+1/2), activation='relu', kernel_initializer='normal', name='camada_oculta1'))
    classificador.add(Dropout(rate=0.2, name='camada_dropout1'))
    classificador.add(Dense(units=1, activation='sigmoid', name='camada_saida'))
    
    classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics='binary_accuracy')
    classificador.fit(X, y, batch_size=10, epochs=epocas)
    
    return classificador
    
classficador = criaRede(X_treino, y_treino, 100)

erro, acuracia = classficador.evaluate(X_teste, y_teste)
previsoes = classficador.predict(X_teste) > 0.7

previsoes = previsoes.astype(int)


# Avaliaçao

from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay, roc_curve, ConfusionMatrixDisplay, auc
from sklearn.metrics import classification_report
# Curva ROC
fpr, tpr, thresholds = roc_curve(y_teste, previsoes)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Modelo')
display.plot()
plt.plot([0,1], '--')

# Matriz de Confusão
matriz_confusao = confusion_matrix(y_teste, previsoes)
display2 = ConfusionMatrixDisplay(matriz_confusao)
display2.plot()

clf_report = classification_report(y_teste, previsoes)

# ---------------------------------------------------------------------------------------------------------








