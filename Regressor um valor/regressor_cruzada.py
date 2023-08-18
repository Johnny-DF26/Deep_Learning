import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import keras
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score

dados = pd.read_csv('autos_clean.csv')

# Divisão Base de dados
X = dados.iloc[:,1:].values
y = dados.iloc[:,0].values


encoder = LabelEncoder()
X[:,0] = encoder.fit_transform(X[:,0])
X[:,1] = encoder.fit_transform(X[:,1])
X[:,3] = encoder.fit_transform(X[:,3])
X[:,5] = encoder.fit_transform(X[:,5])
X[:,8] = encoder.fit_transform(X[:,8])
X[:,9] = encoder.fit_transform(X[:,9])
X[:,10] = encoder.fit_transform(X[:,10])


ct = ColumnTransformer([("one_hot", OneHotEncoder(), [0,1,3,5,8,9,10])], remainder = 'passthrough')
X = ct.fit_transform(X).toarray()

# Rede Neural

def criaRede():
    regressor = Sequential()
    regressor.add(Dense(units=159, activation='relu', input_dim=317))
    regressor.add(Dense(units=159, activation='relu'))
    regressor.add(Dense(units=1, activation='linear'))
    regressor.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return regressor

regressor = KerasRegressor(model= criaRede, batch_size=300, epochs=10)

cross = cross_val_score(estimator=regressor, X=X, y=y, scoring='neg_mean_absolute_error', cv=5)












