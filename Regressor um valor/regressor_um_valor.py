import pandas as pd

dados = pd.read_csv('autos.csv', encoding='iso-8859-1')
dados.columns

# Colunas a excluir
colunas = ['dateCrawled', 'name', 'seller', 'dateCreated', 'lastSeen', 'offerType',
           'nrOfPictures']
dados.drop(colunas, axis=1, inplace=True)


# Valores nulos
dados.isnull().sum()

dados['vehicleType'].value_counts() # kleinwagen
dados['gearbox'].value_counts() # manuell
dados['model'].value_counts() # golf
dados['fuelType'].value_counts() # benzin
dados['notRepairedDamage'].value_counts() # nein

valores_nulos_a_inserir = {'vehicleType': 'kleinwagen', 'gearbox': 'manuell',
                           'model': 'golf', 'fuelType': 'benzin',
                           'notRepairedDamage': 'nein'}

dados.fillna(valores_nulos_a_inserir, inplace=True)
dados.isnull().sum()

dados = dados.loc[dados['price'] > 100]
dados = dados.loc[dados['price'] < 350000]

dados.to_csv('autos_clean.csv', index=False)
# Divisão Base de dados
X = dados.iloc[:,1:].values
y = dados.iloc[:,0].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

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
import keras
from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential()
regressor.add(Dense(units=159, activation='relu', input_dim=317))
regressor.add(Dense(units=159, activation='relu'))
regressor.add(Dense(units=1, activation='linear'))
regressor.compile(optimizer='adam', loss='mae', metrics=['mae'])

regressor.fit(X, y, batch_size=300, epochs=100)












