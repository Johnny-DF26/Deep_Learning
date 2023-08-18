import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


(X_treino, y_treino), (X_teste, y_teste) = keras.datasets.mnist.load_data()

X_treino = X_treino.reshape(X_treino.shape[0], 28,28,1).astype('float32')
X_teste = X_teste.reshape(X_teste.shape[0], 28,28,1).astype('float32')

X_treino /= 255
X_teste /= 255

y_treino = np_utils.to_categorical(y_treino, 10)
y_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()
classificador.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
classificador.add(BatchNormalization())
classificador.add(MaxPool2D(pool_size=(2,2)))

classificador.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
classificador.add(BatchNormalization())
classificador.add(MaxPool2D(pool_size=(2,2)))
classificador.add(Flatten())

classificador.add(Dense(units=64, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=64, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=10, activation='softmax'))

classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classificador.fit(X_treino, y_treino, batch_size=128, epochs=5, validation_data=(X_teste, y_teste))

gerador_treinamento = ImageDataGenerator(rotation_range=7, horizontal_flip=True,
                                         shear_range=0.2, height_shift_range=0.07,
                                         zoom_range=0.2)

gerador_teste = ImageDataGenerator()
base_treinamento = gerador_treinamento.flow(X_treino, y_treino, batch_size=128)
base_teste = gerador_teste.flow(X_teste, y_teste, batch_size=128)
classificador.fit_generator(base_treinamento, steps_per_epoch=60000/128, epochs=5, 
                            validation_data= base_teste, validation_steps=10000/128)





