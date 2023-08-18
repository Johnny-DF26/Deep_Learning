from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


classificador = Sequential()
classificador.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))
classificador.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))
classificador.add(Flatten())

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7,
                                         horizontal_flip=True, shear_range=0.2,
                                         height_shift_range=0.07, zoom_range=0.2)

gerador_teste = ImageDataGenerator(rescale=1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set', target_size=(64,64),
                                                           class_mode='binary', batch_size=32)

base_teste = gerador_teste.flow_from_directory('dataset/test_set', target_size=(64,64),
                                                           class_mode='binary', batch_size=32)

classificador.fit(base_treinamento, steps_per_epoch=4000/32, epochs=10, validation_data=base_teste,
                  validation_steps=1000/32)


novo = gerador_teste.flow_from_directory(directory='dataset/novo', target_size=(61,64),
                                         class_mode='binary', batch_size=32)

novo = classificador.predict(novo)

