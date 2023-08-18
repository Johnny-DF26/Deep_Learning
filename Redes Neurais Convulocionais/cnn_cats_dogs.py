from keras.models import Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
import numpy as np
from keras import utils

modelo1 = Sequential(
    [   
        layers.Rescaling(1./255),
        layers.Input(shape=(64,64,3)),
        layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Flatten(),
        
        layers.Dense(units=128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(units=128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(units=1, activation='sigmoid')
    ])
    
modelo1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7,
                                         horizontal_flip=True, shear_range=0.2,
                                         height_shift_range=0.07, zoom_range=0.2)
gerador_teste = ImageDataGenerator(rescale=1./255)
base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set', target_size=(64,64),
                                                           class_mode='binary', batch_size=32)
base_teste = gerador_teste.flow_from_directory('dataset/test_set', target_size=(64,64),
                                                           class_mode='binary', batch_size=32)
modelo1.fit(base_treinamento, steps_per_epoch=4000/32, epochs=15,
            validation_data=base_teste, validation_steps= 1000/32)


def predicao_novo_imagem(data):
    import numpy as np
    from keras import utils
    imagem_teste = utils.load_img(data, target_size=(64,64))
    imagem_teste = utils.img_to_array(imagem_teste)
    imagem_teste /= 255
    imagem_teste = np.expand_dims(imagem_teste, 0)
    predicao = modelo1.predict(imagem_teste)
    
    if predicao < 0.5:
        print(f'Com {predicao[0][0]:.2f}% de confiança: \033[Cachorro\033[0m')
    else:
        print(f'Com {predicao[0][0]:.2f}% de confiança: \033[31mGato\033[0m')


predicao_novo_imagem('dataset/novo/cat.3555.jpg')

modelo1.save('model_dog.keras')
modelo1.save_weights('pesos_cnn.h5')


