import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA_PATH = "./cifar-10-batches-py/"
data_augmentation = False
modelName = "vggNet"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def show_first_samples(x_train, y_train, labels_name):
    
    for i in range(4):
        imgs = x_train[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB",(i0,i1,i2))
        plt.imshow(img)
        print(labels_name[y_train[i]])
        plt.show()

def create_model():
    model = Sequential()

    model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        data_format = "channels_first",
        padding = "same",
        input_shape = (3, 32, 32)
        ))
    model.add(Activation('relu'))

    model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        data_format = "channels_first",
        padding = "same",
        input_shape = (3, 32, 32)
        ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (2,2), data_format = "channels_first", padding = "same"))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 128,
        kernel_size = (3,3),
        data_format = "channels_first",
        padding = "same",
    ))
    model.add(Activation('relu'))

    model.add(Conv2D(
        filters = 128,
        kernel_size = (3,3),
        data_format = "channels_first",
        padding = "same",
    ))
    

    model.add(Conv2D(
        filters = 128,
        kernel_size = (3,3),
        data_format = "channels_first",
        padding = "same",
    ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (2,2), data_format = "channels_first", padding = "same"))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss = sparse_categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])
    
    return model

def load_data():
    data_first = unpickle(DATA_PATH + "data_batch_1")
    data_x = data_first[b'data']
    data_y = data_first[b'labels']
    for i in range(2,6):
        data = unpickle(DATA_PATH + f"data_batch_{i}")
        data_x = np.vstack((data_x, data[b'data']))
        data_y += data[b'labels']
    return data_x, data_y


if __name__ == "__main__":
    x_train, y_train = load_data() #load the train data cifar

    data_test = unpickle(DATA_PATH + "test_batch") # load the test data
    x_val, y_val = data_test[b'data'], data_test[b'labels'] 

    data_info = unpickle(DATA_PATH + "batches.meta") #load the data infos
    labels_name = data_info[b'label_names']

    x_train = x_train / 255.0
    x_val = x_val / 255.0

    x_train = x_train.reshape(-1, 3, 32, 32)
    x_val = x_val.reshape(-1, 3, 32, 32)

    y_train = np.array(y_train)
    y_val = np.array(y_val)

    model = create_model()

    print(model.summary())
    plot_model(model, f"{modelName}_log/{modelName}.png")

    #show_first_samples(x_train, y_train, labels_name)

    if data_augmentation:
        aug = ImageDataGenerator(width_shift_range = 0.2, height_shift_range = 0.2, horizontal_flip = True)
        aug.fit(x_train)
        gen = aug.flow(x_train, y_train, batch_size=128)
        history = model.fit_generator(generator=gen, 
                                 steps_per_epoch=50000/128, 
                                 epochs=30, 
                                 validation_data=(x_val, y_val),
                                 callbacks=[
                                    EarlyStopping(monitor="val_accuracy", patience=2),
                                    TensorBoard(log_dir=f"{modelName}_log", histogram_freq=1)
                                    ])
    else:
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                epochs=30,
                batch_size=128,
                callbacks=[
                    EarlyStopping(monitor="val_accuracy", patience=2),
                    TensorBoard(log_dir=f"{modelName}_log", histogram_freq=1)
                    ])
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f'./{modelName}_log/{modelName}_accuracy.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f'./{modelName}_log/{modelName}_loss.png')
    plt.show()
