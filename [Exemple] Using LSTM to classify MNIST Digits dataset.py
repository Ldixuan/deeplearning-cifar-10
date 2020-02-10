import matplotlib.pyplot as plt
from tensorflow.keras.activations import *
from tensorflow.keras.datasets import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model


class PrintTrueTrainMetricsAtEpochEnd(Callback):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.x_train, self.y_train, batch_size=8192)
        print(f"Le Vrai loss du train : {loss}")
        print(f"La Vrai acc du train : {acc}")


def create_model():
    input_tensor = Input((28, 28))

    lstm_tensor = Bidirectional(LSTM(64, return_sequences=True))(input_tensor)
    lstm_tensor = Bidirectional(LSTM(64, return_sequences=True))(lstm_tensor)
    lstm_tensor = Bidirectional(LSTM(64, return_sequences=False))(lstm_tensor)

    dense_tensor = Dense(10, activation=softmax)(lstm_tensor)

    model = Model(input_tensor, dense_tensor)

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=[sparse_categorical_accuracy])

    return model


if __name__ == "__main__":

    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    (x_train, y_train), (x_val, y_val) = fashion_mnist.load_data()
    for i in range(4):
        plt.imshow(x_train[i])
        print(y_train[i])
        plt.show()

    m = create_model()

    print(m.summary())
    plot_model(m, "test_lstm.png")

    m.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=100,
          batch_size=8192,
          callbacks=[PrintTrueTrainMetricsAtEpochEnd(x_train, y_train)]
          )
