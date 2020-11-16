from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed, LSTM

from EEGArtifactRemoval.models.metrics import snr_metric


class SimpleLSTM:

    def __init__(self, input_shape):
        self.__model = self.__build_model(input_shape)

    def train(self, train_data, train_labels, test_data, test_labels, epochs=10):
        return self.__model.fit(x=train_data, y=train_labels, epochs=epochs, validation_data=(test_data, test_labels))

    def show_summary(self):
        self.__model.summary()

    def plot_model(self, filename):
        keras.utils.plot_model(self.__model, filename, show_shapes=True, show_layer_names=False)

    def evaluate(self, test):
        return self.__model.predict(test, verbose=1)

    def __build_model(self, window_length):
        shape = (window_length, 1)
        model = Sequential()
        model.add(LSTM(100, input_shape=shape, return_sequences=True, activation=keras.activations.sigmoid))
        model.add(TimeDistributed(Dense(50)))
        model.add(Dense(32, activation=keras.activations.tanh))
        model.add(Dense(1, activation=keras.activations.linear))
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.mean_squared_error,
                      metrics=[snr_metric])
        return model