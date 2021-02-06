from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed, LSTM
from . import AbstractModel
from .metrics import snr_metric


class SimpleLSTM(AbstractModel):

    def __init__(self, input_shape):
        super().__init__(input_shape)

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