from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.layers import BatchNormalization, ReLU, Dense
from tensorflow.keras.layers import Conv1D, Input
from .metrics import snr_metric
from .AbstractModel import AbstractModel


class UnidimensionalResCNN(AbstractModel):

    def __init__(self, input_shape):
        super().__init__(input_shape)

    def __resblock(self, dimension, model):
        resblock = Conv1D(32, dimension, activation=keras.activations.relu, padding='causal')(model)
        resblock = BatchNormalization()(resblock)
        resblock = ReLU()(resblock)

        resblock = Conv1D(16, dimension, activation=keras.activations.relu, padding='causal')(resblock)
        resblock = BatchNormalization()(resblock)
        resblock = ReLU()(resblock)

        resblock = Conv1D(32, dimension, activation=keras.activations.relu, padding='causal')(resblock)
        resblock = BatchNormalization()(resblock)
        resblock = ReLU()(resblock)

        add = Add()([model, resblock])

        resblock = Conv1D(32, dimension, activation=keras.activations.relu, padding='causal')(add)
        resblock = BatchNormalization()(resblock)
        resblock = ReLU()(resblock)

        resblock = Conv1D(16, dimension, activation=keras.activations.relu, padding='causal')(resblock)
        resblock = BatchNormalization()(resblock)
        resblock = ReLU()(resblock)

        resblock = Conv1D(32, dimension, activation=keras.activations.relu, padding='causal')(resblock)
        resblock = BatchNormalization()(resblock)
        resblock = ReLU()(resblock)

        resblock = Add()([resblock, add])
        return resblock

    def _AbstractModel__build_model(self, window_length):
        x = Input(shape=(window_length, 1))

        my_conv = Conv1D(32, 5, activation=keras.activations.relu, padding='causal')(x)
        my_conv = BatchNormalization()(my_conv)
        my_conv = ReLU()(my_conv)

        resblock1 = self.__resblock(3, my_conv)

        resblock2 = self.__resblock(5, my_conv)

        resblock3 = self.__resblock(7, my_conv)

        my_conv = Concatenate()([resblock1, resblock2, resblock3])

        my_conv = Conv1D(32, 1, activation=keras.activations.relu, padding='causal')(my_conv)
        my_conv = BatchNormalization()(my_conv)
        my_conv = ReLU()(my_conv)

        my_conv = Dense(32, activation=keras.activations.tanh)(my_conv)

        my_conv = Dense(1, activation=keras.activations.linear)(my_conv)

        model = Model(inputs=x, outputs=my_conv)

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.mean_squared_error,
                      metrics=[snr_metric])

        return model