from abc import ABC, abstractmethod
from tensorflow import keras


class AbstractModel(ABC):
    def __init__(self, input_shape):
        self.model = self.__build_model(input_shape)
        super().__init__()

    def save(self, path):
        return self.model.sample_weights(path)

    def load(self, path):
        return self.model.load_weights(path)

    def train(self, train_data, train_labels, test_data, test_labels, epochs=10):
        return self.model.fit(x=train_data, y=train_labels, epochs=epochs, validation_data=(test_data, test_labels))

    def show_summary(self):
        self.model.summary()

    def plot_model(self, filename):
        keras.utils.plot_model(self.model, filename, show_shapes=True, show_layer_names=False)

    def evaluate(self, test):
        return self.model.predict(test, verbose=1)

    @abstractmethod
    def __build_model(self, window_length):
        pass
