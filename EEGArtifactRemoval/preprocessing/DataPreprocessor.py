import numpy as np
import math


class DataPreprocessor:

    def __init__(self,window_length,step,separation_factor=0.8):
        self.window_length = window_length
        self.separation_factor = separation_factor
        self.step = step

    def separate_data(self,clean, noisy, factor=0.8, save=False):
        pivot = math.floor(clean.shape[0] * factor)
        clean_train = clean[0:pivot]
        noisy_train = noisy[0:pivot]
        clean_test = clean[pivot:]
        noisy_test = noisy[pivot:]
        return (noisy_train,clean_train,noisy_test, clean_test)

    def __moving_window(self,x, length, step=1):
        windows = []
        for i in range(x.shape[0]):
            start = (i * step) + 1
            end = start + length
            if end >= x.shape[0]:
                return np.array(windows).reshape(-1, self.window_length, 1)
            windows.append(x[start:end])

    def to_list_of_time_windows(self,signal_list, window_length, step):
        time_windows = self.__moving_window(signal_list[0], length=window_length, step=step)
        new_signal_list = np.delete(signal_list, 0, 0)
        for signal in new_signal_list:
            time_window = self.__moving_window(signal, length=window_length, step=step)
            time_windows = np.append(time_windows, time_window, axis=0)
        print(time_windows.shape)
        return time_windows