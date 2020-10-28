import math
import random
import numpy as np
from scipy import stats
from scipy.signal import butter, lfilter, resample


class ArtifactGenerator:

    def __init__(self, sample_rate, snr_db):
        self.sample_rate = sample_rate
        self.snr_db = snr_db

    def __rms(self, signal):
        return np.sqrt(np.mean(signal ** 2))

    def __gaussian_noise(self, length, strength=1):
        return np.random.normal(0, 1, length) * strength

    def __butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def __butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def __pos_gen(self, window_shape, blanc_shape, max_pos, start=0):
        min_window, max_window = window_shape
        min_blanc, max_blanc = blanc_shape
        end = 0
        while end < max_pos:
            h = random.randint(min_window, max_window)
            final = start + h
            end = final if final < max_pos else max_pos
            res = (start, end)
            blanc = random.randint(min_blanc, max_blanc)
            start = end + blanc
            yield res

    def __apply_window(self, signal, window_shape, blanc_shape, drop=0):
        zeroes = np.zeros(signal.shape)
        min_window, max_window = window_shape
        start = -random.randint(min_window, max_window)  # Ojo con la negacion
        gen = self.__pos_gen(window_shape, blanc_shape, signal.shape[0] - 1, start)
        for start, end in gen:
            if drop and self.__decide(drop / 100):
                continue
            zeroes[start:end] = signal[start:end]
        return zeroes

    def __filtered_noise(self, signal, lowcut, highcut):
        snr = 10.0 ** (self.snr_db / 20.0)
        noise = np.random.default_rng().normal(0, self.__rms(signal) * (1 / snr), size=signal.shape)
        filtered = self.__butter_bandpass_filter(noise, lowcut, highcut, self.sample_rate)
        return filtered

    def eyes(self, signal):
        second = math.floor(self.sample_rate)
        hundred_ms = second // 10
        noise = self.__filtered_noise(signal, 1, 3)
        window = (7 * hundred_ms, 14 * hundred_ms)
        blanc = (second, 4 * second)
        noise = self.__apply_window(noise, window, blanc)
        return signal + noise

    def muscles(self, signal):
        second = math.floor(self.sample_rate)
        hundred_ms = second // 10
        noise = self.__filtered_noise(signal, 20, 60)
        window = (7 * hundred_ms, 7 * hundred_ms)
        blanc = (second, 4 * second)
        noise = self.__apply_window(noise, window, blanc, drop=75)
        return signal + noise

    def __decide(self, probability):
        return random.random() < probability

    def generate_noisy_signal(self, signal, eyes=True, muscles=True):
        contaminated = signal
        if (eyes) : contaminated = self.eyes(contaminated)
        if (muscles) : contaminated = self.muscles(contaminated)
        return contaminated
