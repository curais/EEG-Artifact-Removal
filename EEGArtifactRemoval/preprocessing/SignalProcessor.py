import math
import random
import numpy as np
from scipy import stats
from scipy.signal import butter, lfilter, resample


class SignalProcessor:

    def __init__(self, sample_rate, snr_db):
        self.sample_rate = sample_rate
        self.snr_db = snr_db

    def rms(self, signal):
        return np.sqrt(np.mean(signal ** 2))

    def gaussian_noise(self, length, strength=1):
        return np.random.normal(0, 1, length) * strength

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def pos_gen(self, window_shape, blanc_shape, max_pos, start=0):
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

    def apply_window(self,signal, window_shape, blanc_shape, drop=0):
        zeroes = np.zeros(signal.shape)
        min_window, max_window = window_shape
        start = -random.randint(min_window, max_window)  # Ojo con la negacion
        gen = self.pos_gen(window_shape, blanc_shape, signal.shape[0] - 1, start)
        for start, end in gen:
            if drop and self.decide(drop / 100):
                continue
            zeroes[start:end] = signal[start:end]
        return zeroes

    def filtered_noise(self, signal, lowcut, highcut):
        snr = 10.0 ** (self.snr_db / 20.0)
        noise = np.random.default_rng().normal(0, self.rms(signal) * (1 / snr), size=signal.shape)
        filtered = self.butter_bandpass_filter(noise, lowcut, highcut, self.sample_rate)
        return filtered

    def eyes(self, signal):
        second = math.floor(self.sample_rate)
        hundred_ms = second // 10
        noise = self.filtered_noise(signal, 1, 3)
        window = (7 * hundred_ms, 14 * hundred_ms)
        blanc = (second, 4 * second)
        noise = self.apply_window(noise, window, blanc)
        return signal + noise

    def muscles(self, signal):
        second = math.floor(self.sample_rate)
        hundred_ms = second // 10
        noise = self.filtered_noise(signal, 20, 60)
        print(noise)
        window = (7 * hundred_ms, 7 * hundred_ms)
        blanc = (second, 4 * second)
        noise = self.apply_window(noise, window, blanc, drop=75)
        return signal + noise

    def decide(self, probability):
        return random.random() < probability

    def normalize(self, x):
        """
        Normalizes and numpy array to have mean zero and standard deviation one

        :param x: The numpy array
        :return: The normalized array
        """
        return stats.zscore(x)

    def generate_noisy_signal(self, signal):
        return self.eyes(self.muscles(signal))
