import pandas as pd
from scipy.signal import resample

from EEGArtifactRemoval.preprocessing.DataPreprocessor import DataPreprocessor
from EEGArtifactRemoval.preprocessing.SignalProcessor import SignalProcessor


def get_numpy_array_from_csv(path, desired_sample_rate, input_sample_rate, snr_db,mins=1):
    """
    Reads a csv file and returns a normalized and preprocessed numpy array

    """
    seconds_in_samples = input_sample_rate * mins * 60
    secs = mins * 60
    samps = secs * desired_sample_rate
    clean = resample(
        pd.read_csv(path, usecols=[0, 1], index_col=0, nrows=seconds_in_samples).to_numpy(dtype='float64'),
        samps)
    # clean = pd.read_csv(path, usecols=[0,1],index_col=0,nrows=seconds_in_samples).to_numpy(dtype='float64')
    processor = SignalProcessor(desired_sample_rate,snr_db)
    normalized_clean = processor.normalize(clean)
    noisy = processor.generate_noisy_signal(clean)
    normalized_noisy = processor.normalize(noisy)
    return (normalized_clean, normalized_noisy)


def obtain_separated_data(clean, noisy, window_length, step, separation_factor=0.8):
    processor = DataPreprocessor(window_length, step, separation_factor)
    clean_train, noisy_train, clean_test, noisy_test = processor.separate_data(clean, noisy)
    train_data = processor.to_list_of_time_windows(noisy_train, window_length, step=step)
    train_labels = processor.to_list_of_time_windows(clean_train, window_length, step=step)
    test_data = processor.to_list_of_time_windows(noisy_test, window_length, step=step)
    test_labels = processor.to_list_of_time_windows(clean_test, window_length, step=step)

    return (train_data, train_labels, test_data, test_labels)