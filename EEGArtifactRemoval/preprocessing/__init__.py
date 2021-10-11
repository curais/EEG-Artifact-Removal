import pandas as pd
from scipy.signal import resample
import numpy as np
import random
import mpu

from scipy.stats import stats

from EEGArtifactRemoval.preprocessing.DataPreprocessor import DataPreprocessor
from EEGArtifactRemoval.preprocessing.ArtifactGenerator import ArtifactGenerator

def get_numpy_array_from_csv(path, desired_sample_rate, input_sample_rate, snr_db,mins=1,eyes=True,muscles=True):
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
    generator = ArtifactGenerator(desired_sample_rate, snr_db)
    normalized_clean = stats.zscore(clean)
    noisy = generator.generate_noisy_signal(clean,eyes,muscles)
    normalized_noisy = stats.zscore(noisy)
    return (normalized_clean, normalized_noisy)

def get_trials(pathstrings,desired_sample_rate, input_sample_rate,mins=1,augmentation_factor=1, eyes=True,muscles=True):
    snr_db = [-5,-3,-1,0,1,3,5]
    clean = []
    noisy = []
    for path in pathstrings:
        for j in range(augmentation_factor):
            snr = snr_db[random.randint(0,len(snr_db) - 1)]
            clean_aux, noisy_aux = get_numpy_array_from_csv(path, desired_sample_rate, input_sample_rate, snr,mins,eyes,muscles)
            clean.append(clean_aux)
            noisy.append(noisy_aux)
    clean = np.array(clean)
    noisy = np.array(noisy)
    return (clean,noisy)

def separate_data(clean, noisy, window_length, step, separation_factor=0.8, save=False):
    processor = DataPreprocessor(window_length, step, separation_factor)
    noisy_train,clean_train,noisy_test, clean_test = processor.separate_data(clean, noisy,save=save)
    train_data = processor.to_list_of_time_windows(noisy_train, window_length, step=step)
    train_labels = processor.to_list_of_time_windows(clean_train, window_length, step=step)
    test_data = processor.to_list_of_time_windows(noisy_test, window_length, step=step)
    test_labels = processor.to_list_of_time_windows(clean_test, window_length, step=step)
    if save:
        mpu.io.write('train_data.pickle', train_data)
        mpu.io.write('train_labels.pickle', train_labels)
        mpu.io.write('test_data.pickle', test_data)
        mpu.io.write('test_labels.pickle', test_labels)
    return (train_data, train_labels, test_data, test_labels)
