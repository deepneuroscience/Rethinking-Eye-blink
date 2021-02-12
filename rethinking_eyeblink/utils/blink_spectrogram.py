import scipy
from scipy import signal
from astropy.timeseries import LombScargle
from .gausswin import gausswin
from .slidingwin import *
import numpy as np


def blink_spectrogram(Fss, x0):
    isfirst = True
    # Spectrogram parameters
    Tmax = 30 # From Cho 2021: "The first cutoff frequency determines the upper time limit of blinking (30 seconds)"
    lag = Tmax * Fss

    # Windowing
    overlap_sec = 1
    window_list = overlap_windows_sec(x0, Fss, overlap_sec, lag)

    s_PSD_1 = []


    # Building blink spectrograms
    for i in range(len(window_list)):

        # Min-max normalise signal data
        data_use = window_list[i]
        data_info = data_use.copy()  # shallow copy

        max_data = max(data_info)
        min_data = min(data_info)
        for j in range(len(data_info)):
            data_info[j] = (data_info[j] - min_data) / (max_data - min_data)

        # BPF paramters
        filterN = 3
        Wn1 = 0.033
        Wn2 = 0.4167
        Fn = Fss / 2

        # Band pass filtering with Elliptic filter
        filter_b, filter_a = scipy.signal.ellip(filterN, 3, 6, [Wn1 / Fn, Wn2 / Fn], btype='bandpass')
        filtered_featurescaled_data = scipy.signal.lfilter(filter_b, filter_a, data_info, axis=0)
        w = gausswin(lag)

        gaussian_final_window = np.array(filtered_featurescaled_data * w.T, dtype='float64')

        # Compute periodogram of signal data
        t = np.arange(0, (len(gaussian_final_window)) / Fss, 1 / Fss)
        dy = 0.1
        frequency, freq_amplitude = LombScargle(t, gaussian_final_window, dy).autopower(samples_per_peak=10)

        # accumulate power distributions
        if isfirst:
            range1 = np.where(frequency < Wn1)
            range2 = np.where(frequency > Wn2)
            isfirst = False

        s_PSD_1.append(freq_amplitude[np.max(range1):np.min(range2)])

    s_PSD_1 = np.array(s_PSD_1).T

    return s_PSD_1