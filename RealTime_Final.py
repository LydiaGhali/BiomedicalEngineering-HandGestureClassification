import pytrigno
from matplotlib import style
import matplotlib.pyplot as plt
import time 
import csv
import numpy as np
from scipy.signal import welch, butter, filtfilt, iirnotch, spectrogram
from collections import deque
import datetime
import joblib
import warnings
from scipy import signal

def emg_sensor_init(low_chan, high_chan, samples):
    dev_ = pytrigno.TrignoEMG(channel_range=(low_chan, high_chan), samples_per_read=samples, host='localhost', data_port = 50043)
    dev_.start()
    return dev_

# Notch filter to remove 50 Hz
def notch_filter(signal, fs, freq=50, Q=10):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)

# Bandpass filter for EMG (20-450 Hz)
def bandpass_filter_emg(signal, fs, lowcut=20, highcut=450, order=4):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return filtfilt(b, a, signal)

def live_emg_data(emg_instance):
    # Read raw EMG data
    data = emg_instance.read()
    return data

def compute_features(emg_data):
    features = []
    # print(f"EMG window data shape: {emg_data.shape}")
    for ch in range(emg_data.shape[1]):
        emg_data_ch = emg_data[:, ch,0]
        emg_data_ch = notch_filter(emg_data_ch, fs)  # 50 Hz
        emg_data_ch = notch_filter(emg_data_ch, fs, 60)  # 60 Hz (for 60 Hz noise)
        emg_data_ch = bandpass_filter_emg(emg_data_ch, fs)  # 20-450 Hz
        emg_data[:, ch,0] = emg_data_ch
    # print(f"EMG window data shape: {emg_data.shape}")
    #Time-domain features
    rms = np.sqrt(np.mean(emg_data ** 2, axis=0)).flatten()
    mav = np.mean(np.abs(emg_data), axis=0).flatten()
    wav = np.sum(np.abs(np.diff(emg_data, axis=0)), axis=0).flatten()
    var =  np.var(emg_data, axis=0 ).flatten()
    mean_val = np.mean(emg_data, axis=0).flatten()     # shape: (8,)
    std_val = np.std(emg_data, axis=0).flatten()       # shape: (8,)

    mean_f_window = []
    median_f_window = []
    psd_window =[]
    for ch in range(emg_data.shape[1]):
        signal = emg_data[:, ch].flatten()  # or .flatten()
        # nperseg = min(256, len(signal))
        f, Pxx = welch(signal, fs=fs, nperseg=1024)

        mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
        cumulative_power = np.cumsum(Pxx)
        median_freq = f[np.where(cumulative_power >= cumulative_power[-1] / 2)[0][0]]

        mean_f_window.append(mean_freq)
        median_f_window.append(median_freq)
        psd_window.append(np.trapezoid(Pxx, f))

    '''
    # Print sizes of each component
    print("\nFeature Vector Components:")
    print(f"RMS shape: {rms.shape}")  # Should be (num_channels,)
    print(f"MAV shape: {mav.shape}")  # Should be (num_channels,)
    print(f"WAV shape: {wav.shape}")  # Should be (num_channels,)
    print(f"VAR shape: {var.shape}")  # Should be (num_channels,)
    print(f"Mean Freq shape: {np.array(mean_f_window).shape} -> Flattened: {np.array(mean_f_window).flatten().shape}")
    print(f"Median Freq shape: {np.array(median_f_window).shape} -> Flattened: {np.array(median_f_window).flatten().shape}")
    '''

    '''
    # Print values of each component
    print("\nFeature Vector Components:")
    print(f"RMS: {rms}")
    print(f"MAV: {mav}")
    print(f"WAV: {wav}")
    print(f"VAR: {var}")
    print(f"Mean: {mean_val}")
    print(f"Std: {std_val}")
    print(f"Mean Frequencies: {np.array(mean_f_window)}")
    print(f"Median Frequencies: {np.array(median_f_window)}")
    #'''
    '''
    # Plot feature vectors (each is an 8-element array)
    features_list = {
        "RMS": rms,
        "MAV": mav,
        "WAV": wav,
        "VAR": var,
        "Mean": mean_val,
        "Std": std_val,
        "Mean Frequency": mean_f_window,
        "Median Frequency": median_f_window
    }

    # Create subplots
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()

    for i, (title, values) in enumerate(features_list.items()):
        axs[i].bar(range(1, 9), values)
        axs[i].set_title(title)
        axs[i].set_xlabel("Channel")
        axs[i].set_ylabel("Value")
        axs[i].set_xticks(range(1, 9))

    plt.tight_layout()
    plt.show()
    '''
    features = np.concatenate([
        rms,
        # mav,
        # wav,
        var,
        mean_val,
        std_val,
        np.array(mean_f_window).flatten(),
        np.array(median_f_window).flatten()
    ])
    
    # Print final feature vector size
    # print(f"\nFinal Feature Vector shape: {features.shape}")
    # print(f"Total Features: {features.size}")
    return features
    

def Code_Loop():
    while True:
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        elapsed_time = time.time() - start_time

        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        milliseconds = int((elapsed_time % 1) * 1000)
        formatted_time = f'{minutes:02}:{seconds:02}:{milliseconds:03}'

        emg_data = live_emg_data(emg)  # Should return a list/array of values (1 sample for all channels)
        if emg_data.all:
            emg_array = np.array(emg_data)
            emg_buffer.append(emg_array)

            # Save raw EMG
            EMG_writer.writerow([current_time, formatted_time] + list(emg_array))
            EMG_file.flush()
            if len(emg_buffer) == window_size:
                emg_data_window = np.array(emg_buffer)  # shape = (window_size, num_channels)
                features = compute_features(emg_data_window)
                features_string = [str(f) for f in features.flatten()]
                FEATURES_writer.writerow([current_time, formatted_time] + features_string)
                FEATURES_file.flush()
                # print(f"Features shape: {features.shape}")
                y_pred_rf = rf.predict(features.reshape(1, -1))
                y_pred_svm = svm.predict(features.reshape(1, -1))
                predicted_label_rf = int(y_pred_rf[0])  # Convert to plain Python int
                predicted_label_svm = int(y_pred_svm[0])  # Convert to plain Python int
                print(f"Gesture classified as: {predicted_label_rf} - {gesture_labels[predicted_label_rf]} ")
                # print(f"Gesture classified as: {predicted_label_svm} - {gesture_labels[predicted_label_svm]} by svm")
        time.sleep(1 / fs)

#load model
rf = joblib.load('random_forest_model.joblib')
svm = joblib.load('svm_model.joblib')

# Parameters
fs = 1000  # Sampling rate in Hz
window_sec = 3
window_size = fs * window_sec

# Buffers
emg_buffer = deque(maxlen=window_size)

print("EMG init")
# Sensor init
emg = emg_sensor_init(0, 7, 1)
time.sleep(1)  # Give it a moment to stabilize if needed
print("EMG init completed")

# Define gesture label mappings
gesture_labels = {
    0: "unmarked data",
    1: "hand at rest", 
    2: "hand clenched in a fist",
    3: "wrist flexion",
    4: "wrist extension",
    5: "radial deviations",
    6: "ulnar deviations"
}

# Initialize file writers
EMG_file = open('EMG.csv', 'a', newline='')
EMG_writer = csv.writer(EMG_file)
channel_count = len(live_emg_data(emg))
EMG_writer.writerow(['Time', 'Elapsed Time'] + [f'EMG_{i}' for i in range(channel_count)])

FEATURES_file = open('EMG_features.csv', 'a', newline='')
FEATURES_writer = csv.writer(FEATURES_file)
feature_headers = []
for i in range(channel_count):
    feature_headers += [f'RMS_{i}', f'MAV_{i}', f'MeanFreq_{i}', f'MedianFreq_{i}']
FEATURES_writer.writerow(['Time', 'Elapsed Time'] + feature_headers)

start_time = time.time()

try:
    Code_Loop()
except KeyboardInterrupt:
    print("\nTerminated by user.")
    EMG_file.close()
    FEATURES_file.close()