import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, spectrogram, welch
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from sklearn.svm import SVC
from numpy import trapezoid
import os

# Base directory where all the folders are located
base_dir = r"C:\Users\lydia\OneDrive\Desktop\Uni\SPRING 2025\Biomedical Eng\Project\emg+data+for+gestures\EMG_data_for_gestures-master"

file_paths = []
# Loop through each folder in the base directory
for root, dirs, files in os.walk(base_dir):
    for file in files:
        # Check if the file is a text file (ends with .txt)
        if file.endswith('.txt'):
            # Construct the full file path and add it to the list
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

# Notch filter to remove 50 Hz
def notch_filter(signal, fs, freq=50, Q=10):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)

# Bandpass filter for EMG (20-450 Hz)
def bandpass_filter_emg(signal, fs, lowcut=20, highcut=450, order=4):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return filtfilt(b, a, signal)

# Baseline correction and normalization
def preprocess_emg_channel(signal, estimated_fs):
    signal = notch_filter(signal, estimated_fs) #50Hz 
    signal = notch_filter(signal, estimated_fs, 60) #50Hz 
    signal = bandpass_filter_emg(signal, estimated_fs)
    # signal = signal - np.mean(signal)
    # signal = signal / np.max(np.abs(signal))
    return signal
    
# Smoothing using moving average
def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def get_feature_vector(file_path):

    # Print the file path being processed
    print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path, sep='\t')

    # Extract data
    time_ms = df.iloc[:, 0].values
    time_sec = time_ms / 1000
    emg_data = df.iloc[:, 1:9].values
    class_labels = df.iloc[:, 9].values

    time_diff = np.diff(time_sec)  # Time difference between samples
    estimated_fs = 1 / np.mean(time_diff)  # Estimate sampling frequency
    print("Estimated frequency = ", estimated_fs)
    # estimated_fs = 1000  # Sampling frequency in Hz (assumed from file, adjust if needed)

    num_samples = len(time_sec)
    num_channels = emg_data.shape[1]

    # === Plot raw EMG signals ===
    plt.figure(figsize=(12, 10))
    plt.title(f'Raw EMG readings')
    for i in range(num_channels):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_sec, emg_data[:, i])
        plt.title(f'EMG  {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('EMG (V)')
    plt.tight_layout()  
    plt.show()

    # Apply preprocessing to all data
    emg_cleaned = np.zeros_like(emg_data)
    for i in range(num_channels):
        emg_cleaned[:, i] = preprocess_emg_channel(emg_data[:, i], estimated_fs)

    # === Plot raw vs cleaned EMG signals ===
    plt.figure(figsize=(12, 10))
    for i in range(num_channels):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_sec, emg_data[:, i], label='Raw', alpha=0.6)  # Raw signal
        plt.plot(time_sec, emg_cleaned[:, i], label='Filtered', linewidth=1)  # Cleaned signal
        plt.title(f'EMG Channel {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('EMG (V)')
        plt.legend()
    plt.tight_layout()
    plt.show()

    # Rectify
    emg_rect = np.abs(emg_cleaned)
    emg_rect = emg_cleaned
    emg_smoothed = np.zeros_like(emg_rect)  # shape: (num_samples, num_channels)
    for i in range(num_channels):
        emg_smoothed[:, i] = moving_average(emg_rect[:, i])

    # === Plot cleaned vs smoothed EMG signals ===
    plt.figure(figsize=(12, 10))
    for i in range(num_channels):
        plt.subplot(4, 2, i + 1)

        plt.plot(time_sec, emg_rect[:, i], label='Filtered rectified', alpha=0.7, color='blue')
        plt.plot(time_sec, emg_smoothed[:, i], label='Smoothed', linewidth=1, color='red')

        plt.title(f'EMG Channel {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('EMG (V)')
        plt.legend()
    plt.tight_layout()
    plt.show()

    #''' FFT, Power spectrum and Wavelet transform for all emg data 
    # === Compute FFT for each EMG channel ===
    fft_vals = []
    freqs = []

    for i in range(num_channels):
        signal = emg_cleaned[:, i]
        n = len(signal)
        freqs.append(np.fft.rfftfreq(n, d=1/estimated_fs))  # Frequencies
        fft_vals.append(np.abs(np.fft.rfft(signal)))  # Magnitude of FFT

    # === Compute Power Spectrum using Welch ===
    Pxx_vals = []
    f_vals = []

    for i in range(num_channels):
        f, Pxx = welch(emg_cleaned[:, i], fs=estimated_fs, nperseg=1024)
        f_vals.append(f)
        Pxx_vals.append(Pxx)

    # === Compute Continuous Wavelet Transform (CWT) ===
    scales = np.arange(1, 128)
    cwt_vals = []

    for i in range(num_channels):
        signal = emg_cleaned[:, i]
        coef, freqs_cwt = pywt.cwt(signal, scales, 'morl', sampling_period=1/estimated_fs)
        cwt_vals.append((coef, freqs_cwt))
    
    # === Plot FFT ===
    plt.figure(figsize=(12, 10))
    for i in range(num_channels):
        plt.subplot(4, 2, i + 1)
        plt.plot(freqs[i], fft_vals[i], color='purple')
        plt.title(f'FFT - EMG Channel {i + 1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid()
    plt.tight_layout()
    plt.show()

    # === Plot Power Spectrum ===
    plt.figure(figsize=(12, 10))
    for i in range(num_channels):
        plt.subplot(4, 2, i + 1)
        plt.semilogy(f_vals[i], Pxx_vals[i], color='green')
        plt.title(f'Power Spectrum - EMG Channel {i + 1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.grid()
    plt.tight_layout()
    plt.show()

    # === Plot Wavelet Transform ===
    plt.figure(figsize=(12, 10))
    for i in range(num_channels):
        coef, freqs_cwt = cwt_vals[i]
        plt.subplot(4, 2, i + 1)
        plt.imshow(np.abs(coef), extent=[time_sec[0], time_sec[-1], freqs_cwt[-1], freqs_cwt[0]],
                aspect='auto', cmap='jet')
        plt.title(f'CWT - EMG Channel {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Magnitude', shrink=0.8)
    plt.tight_layout()
    plt.show()

    # == Seperate data based on class and trials ==
    emg_trials = []   # List of lists: [ [trial1_class0, trial2_class0], [trial1_class1, trial2_class1], ... ]
    time_trials = []
    gap_threshold = 15  # seconds — adjust based on your data (e.g., if time gap between trials >2s)

    for class_id in range(7):  # classes 0 to 6
        indices = np.where(class_labels == class_id)[0]
        emg_class_data = emg_smoothed[indices, :]
        time_class_data = time_sec[indices]

        # Compute time differences
        time_diffs = np.diff(time_class_data)

        # Find index where time jumps (gap)
        gap_index = np.argmax(time_diffs > gap_threshold)
        if time_diffs[gap_index] <= gap_threshold:
            print(f"No large gap found in class {class_id}, skipping split.")
            emg_trials.append([emg_class_data])
            time_trials.append([time_class_data])
            continue

        # Split at gap
        emg_trial1 = emg_class_data[:gap_index + 1]
        emg_trial2 = emg_class_data[gap_index + 1:]

        time_trial1 = time_class_data[:gap_index + 1]
        time_trial2 = time_class_data[gap_index + 1:]

        # Append split trials
        emg_trials.append([emg_trial1, emg_trial2])
        time_trials.append([time_trial1, time_trial2])

    # === Plot readings of each trial in each class
    for class_id in range(len(emg_trials)):  # Loop over classes
        for trial_id in range(len(emg_trials[class_id])):  # Loop over trials (usually 2)
            emg_trial = emg_trials[class_id][trial_id]
            time_trial = time_trials[class_id][trial_id]

            plt.figure(figsize=(12, 10))
            plt.suptitle(f'Class {class_id} - Trial {trial_id + 1}', fontsize=16)

            for ch in range(emg_trial.shape[1]):  # Loop over EMG channels
                plt.subplot(4, 2, ch + 1)
                plt.plot(time_trial, emg_trial[:, ch])
                plt.title(f'EMG Channel {ch + 1}')
                plt.xlabel('Time (s)')
                plt.ylabel('EMG (V)')
                plt.tight_layout()

            plt.subplots_adjust(top=0.9)  # To make room for suptitle
            plt.show()

    # Initialize empty lists to hold results
    rms_values = []             # RMS values for each trial of each class
    mav_values = []             # MAV
    wav_values = []             # Waveform length
    var_values = []             # Variance
    mean_values = []            # Mean value
    std_values = []             # Std value
    fft_values = []             # FFT results (magnitude or complex) per channel
    power_spectra = []          # Power spectrum values
    power_freqs = []
    mean_freqs = []             # Mean frquencies
    median_freqs = []           #Median frequencies
    psd_values = []             # Power spectrum density
    wavelet_transforms = []     # Wavelet transform coefficients

    for class_trials in emg_trials:  # emg_trials[class_id][trial_id]
        class_rms = []
        class_mav = []
        class_wav = []
        class_var = []
        class_mean = []
        class_std = []
        class_fft = []
        class_power = []
        class_freqs = []
        class_mean_f = []
        class_median_f = []
        class_psd = []
        class_wavelet = []

        for trial in class_trials:
            n_channels = trial.shape[1]

            rms = np.sqrt(np.mean(trial ** 2, axis=0))  #Root mean squared, shape: (8,)
            mav = np.mean(np.abs(trial), axis=0)       #Mean absolute value, shape: (8,)
            wav = np.sum(np.abs(np.diff(trial, axis=0)), axis=0)  #Waveform length, shape: (8,)
            var =  np.var(trial, axis=0 )   #Variance, shape: (8,)
            mean_val = np.mean(trial, axis=0)     # shape: (8,)
            std_val = np.std(trial, axis=0)       # shape: (8,)
            class_rms.append(rms)
            class_mav.append(mav)
            class_wav.append(wav)
            class_var.append(var)
            class_mean.append(mean_val)
            class_std.append(std_val)
        
            # === FFT: One spectrum per channel
            fft_trial = [np.fft.rfft(trial[:, ch]) for ch in range(trial.shape[1])]  # list of arrays
            class_fft.append(fft_trial)

            # === Power Spectrum using Welch’s method
            power_trial = []
            freqs_trial = []
            mean_f_trial = []
            median_f_trial = []
            psd_trial = []
            for ch in range(trial.shape[1]):
                f, Pxx = welch(trial[:, ch], fs=estimated_fs, nperseg=1024)
                power_trial.append(Pxx)
                freqs_trial.append(f)
                psd_trial.append(np.trapz(Pxx, f))  # Total power = area under PSD curve
                
                # Mean frequency: sum(f * Pxx) / sum(Pxx)
                mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
                mean_f_trial.append(mean_freq)

                # Median frequency: frequency where cumulative sum reaches half the total power
                cumulative_power = np.cumsum(Pxx) 
                median_freq = f[np.where(cumulative_power >= cumulative_power[-1] / 2)[0][0]] #cumulative_power[-1] = total power, [0][0]: to get the first index where cumpow>half total cum power
                median_f_trial.append(median_freq)

            power_trial = np.array(power_trial)       
            freqs_trial = np.array(freqs_trial)      
            mean_f_trial = np.array(mean_f_trial)     
            median_f_trial = np.array(median_f_trial) 
            class_power.append(power_trial)
            class_freqs.append(freqs_trial)
            class_mean_f.append(mean_f_trial)
            class_median_f.append(median_f_trial)
            class_psd.append(psd_trial)

            # === Wavelet Transform (e.g., using pywt)
            wavelet_trial = [pywt.wavedec(trial[:, ch], 'db4', level=4) for ch in range(trial.shape[1])]
            class_wavelet.append(wavelet_trial)

        # Append all trials' results for this class
        rms_values.append(class_rms)
        mav_values.append(class_mav)
        wav_values.append(class_wav)
        var_values.append(class_var)
        fft_values.append(class_fft)
        power_spectra.append(class_power)
        power_freqs.append(class_freqs)
        mean_freqs.append(class_mean_f)
        median_freqs.append(class_median_f)
        psd_values.append(class_psd)
        wavelet_transforms.append(class_wavelet)
    #'''

    #''' === Plot RMS ===
    fig, ax = plt.subplots(figsize=(6, 4))
    rms_trial = rms_values[0][0]
    ax.bar(range(len(rms_trial)), rms_trial, color='darkcyan')
    ax.set_title('Class 0, Trial 0')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('RMS Value')
    ax.set_xticks(range(len(rms_trial)))
    ax.set_xticklabels([f'Ch {i+1}' for i in range(len(rms_trial))])
    ax.grid(axis='y')
    plt.tight_layout()
    plt.show()
    # --- Plot Classes 1+ ---
    remaining_trials = [(class_id, trial_id, trial) 
                        for class_id, class_trials in enumerate(rms_values[1:], start=1)
                        for trial_id, trial in enumerate(class_trials)]
    cols = 4
    rows = -(-len(remaining_trials) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for idx, (class_id, trial_id, trial) in enumerate(remaining_trials):
        ax = axes[idx]
        ax.bar(range(len(trial)), trial, color='darkcyan')
        ax.set_title(f'Class {class_id}, Trial {trial_id}')
        ax.set_xlabel('EMG Channel')
        ax.set_ylabel('RMS Value')
        ax.set_xticks(range(len(trial)))
        ax.set_xticklabels([f'Ch {i+1}' for i in range(len(trial))])
        ax.grid(axis='y')
    for i in range(len(remaining_trials), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()
    #'''

    # --- Plot MAV  ---
    fig, ax = plt.subplots(figsize=(6, 4))
    mav_trial = mav_values[0][0]
    ax.bar(range(len(mav_trial)), mav_trial, color='darkcyan')
    ax.set_title('MAV - Class 0, Trial 0')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('MAV Value')
    ax.set_xticks(range(len(mav_trial)))
    ax.set_xticklabels([f'Ch {i+1}' for i in range(len(mav_trial))])
    ax.grid(axis='y')
    plt.tight_layout()
    plt.show()
    # --- Classes 1+ ---
    remaining_trials = [(class_id, trial_id, trial) 
                        for class_id, class_trials in enumerate(mav_values[1:], start=1)
                        for trial_id, trial in enumerate(class_trials)]
    cols = 4
    rows = -(-len(remaining_trials) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for idx, (class_id, trial_id, trial) in enumerate(remaining_trials):
        ax = axes[idx]
        ax.bar(range(len(trial)), trial, color='darkcyan')
        ax.set_title(f'MAV - Class {class_id}, Trial {trial_id}')
        ax.set_xlabel('EMG Channel')
        ax.set_ylabel('MAV Value')
        ax.set_xticks(range(len(trial)))
        ax.set_xticklabels([f'Ch {i+1}' for i in range(len(trial))])
        ax.grid(axis='y')
    for i in range(len(remaining_trials), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()
    #'''

    # --- Plot WAV  ---
    fig, ax = plt.subplots(figsize=(6, 4))
    wav_trial = wav_values[0][0]
    ax.bar(range(len(wav_trial)), wav_trial, color='darkcyan')
    ax.set_title('WAV - Class 0, Trial 0')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('WAV Value')
    ax.set_xticks(range(len(wav_trial)))
    ax.set_xticklabels([f'Ch {i+1}' for i in range(len(wav_trial))])
    ax.grid(axis='y')
    plt.tight_layout()
    plt.show()
    # --- Classes 1+ ---
    remaining_trials = [(class_id, trial_id, trial) 
                        for class_id, class_trials in enumerate(wav_values[1:], start=1)
                        for trial_id, trial in enumerate(class_trials)]
    cols = 4
    rows = -(-len(remaining_trials) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for idx, (class_id, trial_id, trial) in enumerate(remaining_trials):
        ax = axes[idx]
        ax.bar(range(len(trial)), trial, color='darkcyan')
        ax.set_title(f'WAV - Class {class_id}, Trial {trial_id}')
        ax.set_xlabel('EMG Channel')
        ax.set_ylabel('WAV Value')
        ax.set_xticks(range(len(trial)))
        ax.set_xticklabels([f'Ch {i+1}' for i in range(len(trial))])
        ax.grid(axis='y')
    for i in range(len(remaining_trials), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()
    #'''

    #''' --- Plot VAR  ---
    fig, ax = plt.subplots(figsize=(6, 4))
    var_trial = var_values[0][0]
    ax.bar(range(len(var_trial)), var_trial, color='darkcyan')
    ax.set_title('VAR - Class 0, Trial 0')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('VAR Value')
    ax.set_xticks(range(len(var_trial)))
    ax.set_xticklabels([f'Ch {i+1}' for i in range(len(var_trial))])
    ax.grid(axis='y')
    plt.tight_layout()
    plt.show()
    # --- Classes 1+ ---
    remaining_trials = [(class_id, trial_id, trial) 
                        for class_id, class_trials in enumerate(var_values[1:], start=1)
                        for trial_id, trial in enumerate(class_trials)]
    cols = 4
    rows = -(-len(remaining_trials) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for idx, (class_id, trial_id, trial) in enumerate(remaining_trials):
        ax = axes[idx]
        ax.bar(range(len(trial)), trial, color='darkcyan')
        ax.set_title(f'VAR - Class {class_id}, Trial {trial_id}')
        ax.set_xlabel('EMG Channel')
        ax.set_ylabel('VAR Value')
        ax.set_xticks(range(len(trial)))
        ax.set_xticklabels([f'Ch {i+1}' for i in range(len(trial))])
        ax.grid(axis='y')
    for i in range(len(remaining_trials), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()
    #'''
    
    #''' == Plot FFT ==
    for class_id, class_fft in enumerate(fft_values):
        for trial_id, fft_trial in enumerate(class_fft):
            plt.figure(figsize=(12, 10))
            n_samples = emg_trials[class_id][trial_id].shape[0]  # original signal length
            freqs = np.fft.rfftfreq(n_samples, d=1 / estimated_fs)
            for ch in range(len(fft_trial)):
                fft_ch = fft_trial[ch]
                plt.subplot(4, 2, ch + 1)
                plt.plot(freqs, np.abs(fft_ch), color='purple')
                plt.title(f'FFT - Class {class_id}, Trial {trial_id}, Ch {ch + 1}')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Magnitude')
                plt.grid()
            plt.tight_layout()
            plt.show()
    #'''
    
    #''' === Plot Power Spectrum (Welch-based) ===
    for class_id, class_power in enumerate(power_spectra):
        for trial_id, power_trial in enumerate(class_power):
            trial_freqs = power_freqs[class_id][trial_id]  # List of 8 freq arrays
            plt.figure(figsize=(12, 10))
            for ch in range(len(power_trial)):
                power_ch = power_trial[ch]        # Power values for channel ch
                freqs_ch = trial_freqs[ch]        # Frequencies for channel ch

                plt.subplot(4, 2, ch + 1)
                plt.semilogy(freqs_ch, power_ch, color='green')
                plt.title(f'Power Spectrum - Class {class_id}, Trial {trial_id}, Ch {ch + 1}')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power')
                plt.grid()
            plt.tight_layout()
            plt.show()
    #'''

    #''' == Plot PSD ==
    for class_id, class_power in enumerate(power_spectra):
        for trial_id, power_trial in enumerate(class_power):
            freqs_trial = power_freqs[class_id][trial_id]  # list of 8 frequency arrays
            plt.figure(figsize=(12, 10))

            for ch in range(8):  # assuming 8 EMG sensors
                freqs = freqs_trial[ch]
                psd = power_trial[ch]

                plt.subplot(4, 2, ch + 1)
                plt.semilogy(freqs, psd, color='darkgreen')
                plt.title(f'PSD - Class {class_id}, Trial {trial_id}, Ch {ch + 1}')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power (dB/Hz)')
                plt.grid(True)

            plt.tight_layout()
            plt.show()
    #'''

    #''' --- Plot Mean Frequency ---
    fig, ax = plt.subplots(figsize=(6, 4))
    mean_values = mean_freqs[0][0]
    channels = [f'Ch {i+1}' for i in range(len(mean_values))]
    ax.bar(channels, mean_values, color='skyblue')
    ax.set_title('Mean Frequencies - Class 0, Trial 0')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(0, max(mean_values) * 1.5)
    ax.grid(axis='y')
    plt.tight_layout()
    plt.show()
    # --- Plot Mean Frequencies for Classes 1+ ---
    remaining_trials = [(class_id, trial_id, trial) 
                        for class_id, class_trials in enumerate(mean_freqs[1:], start=1)
                        for trial_id, trial in enumerate(class_trials)]
    cols = 4
    rows = -(-len(remaining_trials) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for idx, (class_id, trial_id, mean_values) in enumerate(remaining_trials):
        ax = axes[idx]
        channels = [f'Ch {i+1}' for i in range(len(mean_values))]
        ax.bar(channels, mean_values, color='skyblue')
        ax.set_title(f'Class {class_id}, Trial {trial_id}')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(0, max(mean_values) * 1.5)
        ax.grid(axis='y')
    for i in range(len(remaining_trials), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()
    #'''
    
    #''' --- Plot Median Frequency ---
    fig, ax = plt.subplots(figsize=(6, 4))
    median_values = median_freqs[0][0]
    channels = [f'Ch {i+1}' for i in range(len(median_values))]
    ax.bar(channels, median_values, color='skyblue')
    ax.set_title('Median Frequencies - Class 0, Trial 0')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(0, max(median_values) * 1.5)
    ax.grid(axis='y')
    plt.tight_layout()
    plt.show()
    # --- Classes 1+ ---
    remaining_trials = [(class_id, trial_id, trial) 
                        for class_id, class_trials in enumerate(median_freqs[1:], start=1)
                        for trial_id, trial in enumerate(class_trials)]
    cols = 4
    rows = -(-len(remaining_trials) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for idx, (class_id, trial_id, median_values) in enumerate(remaining_trials):
        ax = axes[idx]
        channels = [f'Ch {i+1}' for i in range(len(median_values))]
        ax.bar(channels, median_values, color='skyblue')
        ax.set_title(f'Class {class_id}, Trial {trial_id}')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(0, max(median_values) * 1.5)
        ax.grid(axis='y')
    for i in range(len(remaining_trials), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()
    #'''

    #''' === Plot wavelet transform ==
    # Define wavelet and scales
    wavelet = 'morl'
    scales = np.arange(1, 128)
    for class_id, class_trials in enumerate(emg_trials):
        for trial_id, trial in enumerate(class_trials):
            plt.figure(figsize=(12, 10))  # One figure per trial
            for ch in range(trial.shape[1]):
                signal = trial[:, ch]
                coef, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1 / estimated_fs)
                plt.subplot(4, 2, ch + 1)
                plt.imshow(np.abs(coef),
                        extent=[0, signal.shape[0] / estimated_fs, freqs[-1], freqs[0]],
                        cmap='jet', aspect='auto')
                plt.colorbar(label='Magnitude', format='%.2e')
                plt.title(f'Ch {ch + 1}')
                plt.xlabel('Time (s)')
                plt.ylabel('Freq (Hz)')
            plt.suptitle(f'CWT - Class {class_id}, Trial {trial_id}', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the super title
            plt.show()
    #'''

    # == Create feature vector ==
    X = []  # Feature vectors
    y = []  # Labels (class ID)
    n_classes = len(rms_values)

    for class_id in range(n_classes):
        n_trials = len(rms_values[class_id])
        for trial_id in range(n_trials):
            #'''
            print(f"\n--- Class {class_id} - Trial {trial_id} ---")
            print("RMS shape:", rms_values[class_id][trial_id].shape)
            print("MAV shape:", mav_values[class_id][trial_id].shape)
            print("WAV shape:", wav_values[class_id][trial_id].shape)
            print("VAR shape:", var_values[class_id][trial_id].shape)
            print("Mean shape:", mean_values[class_id][trial_id].shape)
            print("Std shape:", std_values[class_id][trial_id].shape)
            print("Mean Freqs shape:", mean_freqs[class_id][trial_id].shape)
            print("Median Freqs shape:", median_freqs[class_id][trial_id].shape)
            #'''
            '''
            # Print values of each component
            print("\nFeature Vector Components:")
            print(f"RMS: {rms_values[class_id][trial_id]}")
            print(f"MAV: {mav_values[class_id][trial_id]}")
            print(f"WAV: {wav_values[class_id][trial_id]}")
            print(f"VAR: {var_values[class_id][trial_id]}")
            print(f"Mean: {mean_values[class_id][trial_id]}")
            print(f"Std: {std_values[class_id][trial_id]}")
            print(f"Mean Frequencies: {mean_freqs[class_id][trial_id]}")
            print(f"Median Frequencies: {median_freqs[class_id][trial_id]}")
            '''

            basic_features = np.concatenate([
                rms_values[class_id][trial_id],
                mav_values[class_id][trial_id],
                wav_values[class_id][trial_id],
                var_values[class_id][trial_id],
                mean_values[class_id][trial_id],
                std_values[class_id][trial_id],
                mean_freqs[class_id][trial_id],
                median_freqs[class_id][trial_id],
            ])
            '''
            # === Add PSD features ===
            # Using the total power from Welch's method (already calculated in psd_values)
            psd_features = psd_values[class_id][trial_id]
            
            # === Add Wavelet features ===
            # Extract relevant wavelet coefficients (example: energy of coefficients)
            wavelet_coeffs = wavelet_transforms[class_id][trial_id]
            wavelet_features = []
            
            for ch_coeffs in wavelet_coeffs:  # For each channel
                # Calculate energy for each decomposition level
                level_energies = [np.sum(np.square(level)) for level in ch_coeffs]
                wavelet_features.extend(level_energies)
            '''

            # === Combine all features ===
            trial_features = np.concatenate([
                basic_features,
                # psd_features,
                # np.array(wavelet_features)
            ])  
            # === Add to dataset ===
            X.append(trial_features)
            y.append(class_id)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    print("Shape of x:", X.shape)
    print("Shape of y:", y.shape)
    return X, y

# Create the model
rf = RandomForestClassifier(n_estimators=400,warm_start= True, random_state=42)
svm = SVC(kernel='rbf', random_state=42)  

# Load all data
X, y = [], []
for path in file_paths:
    features, label = get_feature_vector(path)  # features: (13, 48), label: (13,)
    X.extend(features)  # instead of X.append(features)
    y.extend(label)     # instead of y.append(label)

X, y = np.array(X), np.array(y)
print("X shape:", X.shape)  # Should be (910, 48)
print("y shape:", y.shape)  # Should be (910,)

# Split data - last 4 for testing, rest for training
X_train, X_test = X[:-182], X[-182:]
y_train, y_test = y[:-182], y[-182:]
# Second split: 80% train, 20% validation (of the remaining 80%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2, 
    random_state=42,
    stratify=y_train
)
    
best_acc = 0
best_model = None
no_improve_epochs = 0
max_no_improve = 5

for n_trees in range(100, 501, 100):
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf.fit(X_train, y_train)
    
    val_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"{n_trees} trees - Validation Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_rf_model = rf
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
    
    if no_improve_epochs >= max_no_improve:
        print("Early stopping triggered.")
        break

best_acc = 0
best_model = None
no_improve_epochs = 0
max_no_improve = 3

C_values = [0.01, 0.1, 1, 10, 100]  # Try increasing C (regularization)

for C in C_values:
    svm = SVC(kernel='rbf', C=C, random_state=42)
    svm.fit(X_train, y_train)
    
    val_pred = svm.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"C={C} - Validation Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_svm_model = svm
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= max_no_improve:
        print("Early stopping triggered.")
        break

# Predict on test data
y_pred_rf = best_rf_model.predict(X_test)
y_pred_svm = best_svm_model.predict(X_test)


# Create comparison DataFrame
results_df = pd.DataFrame({
    'Sample': range(len(y_test)),
    'True Label': y_test,
    'RF Predicted': y_pred_rf,
    'SVM Predicted': y_pred_svm
})

# Compare predictions
print("Comparison of Predictions:")
print(results_df.head(13).to_string(index=False))  # Show 13 samples

print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print(f"RF Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

print("\nSVM Performance:")
print(classification_report(y_test, y_pred_svm))
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

# Plot confusion matrices side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, cmap='Blues', ax=ax1, xticks_rotation=45)
ax1.set_title("Random Forest")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, cmap='Reds', ax=ax2, xticks_rotation=45)
ax2.set_title("SVM")

plt.tight_layout()
plt.show()


joblib.dump(best_rf_model, 'random_forest_model.joblib')
joblib.dump(best_svm_model, 'svm_model.joblib')