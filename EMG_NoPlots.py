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
    
    # Apply preprocessing to all data
    emg_cleaned = np.zeros_like(emg_data)
    for i in range(num_channels):
        emg_cleaned[:, i] = preprocess_emg_channel(emg_data[:, i], estimated_fs)
    # Rectify
    # emg_rect = np.abs(emg_cleaned)
    emg_rect = emg_cleaned
    emg_smoothed = np.zeros_like(emg_rect)  # shape: (num_samples, num_channels)
    for i in range(num_channels):
        emg_smoothed[:, i] = moving_average(emg_rect[:, i])
    '''
    # FFT, Power spectrum and Wavelet transform for all emg data 
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
    #'''
    
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
                psd_trial.append(np.trapezoid(Pxx, f))  # Total power = area under PSD curve
                
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
        mean_values.append(class_mean)
        std_values.append(class_std)
        fft_values.append(class_fft)
        power_spectra.append(class_power)
        power_freqs.append(class_freqs)
        mean_freqs.append(class_mean_f)
        median_freqs.append(class_median_f)
        psd_values.append(class_psd)
        wavelet_transforms.append(class_wavelet)
    #'''

    # == Create feature vector ==
    X = []  # Feature vectors
    y = []  # Labels (class ID)
    n_classes = len(rms_values)

    for class_id in range(n_classes):
        n_trials = len(rms_values[class_id])
        for trial_id in range(n_trials):
           
            basic_features = np.concatenate([
                rms_values[class_id][trial_id],
                # mav_values[class_id][trial_id],
                # wav_values[class_id][trial_id],
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
            # print("Feature Vector shape:", trial_features.shape)
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
X_train, X_test = X[:-130], X[-130:]
y_train, y_test = y[:-130], y[-130:]
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
print(f"RF test accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

print("\nSVM Performance:")
print(classification_report(y_test, y_pred_svm))
print(f"SVM test accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

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
