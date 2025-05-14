# EMG and EEG Signal Processing and Analysis

## Project Overview
This project aims to better understand the human neuromuscular system and brain activity by analyzing physiological signals and using them to classify different motions. In our implementation, we worked with both EMG and EEG data—applying preprocessing and feature extraction techniques—then used that data to train machine learning models for gesture classification. We also built a real-time system for detecting hand gestures using EMG sensors.


## Datasets

We used publicly available data from the UCI Machine Learning Repository:  
🔗 [EMG data for different hand gestures](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)
🔗 [EEG data for different eye states](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)

- **Data Description**:
  - **EMG**: readings from **8 sensors**, collected from **35 patients**, which include **7 different hand movement classes**.
  - **EEG**: readings from **14 channels**, collected from **1 patient**, which include **2 eye states** open and closed.
---

## Our Codes:

### Offline Processing Pipeline
1. **Filtering**:
   - Notch filter at **50 Hz** and **60 Hz** to remove powerline interference
   - Bandpass filter from **20–450 Hz** 

2. **Smoothing**:
   - Signals are smoothed to reduce noise while preserving useful activity patterns

3. **Segmentation**:
   - Data is split by class label
   - Within each class, signals are further split into **trials** based on time gaps

4. **Feature Extraction**:
   For each trial, we compute:
   - Root Mean Square (RMS)
   - Mean Absolute Value (MAV)
   - Waveform Length (WAV)
   - Mean, Standard Deviation, Variance
   - FFT
   - Power Spectral Density (PSD)
   - Mean Frequency, Median Frequency

   These features are compiled into a feature vector for each class.

5. **Classification**:
   - A machine learning model is trained on feature vectors
   - Tested using data from new patients to evaluate generalization

---

### Real-Time Prediction Pipeline
1. **Model Loading**:
   - Loads the trained classifier model

2. **Real-Time Data Acquisition**:
   - EMG data is collected in real time and saved to `.csv` files

3. **Filtering and Feature Extraction**:
   - Applies the same filtering and feature pipeline as offline
   - Extracts features from new data

4. **Gesture Prediction**:
   - The classifier predicts the motion based on the processed EMG signals

---

## Team Members
- Lydia Safwat Ghali - 20P6621
- Shahd Mohsen - 22P0294
- Ziad Walid ElHanafy - 20P4035
