# EMG and EEG Signal Processing and Analysis

## Project Overview
This project explores advanced signal processing and feature extraction techniques for electromyography (EMG) signals in a biomedical context. The main focus is on real-time identification and classification of hand gestures based on EMG signals collected from multiple sensors. This work contributes to the development of robust EMG processing pipelines that can be used for assistive technologies, rehabilitation devices, or prosthetic control.

## Dataset

### EMG daata:

We used publicly available data from the UCI Machine Learning Repository:  
🔗 [EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)

- **Data Description**:
  - EMG readings from **8 sensors**, collected from **35 patients**, which include **7 different hand movement classes**.
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
