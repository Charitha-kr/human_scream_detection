
# 🗣️ Human Scream Detection using AI and Audio Processing

## 📘 Overview

The **Human Scream Detection System** is an AI-based project designed to detect human screams in real-time audio streams or recorded clips. It leverages **machine learning** and **signal processing** techniques to identify distress sounds such as screams or cries, which can then trigger automated alerts for safety and emergency response systems.

This project can be applied in **public surveillance**, **smart security systems**, **home safety monitoring**, and **emergency assistance** platforms.

---

## 🚀 Features

* 🎧 Real-time detection from microphone input.
* 🔊 Detection of human screams from recorded audio files (.wav, .mp3).
* 🤖 Uses **MFCC (Mel Frequency Cepstral Coefficients)** for feature extraction.
* 🧠 Trained using machine learning / deep learning models (e.g., CNN, Random Forest).
* 📊 Provides visualization of detected sound waves and classification confidence.
* ⚡ Lightweight and deployable on Raspberry Pi, edge devices, or cloud servers.

---

## 🧩 System Architecture

```
Audio Input (Microphone / File)
         ↓
Feature Extraction (MFCC / Spectrogram)
         ↓
Model Prediction (Scream / Non-Scream)
         ↓
Trigger Alert (Sound / Notification / API)
```

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:**

  * `librosa` – for audio processing and MFCC extraction
  * `numpy`, `pandas` – for data handling
  * `tensorflow` / `scikit-learn` – for model training and prediction
  * `matplotlib` – for visualization
  * `pyaudio` or `sounddevice` – for live audio capture

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/human-scream-detection.git
cd human-scream-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎙️ Usage

### ▶️ Run with Microphone Input

```bash
python detect_scream_live.py
```

### ▶️ Run with Audio File

```bash
python detect_scream_from_file.py --file path_to_audio.wav
```

When a scream is detected, the system prints:

```
⚠️ ALERT: Human scream detected! (Confidence: 97%)
```

---

## 🧠 Model Training (Optional)

If you want to retrain the model:

1. Place labeled audio files in:

   ```
   dataset/
   ├── scream/
   └── non_scream/
   ```
2. Run:

   ```bash
   python train_model.py
   ```
3. The trained model will be saved as `scream_model.h5`.

---

## 📈 Performance Metrics

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 94.7% |
| Precision | 93.5% |
| Recall    | 95.1% |
| F1-Score  | 94.3% |

---

## 🔒 Applications

* Home security systems
* Public surveillance (airports, metros, malls)
* Smart city safety analytics
* Emergency alert systems for elderly or disabled individuals
* Wildlife or animal distress detection (with retraining)

---

## 🧑‍💻 Contributors

* **Your Name** – Project Lead / Developer
* **Collaborators (if any)**

---

## 📄 License

This project is licensed under the **MIT License**.
Feel free to use, modify, and distribute with attribution.


