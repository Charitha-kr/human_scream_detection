import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

class MLScreamDetector:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.feature_count = 40
        self.model_path = "models/scream_detector.joblib"
    
    def extract_features(self, audio_data, sample_rate):
        # Basic features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=self.feature_count)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # Additional features
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate))
        
        # Combine all features
        features = np.concatenate([
            mfccs_scaled,
            np.array([zero_crossing_rate]),
            np.array([spectral_centroids]),
            np.array([spectral_rolloff]),
            np.array([spectral_bandwidth])
        ])
        
        return features.reshape(1, -1)
    
    def augment_audio(self, audio_data, sr):
        augmented = []
        # Original audio
        augmented.append(audio_data)
        
        # Time stretched
        augmented.append(librosa.effects.time_stretch(audio_data, rate=0.9))
        augmented.append(librosa.effects.time_stretch(audio_data, rate=1.1))
        
        # Pitch shifted
        augmented.append(librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=1))
        augmented.append(librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=-1))
        
        return augmented
    
    def train(self, scream_folder, normal_folder):
        features = []
        labels = []
        
        # Process scream audio files with augmentation
        for file in os.listdir(scream_folder):
            if file.endswith('.wav'):
                audio_path = os.path.join(scream_folder, file)
                audio, sr = librosa.load(audio_path)
                
                # Get augmented versions
                augmented_audio = self.augment_audio(audio, sr)
                for aug_audio in augmented_audio:
                    features.append(self.extract_features(aug_audio, sr).flatten())
                    labels.append(1)
        
        # Process normal audio files with augmentation
        for file in os.listdir(normal_folder):
            if file.endswith('.wav'):
                audio_path = os.path.join(normal_folder, file)
                audio, sr = librosa.load(audio_path)
                
                # Get augmented versions
                augmented_audio = self.augment_audio(audio, sr)
                for aug_audio in augmented_audio:
                    features.append(self.extract_features(aug_audio, sr).flatten())
                    labels.append(0)
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.2f}")
        print(f"Testing accuracy: {test_score:.2f}")
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False
        
    def predict(self, audio_data, sample_rate):
        features = self.extract_features(audio_data, sample_rate)
        if not self.load_model():
            raise Exception("No trained model found. Please train the model first.")
        confidence = self.model.predict_proba(features)[0][1]
        return confidence > 0.45, confidence  # Adjusted threshold to 0.45
    
    def test_detection(self, audio_data, sample_rate):
        """Test method to simulate scream detection"""
        is_scream, confidence = self.predict(audio_data, sample_rate)
        
        print("\nTest Results:")
        print(f"Scream Detected: {'YES' if is_scream else 'NO'}")
        print(f"Confidence Score: {confidence:.2f}")
        
        # Detailed feature analysis
        features = self.extract_features(audio_data, sample_rate)
        mfcc_mean = np.mean(features[0][:self.feature_count])
        zero_crossing = features[0][self.feature_count]
        
        print("\nFeature Analysis:")
        print(f"MFCC Mean: {mfcc_mean:.2f}")
        print(f"Zero Crossing Rate: {zero_crossing:.2f}")
        
        return is_scream, confidence