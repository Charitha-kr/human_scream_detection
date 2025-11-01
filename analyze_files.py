import os
from ml_detector import MLScreamDetector
from scream_detector import ScreamDetector
import wave
import numpy as np
from colorama import init, Fore

def display_result(value, threshold=0.45):
    return f"{Fore.GREEN if value < threshold else Fore.RED}{value:.2f}{Fore.RESET}"

def analyze_wav_file(file_path):
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
    
    ml_detector = MLScreamDetector()
    traditional_detector = ScreamDetector()
    
    is_scream, ml_confidence = ml_detector.predict(audio_data, sample_rate)
    trad_result, energy = traditional_detector.detect_scream(audio_data, sample_rate)
    
    return sample_rate, len(audio_data)/sample_rate, ml_confidence, energy

def main():
    init()  # Initialize colorama
    base_dir = "c:\\Users\\chari\\human_scream\\data"
    
    print("\n=== Analyzing Scream Files ===")
    scream_dir = os.path.join(base_dir, "screams")
    for file in os.listdir(scream_dir):
        if file.endswith('.wav'):
            print(f"\nFile: {file}")
            sr, duration, conf, energy = analyze_wav_file(os.path.join(scream_dir, file))
            print(f"Sample Rate: {sr} Hz")
            print(f"Duration: {duration:.2f} seconds")
            print(f"ML Confidence: {display_result(conf)}")
            print(f"Energy Level: {display_result(energy)}")
            print("-" * 40)
    
    print("\n=== Analyzing Normal Files ===")
    normal_dir = os.path.join(base_dir, "normal")
    for file in os.listdir(normal_dir):
        if file.endswith('.wav'):
            print(f"\nFile: {file}")
            sr, duration, conf, energy = analyze_wav_file(os.path.join(normal_dir, file))
            print(f"Sample Rate: {sr} Hz")
            print(f"Duration: {duration:.2f} seconds")
            print(f"ML Confidence: {display_result(conf)}")
            print(f"Energy Level: {display_result(energy)}")
            print("-" * 40)

if __name__ == "__main__":
    main()