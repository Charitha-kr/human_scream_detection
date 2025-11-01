from audio_processor import AudioProcessor
from scream_detector import ScreamDetector
from ml_detector import MLScreamDetector
import time
import numpy as np
from colorama import init, Fore, Back
import datetime
import wave
import os
import sys  # Added sys import

def display_meter(value, threshold=0.45):  # Adjusted from 0.3 to 0.45
    bars = int(value * 20)
    meter = '█' * bars + '░' * (20 - bars)
    color = Fore.GREEN if value < threshold else Fore.RED
    return f"{color}{meter} {value:.2f}{Fore.RESET}"

def save_detection_event(audio_data, sample_rate, ml_confidence, energy_level):
    # Create directories if they don't exist
    os.makedirs("detections/audio", exist_ok=True)
    os.makedirs("detections/logs", exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save audio file with better naming
    audio_path = f"detections/audio/scream_{timestamp.replace(':', '-')}.wav"
    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    # Enhanced logging with more details
    log_path = "detections/logs/detection_log.txt"
    with open(log_path, 'a') as f:
        f.write(f"\n=== New Detection Event ===")
        f.write(f"\nTimestamp: {timestamp}")
        f.write(f"\nML Confidence: {ml_confidence:.4f}")
        f.write(f"\nEnergy Level: {energy_level:.4f}")
        f.write(f"\nPeak Amplitude: {np.max(np.abs(audio_data)):.4f}")
        f.write(f"\nRecording Duration: {len(audio_data)/sample_rate:.2f} seconds")
        f.write(f"\nSample Rate: {sample_rate} Hz")
        f.write(f"\nAudio File: {os.path.basename(audio_path)}")
        f.write("\n" + "-"*50)

def main():
    init()  # Initialize colorama
    audio_processor = AudioProcessor()
    traditional_detector = ScreamDetector()
    ml_detector = MLScreamDetector()
    
    print("Loading trained model...")
    ml_detector.load_model()
    print("Starting enhanced scream detection system...")
    print("\nPress Ctrl+C to stop monitoring")
    
    try:
        while True:
            print("\033[H\033[J")  # Clear screen
            print("🎤 Monitoring Audio...")
            audio_data = audio_processor.record_audio(duration=3)
            
            traditional_result, traditional_energy = traditional_detector.detect_scream(
                audio_data, 
                audio_processor.rate
            )
            
            ml_result, ml_confidence = ml_detector.predict(
                audio_data, 
                audio_processor.rate
            )
            
            print("\nDetection Levels:")
            print(f"ML Confidence:    {display_meter(ml_confidence)}")
            print(f"Energy Level:     {display_meter(traditional_energy)}")
            
            if traditional_result and ml_result:
                print(f"\n{Back.RED}⚠️ SCREAM DETECTED!{Back.RESET}")
                save_detection_event(audio_data, audio_processor.rate, ml_confidence, traditional_energy)
                print("\nDetection saved!")
            
            time.sleep(10)
    
    except KeyboardInterrupt:
        print("\nStopping scream detection system...")

def analyze_audio_file(filename):
    print(f"\nAnalyzing audio file: {filename}")
    
    audio_path = os.path.join("c:\\Users\\chari\\human_scream\\data\\screams", filename)
    
    if not os.path.exists(audio_path):
        print(f"Error: File not found at {audio_path}")
        return None, None, None
        
    # Load and analyze the audio file
    with wave.open(audio_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize
    
    # Test with both detectors
    ml_detector = MLScreamDetector()
    traditional_detector = ScreamDetector()
    
    is_scream, ml_confidence = ml_detector.test_detection(audio_data, sample_rate)
    trad_result, energy = traditional_detector.detect_scream(audio_data, sample_rate)
    
    # Save detection results
    if is_scream and trad_result:
        save_detection_event(audio_data, sample_rate, ml_confidence, energy)
        print("\nDetection saved!")
    
    print(f"\nFrequency Analysis:")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
    
    return is_scream, ml_confidence, energy

def view_detection_history():
    log_path = "detections/logs/detection_log.txt"
    audio_path = "detections/audio"
    
    print("\n=== Scream Detection History ===")
    
    # Check for log file
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()
            print(content)
    else:
        print("No detection logs found.")
    
    # List recorded audio files
    print("\n=== Recorded Audio Files ===")
    if os.path.exists(audio_path):
        files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
        for file in sorted(files, reverse=True):
            file_path = os.path.join(audio_path, file)
            size = os.path.getsize(file_path) / 1024  # Convert to KB
            timestamp = ' '.join(file.split('_')[1:]).replace('.wav', '')
            print(f"File: {file}")
            print(f"Time: {timestamp}")
            print(f"Size: {size:.2f} KB")
            print("-" * 30)
    else:
        print("No recorded audio files found.")

# Modify the main() function to include a command to view history
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--history":
        view_detection_history()
    else:
        analyze_audio_file("1rachit.wav")
        main()