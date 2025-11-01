import numpy as np
import pyaudio
import wave
from scipy.io import wavfile
from scipy import signal
import datetime
import os
import time  # Added time import

class AudioProcessor:
    def __init__(self):
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 44100
        self.chunk = 4096
        self.history = []
        self.max_history = 10
        
        # Use absolute paths
        self.base_dir = "c:\\Users\\chari\\human_scream\\data"
        self.history_dir = os.path.join(self.base_dir, "history")
        self.history_file = os.path.join(self.history_dir, "detection_history.txt")
        self.audio_dir = os.path.join(self.base_dir, "recordings")
        
        # Create all directories at once
        try:
            os.makedirs(self.history_dir, exist_ok=True)
            os.makedirs(self.audio_dir, exist_ok=True)
            print(f"Created directories in: {self.base_dir}")
            
            # Create history file
            with open(self.history_file, 'a') as f:
                if os.path.getsize(self.history_file) == 0:
                    f.write("=== Audio Recording History ===\n")
                    print(f"Initialized history file: {self.history_file}")
                
        except Exception as e:
            print(f"Error creating directories: {str(e)}")
        
        try:
            self.audio = pyaudio.PyAudio()
            default_device = self.audio.get_default_input_device_info()
            print(f"Using audio device: {default_device['name']}")
        except Exception as e:
            print(f"Error initializing audio: {str(e)}")
            raise

    def record_audio(self, duration=10):
        stream = None
        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=None
            )

            print("\nRecording...")
            frames = []
            total_frames = int(self.rate / self.chunk * duration)
            
            # Store recording timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Slower updates with longer pauses
            for i in range(total_frames):
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))
                    if i % 20 == 0:  # Update every 20th frame instead of 10th
                        time.sleep(0.5)  # Increased delay from 0.1 to 0.5 seconds
                except Exception as e:
                    continue  # Remove warning print

            audio_data = np.concatenate(frames)
            
            # Add to history
            self.history.append({
                'timestamp': timestamp,
                'data': audio_data,
                'max_amplitude': np.max(np.abs(audio_data)),
                'mean_energy': np.mean(audio_data ** 2)
            })
            
            # Keep only last N recordings
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Save history to file
            self._save_history()
            
            return audio_data
        
        except Exception as e:
            print(f"Error recording audio: {str(e)}")
            return np.array([])
        
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass

    def _save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                f.write("\n=== Audio Recording History ===\n")
                for entry in self.history:
                    f.write(f"\nTimestamp: {entry['timestamp']}")
                    f.write(f"\nMax Amplitude: {entry['max_amplitude']:.4f}")
                    f.write(f"\nMean Energy: {entry['mean_energy']:.4f}")
                    f.write("\n" + "-"*30)
        except Exception as e:
            print(f"Error saving history: {str(e)}")

    def __del__(self):
        try:
            self.audio.terminate()
        except:
            pass