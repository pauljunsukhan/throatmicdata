import pyaudio
import wave
import os
import csv
import time
from typing import List, Dict
import sounddevice as sd
import json
from pathlib import Path
import numpy as np

class ThroatMicRecorder:
    def __init__(self):
        self.CHUNK = 4096
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 10
        self.data_dir = "throatmic_data"
        self.metadata_file = "metadata.csv"
        self.progress_file = "recording_progress.json"
        self.prompts_file = "prompts.txt"
        self.p = pyaudio.PyAudio()
        
        # Audio monitoring settings
        self.CLIPPING_THRESHOLD = 0.95  # Normalized threshold for clipping
        self.show_levels = True  # Can be toggled
        self.clipping_window = 0.1  # Seconds to show clipping warning
        self._last_clipping_time = 0
        
    def setup_directories(self):
        """Create necessary directories and files if they don't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory: {self.data_dir}")
            
        if not os.path.exists(self.prompts_file):
            print("\nError: prompts.txt not found!")
            print("Please run 'python prepare_sentences.py' first to download Common Voice sentences.")
            exit(1)

    def load_prompts(self) -> list:
        """Load prompts from the prompts file"""
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def load_progress(self) -> int:
        """Load the current progress"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f).get('current_index', 0)
        return 0

    def save_progress(self, index: int):
        """Save the current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump({'current_index': index}, f)

    def get_audio_devices(self) -> List[Dict]:
        """List all available audio input devices"""
        devices = []
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Only input devices
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels']
                })
        return devices

    def select_audio_device(self) -> int:
        """Let user select audio input device"""
        devices = self.get_audio_devices()
        print("\nAvailable audio input devices:")
        for i, device in enumerate(devices):
            print(f"{i + 1}. {device['name']}")
        
        while True:
            try:
                selection = int(input("\nSelect device number: ")) - 1
                if 0 <= selection < len(devices):
                    return devices[selection]['index']
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number.")

    def get_next_file_number(self) -> int:
        """Get the next available file number"""
        existing_files = [f for f in os.listdir(self.data_dir) if f.endswith('.wav')]
        if not existing_files:
            return 1
        numbers = [int(f.split('_')[0]) for f in existing_files]
        return max(numbers) + 1 if numbers else 1

    def _normalize_audio_chunk(self, chunk):
        """Convert audio chunk to normalized float values"""
        data = np.frombuffer(chunk, dtype=np.int16)
        normalized = data.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]
        return normalized
        
    def _check_clipping(self, chunk):
        """Check if audio is clipping and return level info"""
        normalized = self._normalize_audio_chunk(chunk)
        peak = np.max(np.abs(normalized))
        is_clipping = peak > self.CLIPPING_THRESHOLD
        
        return {
            'is_clipping': is_clipping,
            'peak_level': peak,
            'level_indicator': self._get_level_indicator(peak)
        }
    
    def _get_level_indicator(self, peak):
        """Generate a visual level indicator"""
        if peak > self.CLIPPING_THRESHOLD:
            return "\r\033[91m█████ CLIPPING!\033[0m"  # Red warning
        
        level = int(peak * 20)  # 20 segments for visualization
        bars = "█" * level + "░" * (20 - level)
        
        if peak > 0.8:
            return f"\r\033[93m{bars}\033[0m"  # Yellow for high levels
        return f"\r\033[92m{bars}\033[0m"  # Green for normal levels
    
    def _clear_line(self):
        """Clear the current console line"""
        print('\r', end='', flush=True)
        
    def record_audio(self, device_index: int, prompt: str, take_number: int = 1) -> str:
        """Record audio and save to file"""
        file_number = self.get_next_file_number()
        take_suffix = f"_take{take_number}" if take_number > 1 else ""
        filename = f"{file_number:03d}_{prompt[:30].lower().replace(' ', '_')}{take_suffix}.wav"
        filepath = os.path.join(self.data_dir, filename)

        while True:
            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK
            )

            print(f"\nRecording prompt: {prompt}")
            print("Starting in: ", end='', flush=True)
            for count in [3, 2, 1]:
                print(f"{count}...", end='', flush=True)
                time.sleep(0.3)
            print("\nRecording...")
            if self.show_levels:
                print("Level:", end='', flush=True)

            frames = []
            clipping_detected = False
            try:
                for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                    try:
                        chunk = stream.read(self.CHUNK, exception_on_overflow=False)
                        frames.append(chunk)
                        
                        if self.show_levels:
                            level_info = self._check_clipping(chunk)
                            if level_info['is_clipping']:
                                clipping_detected = True
                            print(f"{level_info['level_indicator']}", end='', flush=True)
                            
                    except OSError as e:
                        print(f"\nWarning: Buffer overflow detected - some audio may be lost")
                        continue
                        
            except KeyboardInterrupt:
                print("\nRecording stopped by user")
                stream.stop_stream()
                stream.close()
                return self.record_audio(device_index, prompt, take_number)

            print("\nDone recording!")
            stream.stop_stream()
            stream.close()

            # Save the audio file
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))

            if clipping_detected:
                print("\n⚠️  Warning: Clipping was detected in this recording!")

            while True:
                print("\nOptions: (press Enter to save and continue)")
                print("1. Play recording")
                print("2. Re-record")
                print("3. Save and continue to next")
                print("4. Save and quit to menu")
                print("5. Discard and quit to menu")
                
                choice = input("\nSelect option: ").strip().lower()
                
                if choice == "" or choice == "3":  # Empty string for Enter key
                    return filepath
                    
                elif choice == "1":
                    print("\nPlaying back recording...")
                    sd.play(np.frombuffer(b''.join(frames), dtype=np.int16), self.RATE)
                    sd.wait()
                    continue
                    
                elif choice == "2":
                    print("\nRe-recording...")
                    take_number += 1
                    filename = f"{file_number:03d}_{prompt[:30].lower().replace(' ', '_')}_take{take_number}.wav"
                    filepath = os.path.join(self.data_dir, filename)
                    break
                    
                elif choice == "4":
                    self.update_metadata(filepath, prompt)
                    return None  # Signal to quit to menu
                    
                elif choice == "5":
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return None  # Signal to quit to menu
                    
                else:
                    print("Invalid option. Press Enter to continue, or select 1-5.")

    def toggle_level_meter(self):
        """Toggle the level meter display"""
        self.show_levels = not self.show_levels
        print(f"Level meter {'enabled' if self.show_levels else 'disabled'}")
        
    def update_metadata(self, filepath: str, text: str):
        """Update the metadata CSV file"""
        file_exists = os.path.exists(self.metadata_file)
        
        with open(self.metadata_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['audio_filepath', 'text'])
            writer.writerow([filepath, text])

    def show_progress(self, prompts: list, current_index: int):
        """Show detailed progress information"""
        total_prompts = len(prompts)
        completed = current_index
        
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                count = sum(1 for row in reader)
            total_time = count * self.RECORD_SECONDS / 60
            
            print(f"\nProgress:")
            print(f"Completed prompts: {completed}/{total_prompts} ({(completed/total_prompts)*100:.1f}%)")
            print(f"Total recordings: {count}")
            print(f"Total recording time: {total_time:.1f} minutes")
            
            if completed < total_prompts:
                print("\nNext prompt will be:")
                print(f'"{prompts[current_index]}"')
        else:
            print("\nNo recordings yet.")

    def test_audio_device(self, device_index: int) -> bool:
        """Test the audio device with current settings"""
        print("\nTesting audio device...")
        try:
            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK,
                stream_callback=None
            )
            
            # Record a short test
            print("Recording 3 second test...")
            for _ in range(0, int(self.RATE / self.CHUNK * 3)):  # 3 seconds test
                data = stream.read(self.CHUNK, exception_on_overflow=False)
            
            stream.stop_stream()
            stream.close()
            print("Audio device test successful!")
            return True
            
        except Exception as e:
            print(f"\nError testing audio device: {e}")
            print("Try selecting a different device or adjusting your system's audio settings.")
            return False

    def run(self):
        """Main recording process"""
        print("Welcome to Throat Mic Recording Tool")
        self.setup_directories()
        
        prompts = self.load_prompts()
        current_index = self.load_progress()
        
        if current_index >= len(prompts):
            print("All prompts have been recorded! Starting over from the beginning.")
            current_index = 0
            
        print(f"\nLoaded {len(prompts)} prompts. Starting from prompt {current_index + 1}")
        
        device_index = self.select_audio_device()
        
        # Test the selected device
        if not self.test_audio_device(device_index):
            print("\nAudio device test failed. Please try:")
            print("1. Selecting a different audio device")
            print("2. Checking your system's audio settings")
            print("3. Ensuring the device is properly connected")
            return
        
        while True:
            print("\nOptions:")
            print("1. Continue recording prompts")
            print("2. View progress")
            print("3. Toggle level meter")
            print("4. Exit")
            
            choice = input("\nSelect option: ")
            
            if choice == '1':
                while current_index < len(prompts):
                    prompt = prompts[current_index]
                    print("\n" + "="*50)
                    print(f"Prompt {current_index + 1}/{len(prompts)}:")
                    print(f"\n{prompt}\n")
                    print("="*50)
                    print("\nPress Enter to start recording, or 'q' to quit to menu")
                    
                    if input().lower() == 'q':
                        break
                    
                    filepath = self.record_audio(device_index, prompt)
                    if filepath is None:  # User chose to quit to menu
                        break
                        
                    self.update_metadata(filepath, prompt)
                    current_index += 1
                    self.save_progress(current_index)
                    
                if current_index >= len(prompts):
                    print("\nAll prompts have been recorded!")
                    
            elif choice == '2':
                self.show_progress(prompts, current_index)
            
            elif choice == '3':
                self.toggle_level_meter()
                    
            elif choice == '4':
                print("\nThank you for using Throat Mic Recording Tool!")
                break
            
            else:
                print("\nInvalid option. Please try again.")

    def __del__(self):
        """Cleanup PyAudio"""
        self.p.terminate()

if __name__ == "__main__":
    recorder = ThroatMicRecorder()
    recorder.run()
