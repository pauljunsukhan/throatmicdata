import pyaudio
import wave
import os
import csv
import time
from typing import List, Dict
import sounddevice as sd
import json
from pathlib import Path

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

    def record_audio(self, device_index: int, prompt: str) -> str:
        """Record audio and save to file"""
        file_number = self.get_next_file_number()
        filename = f"{file_number:03d}_{prompt[:30].lower().replace(' ', '_')}.wav"
        filepath = os.path.join(self.data_dir, filename)

        # Configure the stream with error handling
        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.CHUNK,
            stream_callback=None
        )

        print(f"\nRecording prompt: {prompt}")
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("Recording...")

        frames = []
        for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
            except OSError as e:
                print(f"\nWarning: Buffer overflow detected - some audio may be lost")
                # Continue recording despite the overflow
                continue

        print("Done recording!")

        try:
            stream.stop_stream()
            stream.close()

            # Save the audio file
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))

            return filepath

        except Exception as e:
            print(f"\nError saving recording: {e}")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            raise

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
            print("3. Exit")
            
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
                    self.update_metadata(filepath, prompt)
                    current_index += 1
                    self.save_progress(current_index)
                    
                    if current_index < len(prompts):
                        print("\nPress Enter for next prompt, or 'q' to quit to menu")
                        if input().lower() == 'q':
                            break
                
                if current_index >= len(prompts):
                    print("\nAll prompts have been recorded!")
                    
            elif choice == '2':
                self.show_progress(prompts, current_index)
                    
            elif choice == '3':
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
