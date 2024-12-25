"""
Main recorder module for throat mic data collection.
"""

import pyaudio
import wave
import sounddevice as sd
import numpy as np
import time
import csv
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set, Any
from .sentence_repo import SentenceRepository
from .sentence_filter import SentenceFilter
from .config import Config, config
import argparse
import os
import json
import soundfile as sf
from datetime import datetime
from tqdm import tqdm
import logging
from .analyzer import DatasetAnalytics, DatasetAnalyzer
import sys, tty, termios

# Setup logger
logger = logging.getLogger(__name__)

def list_audio_devices():
    """List all available audio input devices."""
    p = pyaudio.PyAudio()
    info = []
    
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:  # Only input devices
            info.append(device_info)
            print(f"{i}: {device_info['name']}")
            
    p.terminate()
    return info

class ThroatMicRecorder:
    def __init__(self, config: Config = None):
        """Initialize recorder with configuration.
        
        Args:
            config (Config, optional): Configuration instance. If None, uses default config.
        """
        self.config = config or config  # Use passed config or default
        self.p = pyaudio.PyAudio()
        self.audio_config = self.config.audio
        self.sentence_filter = SentenceFilter()
        # Audio settings
        self.CHUNK = self.config.audio.chunk_size
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = self.config.audio.channels
        self.RATE = self.config.audio.sample_rate
        self.RECORD_SECONDS = self.config.audio.duration
        self.CLIPPING_THRESHOLD = self.config.audio.clipping_threshold
        
        # Initialize paths
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.recordings_dir = self.data_dir / "recordings"
        self.metadata_dir = self.data_dir / "metadata"
        self.cache_dir = self.data_dir / "cache"
        self.metadata_file = self.metadata_dir / "metadata.csv"
        
        # Create directories
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata file if it doesn't exist
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(['audio_filepath', 'text', 'duration'])
        
        # Initialize repository
        self.repo = SentenceRepository(self.base_dir)
        
        # Load dataset cache and sync with metadata
        self.dataset_cache = self._load_dataset_cache()
        self._sync_with_metadata()
        
        # UI settings
        self.show_levels = True
        
        self.analyzer = DatasetAnalyzer(self.config)  # Create once and reuse
        self._invalidate_analysis_cache()  # Track if cache needs refresh
    
    def _load_dataset_cache(self) -> Dict[str, Any]:
        """Load cached dataset information."""
        cache_file = self.cache_dir / "dataset_cache.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return {"texts": [], "last_index": 0, "total_duration": 0.0}
    
    def _update_dataset_cache(self, text: str, duration: float) -> None:
        """Update the dataset cache with new recording."""
        self.dataset_cache["texts"].append(text)
        self.dataset_cache["last_index"] += 1
        self.dataset_cache["total_duration"] += duration
        
        with open(self.cache_dir / "dataset_cache.json", 'w') as f:
            json.dump(self.dataset_cache, f, indent=2)

    def _add_to_metadata(self, filepath: Path, text: str) -> None:
        """Add a recording to the metadata CSV file."""
        # Convert to relative path from data directory
        rel_path = filepath.relative_to(self.base_dir)
        
        # Get audio duration
        with sf.SoundFile(filepath) as f:
            duration = len(f) / f.samplerate
        
        # Update dataset cache
        self._update_dataset_cache(text, duration)
        
        with open(self.metadata_file, 'a', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([str(rel_path), text, duration])

    def get_next_file_number(self) -> int:
        """Get the next available file number"""
        return self.dataset_cache["last_index"] + 1

    def is_duplicate_text(self, text: str) -> bool:
        """Check if this text has already been recorded."""
        return text in self.dataset_cache["texts"]

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

    def select_audio_device(self) -> Optional[int]:
        """Let user select audio input device"""
        devices = self.get_audio_devices()
        print("\nAvailable audio input devices:")
        for i, device in enumerate(devices):
            print(f"{i + 1}. {device['name']}")
        print("Enter 'q' to quit")
        
        while True:
            try:
                selection = input("\nSelect device number: ").strip().lower()
                if selection == 'q':
                    return None
                selection = int(selection) - 1
                if 0 <= selection < len(devices):
                    return devices[selection]['index']
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number or 'q' to quit.")

    def _normalize_audio_chunk(self, chunk: bytes) -> np.ndarray:
        """Convert audio chunk to normalized float values"""
        data = np.frombuffer(chunk, dtype=np.int16)
        return data.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]
        
    def _check_clipping(self, chunk: bytes) -> Dict:
        """Check if audio is clipping and return level info"""
        normalized = self._normalize_audio_chunk(chunk)
        peak = np.max(np.abs(normalized))
        is_clipping = peak > self.CLIPPING_THRESHOLD
        
        return {
            'is_clipping': is_clipping,
            'peak_level': peak,
            'level_indicator': self._get_level_indicator(peak)
        }
    
    def _get_level_indicator(self, peak: float) -> str:
        """Generate a visual level indicator"""
        if peak > self.CLIPPING_THRESHOLD:
            return "\r\033[91m█████ CLIPPING!\033[0m"  # Red warning
        
        level = int(peak * 20)  # 20 segments for visualization
        bars = "█" * level + "░" * (20 - level)
        
        if peak > 0.8:
            return f"\r\033[93m{bars}\033[0m"  # Yellow for high levels
        return f"\r\033[92m{bars}\033[0m"  # Green for normal levels
    
    def record_audio(self, device_index: int, prompt: str) -> Tuple[Optional[Path], bool]:
        """Record audio and save to file. Returns (filepath, should_quit)"""
        file_number = self.get_next_file_number()
        filename = f"{file_number:05d}_{prompt[:30].lower().replace(' ', '_')}.wav"
        filepath = self.recordings_dir / filename

        # Get estimated duration for this sentence
        estimated_duration = self.sentence_filter.estimate_duration(prompt)
        recording_duration = estimated_duration  # Removed the 0.85 multiplier

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
            print(f"Estimated speaking time: {estimated_duration:.1f}s (Recording for: {recording_duration:.1f}s)")
            print("Starting in: ", end='', flush=True)
            for count in [3, 2, 1]:
                print(f"{count}...", end='', flush=True)
                time.sleep(0.3)
            print("\nRecording...")
            if self.show_levels:
                print("Level:", end='', flush=True)

            frames = []
            clipping_detected = False
            
            # Calculate total chunks based on estimated duration
            total_chunks = int(recording_duration * self.RATE / self.CHUNK)
            
            try:
                with tqdm(total=total_chunks, desc="Recording", position=1) as pbar:
                    for _ in range(total_chunks):
                        try:
                            chunk = stream.read(self.CHUNK, exception_on_overflow=False)
                            frames.append(chunk)
                            
                            if self.show_levels:
                                level_info = self._check_clipping(chunk)
                                if level_info['is_clipping']:
                                    clipping_detected = True
                                print(f"\r{level_info['level_indicator']}", end='', flush=True)
                                print("\033[K", end='')  # Clear to end of line
                            
                            pbar.update(1)
                                
                        except OSError as e:
                            print(f"\nWarning: Buffer overflow detected - some audio may be lost")
                            continue
                            
            except KeyboardInterrupt:
                print("\nRecording stopped by user")
                stream.stop_stream()
                stream.close()
                return self.record_audio(device_index, prompt)

            print("\nDone recording!")
            stream.stop_stream()
            stream.close()

            # Save the audio file
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))

            if clipping_detected:
                print("\n⚠️  Warning: Clipping was detected in this recording!")

            while True:
                print("\nOptions:")
                print("1. Play recording")
                print("2. Re-record")
                print("3. Save and continue to next")
                print("4. Save and quit to menu")
                print("5. Discard and quit to menu")
                print("6. Trash sentence and continue to next")
                print("\nPress Enter for option 3 (Save and continue)")
                print("Press Space for option 6 (Trash and continue)")
                
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                
                if ch == '\r' or ch == '3':  # Enter key or '3'
                    print("\nSaving recording and continuing...")
                    self._add_to_metadata(filepath, prompt)
                    self._invalidate_analysis_cache()  # New recording added
                    return filepath, False
                    
                elif ch == ' ' or ch == '6':  # Space or 6
                    print("\nTrashing sentence and continuing...")
                    filepath.unlink(missing_ok=True)  # Delete the recording
                    self.repo.trash_sentence(prompt)  # Move sentence to trash
                    return None, False  # Continue to next sentence
                    
                elif ch == '1':
                    print("\nPlaying back recording...")
                    sd.play(np.frombuffer(b''.join(frames), dtype=np.int16), self.RATE)
                    sd.wait()
                    continue
                    
                elif ch == '2':
                    print("\nRe-recording...")
                    filepath.unlink(missing_ok=True)
                    break
                    
                elif ch == '4':
                    print("\nSaving recording and returning to menu...")
                    self._add_to_metadata(filepath, prompt)
                    return filepath, True
                    
                elif ch == '5':
                    print("\nDiscarding recording and returning to menu...")
                    filepath.unlink(missing_ok=True)
                    return None, True
                    
                else:
                    print("Invalid option. Enter 1-6, or press Enter to save and continue.")

    def toggle_level_meter(self) -> None:
        """Toggle the level meter display"""
        self.show_levels = not self.show_levels
        print(f"Level meter {'enabled' if self.show_levels else 'disabled'}")

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
                frames_per_buffer=self.CHUNK
            )
            
            print("Recording 1 second test...")
            for _ in range(0, int(self.RATE / self.CHUNK * 1)):  # 1 second test
                stream.read(self.CHUNK, exception_on_overflow=False)
            
            stream.stop_stream()
            stream.close()
            print("Audio device test successful!")
            return True
            
        except Exception as e:
            print(f"\nError testing audio device: {e}")
            print("Try selecting a different device or adjusting your system's audio settings.")
            return False

    def run(self) -> None:
        """Main recording process"""
        print("Welcome to Throat Mic Recording Tool")
        
        device_index = self.select_audio_device()
        if device_index is None:
            print("\nExiting program...")
            return
        
        # Test the selected device
        if not self.test_audio_device(device_index):
            print("\nAudio device test failed. Please try:")
            print("1. Selecting a different audio device")
            print("2. Checking your system's audio settings")
            print("3. Ensuring the device is properly connected")
            return
        
        recording_session = False
        
        while True:
            if not recording_session:
                print("\nOptions:")
                print("1. Continue recording prompts")
                print("2. View progress")
                print("3. Toggle level meter")
                print("4. Manage sentences")
                print("5. Analyze dataset")
                print("6. Exit")
                
                choice = input("\nSelect option: ").strip()
                
                if choice == '1':
                    if self.repo.get_stats()['remaining_sentences'] == 0:
                        print("\nNo sentences available. Please download or add some sentences first.")
                        continue
                    recording_session = True
                
                elif choice == '2':
                    stats = self.repo.get_stats()
                    print(f"\nProgress:")
                    print(f"Completed prompts: {stats['recorded_sentences']}/{stats['total_sentences']} "
                          f"({stats['completion_percentage']:.1f}%)")
                    print(f"Total recording time: {stats['total_audio_time']:.1f} minutes")
                    print(f"Remaining sentences: {stats['remaining_sentences']}")
                    print(f"Trashed sentences: {stats['trashed_sentences']}")
                
                elif choice == '3':
                    self.toggle_level_meter()
                
                elif choice == '4':
                    self.manage_sentences()
                        
                elif choice == '5':
                    print("\nAnalyzing dataset...")
                    try:
                        if self.analysis_needs_refresh:
                            results = self.analyzer.analyze_dataset(self.config.dataset.metadata_file)
                            self.analysis_needs_refresh = False
                        else:
                            results = self.analyzer.last_analysis
                            print("Using cached analysis results...")
                            print(self.analyzer._format_analysis_display(self.analyzer.last_analysis))
                        
                        print("\nPress Space to generate a detailed report, or Enter to continue...")
                        import sys, tty, termios
                        fd = sys.stdin.fileno()
                        old_settings = termios.tcgetattr(fd)
                        try:
                            tty.setraw(sys.stdin.fileno())
                            ch = sys.stdin.read(1)
                        finally:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                        if ch == ' ':  # Only generate report on Space
                            try:
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                report_file = Path(self.config.dataset.recordings_dir).parent / f"analysis_report_{timestamp}.txt"
                                # Will use cached results if available
                                self.analyzer.generate_report(self.config.dataset.metadata_file, str(report_file))
                                print(f"\nReport saved to: {report_file}")
                            except Exception as e:
                                print(f"\nError generating report: {e}")
                    
                    except Exception as e:
                        print(f"\nError analyzing dataset: {e}")
                        # Don't show report option if analysis failed
                    
                    input("\nPress Enter to continue...")
                    continue
                elif choice == '6':
                    return
                
                else:
                    print("\nInvalid option. Please enter a number between 1 and 6.")
                    continue
            
            else:  # Recording session
                # Get next sentence
                sentence = self.repo.get_next_sentence()
                if not sentence:
                    print("\nNo more sentences available!")
                    recording_session = False
                    continue
                
                # Display prompt
                print("\n" + "="*50)
                stats = self.repo.get_stats()
                print(f"Prompt {stats['recorded_sentences'] + 1}/{stats['total_sentences']}:")
                print(f"\n{sentence}\n")
                print("="*50)
                print("\nPress Enter to start recording, 't' to trash this sentence, or 'q' to quit to menu")
                
                choice = input().strip().lower()
                if choice == 'q':
                    recording_session = False
                    continue
                elif choice == 't':
                    self.repo.trash_sentence(sentence)
                    print(f"\nSentence moved to trash.")
                    continue
                
                # Record the prompt
                filepath, should_quit = self.record_audio(device_index, sentence)
                
                if filepath:
                    # Mark the sentence as used only if we saved the recording
                    self.repo.mark_sentence_used(sentence)
                
                if should_quit:
                    recording_session = False

    def manage_sentences(self) -> None:
        """Manage sentences in the repository."""
        while True:
            print("\nSentence Management:")
            print("1. Download more sentences")
            print("2. Add custom sentences")
            print("3. View sentence stats")
            print("4. View trashed sentences")
            print("5. Analyze dataset")
            print("6. Back to main menu")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                try:
                    count = int(input("\nHow many sentences to download? (default: 100): ").strip() or "100")
                    if count <= 0:
                        print("Please enter a positive number.")
                        continue
                    
                    added = self.repo.download_sentences(count)
                    print(f"\nSuccessfully added {added} new sentences.")
                    
                except ValueError:
                    print("Please enter a valid number.")
                except Exception as e:
                    print(f"Error downloading sentences: {e}")
            
            elif choice == '2':
                print("\nEnter sentences (one per line, empty line to finish):")
                sentences = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    sentences.append(line)
                
                if sentences:
                    added = self.repo.add_sentences(sentences)
                    print(f"\nSuccessfully added {added} new sentences.")
                else:
                    print("No sentences were added.")
            
            elif choice == '3':
                stats = self.repo.get_stats()
                print(f"\nSentence Statistics:")
                print(f"Total sentences: {stats['total_sentences']}")
                print(f"Available sentences: {stats['remaining_sentences']}")
                print(f"Used sentences: {stats['recorded_sentences']}")
                print(f"Trashed sentences: {stats['trashed_sentences']}")
            
            elif choice == '4':
                if not self.repo.trashed_sentences:
                    print("\nNo trashed sentences.")
                else:
                    print("\nTrashed Sentences:")
                    for i, sentence in enumerate(self.repo.trashed_sentences, 1):
                        print(f"{i}. {sentence}")
            
            elif choice == '5':
                print("\nAnalyzing dataset...")
                try:
                    results = self.analyzer.analyze_dataset(self.config.dataset.metadata_file)
                    
                    print("\nPress Space to generate a detailed report, or Enter to continue...")
                    import sys, tty, termios
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setraw(sys.stdin.fileno())
                        ch = sys.stdin.read(1)
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                    if ch == ' ':  # Only generate report on Space
                        try:
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            report_file = Path(self.config.dataset.recordings_dir).parent / f"analysis_report_{timestamp}.txt"
                            # Will use cached results if available
                            self.analyzer.generate_report(self.config.dataset.metadata_file, str(report_file))
                            print(f"\nReport saved to: {report_file}")
                        except Exception as e:
                            print(f"\nError generating report: {e}")
                    
                except Exception as e:
                    print(f"\nError analyzing dataset: {e}")
                    # Don't show report option if analysis failed
                
                input("\nPress Enter to continue...")
                continue
            
            elif choice == '6':
                return
            
            else:
                print("\nInvalid option. Please enter a number between 1 and 6.")

    def __del__(self) -> None:
        """Cleanup PyAudio"""
        if hasattr(self, 'p'):
            self.p.terminate()

    def _sync_with_metadata(self) -> None:
        """Sync sentence repository with existing metadata."""
        if not self.metadata_file.exists():
            return
            
        try:
            with open(self.metadata_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Mark each sentence in metadata as used
                    self.repo.mark_sentence_used(row['text'])
                    
                    # Update cache if needed
                    if row['text'] not in self.dataset_cache['texts']:
                        self.dataset_cache['texts'].append(row['text'])
                        self.dataset_cache['total_duration'] += float(row['duration'])
                        
            # Update last index based on number of recordings
            self.dataset_cache['last_index'] = len(self.dataset_cache['texts'])
            
            # Save updated cache
            with open(self.cache_dir / "dataset_cache.json", 'w') as f:
                json.dump(self.dataset_cache, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Error syncing with metadata: {e}")
            # Continue anyway - this is not fatal

    def record_sentence(self, sentence: str) -> Optional[str]:
        """Record a single sentence."""
        try:
            # Estimate duration for this specific sentence
            estimated_duration = self.sentence_filter.estimate_duration(sentence)
            # Add a small buffer (e.g., 20% extra time)
            recording_duration = estimated_duration * 1.2

            print(f"\nPlease read: {sentence}")
            print(f"Estimated duration: {estimated_duration:.1f}s (Recording for: {recording_duration:.1f}s)")
            print("Press Enter to start recording...")
            input()

            # Use estimated duration instead of fixed 10s
            frames = self._record_audio(duration=recording_duration)
            if not frames:
                return None

            # Save the recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.wav"
            filepath = os.path.join(self.config.dataset.recordings_dir, filename)
            
            self._save_audio(frames, filepath)
            
            return filepath

        except Exception as e:
            logger.error(f"Error recording sentence: {e}")
            return None

    def _record_audio(self, duration: float) -> Optional[List[bytes]]:
        """Record audio for specified duration."""
        try:
            frames = []
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.audio_config.channels,
                rate=self.audio_config.sample_rate,
                input=True,
                frames_per_buffer=self.audio_config.chunk_size,
                input_device_index=self.device_index
            )

            # Calculate number of chunks based on estimated duration
            num_chunks = int((duration * self.audio_config.sample_rate) / self.audio_config.chunk_size)
            
            print("\nRecording...")
            progress = tqdm(total=num_chunks, desc="Recording")
            
            for _ in range(num_chunks):
                data = stream.read(self.audio_config.chunk_size)
                frames.append(data)
                progress.update(1)
            
            progress.close()
            stream.stop_stream()
            stream.close()
            
            return frames

        except Exception as e:
            logger.error(f"Error during recording: {e}")
            return None

    def _invalidate_analysis_cache(self):
        """Mark the analysis cache as needing refresh."""
        self.analysis_needs_refresh = True

def main():
    """Main entry point for the recording CLI."""
    parser = argparse.ArgumentParser(
        description="Record throat microphone audio data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help=f"Data directory (default: {config.paths.data_dir})"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=config.audio.duration,
        help=f"Recording duration in seconds (default: {config.audio.duration})"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    
    args = parser.parse_args()
    
    if args.data_dir:
        os.environ["THROATMIC_DATA_DIR"] = args.data_dir
    
    if args.list_devices:
        list_audio_devices()
        return
    
    recorder = ThroatMicRecorder()
    recorder.run()

if __name__ == "__main__":
    main()
