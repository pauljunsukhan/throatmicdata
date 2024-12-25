#!/usr/bin/env python3

import argparse
import csv
import json
import sounddevice as sd
import soundfile as sf
import numpy as np
import fcntl
import select
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from collections import OrderedDict
import contextlib
import os
import pandas as pd
from src.sentence_filter import SentenceFilter

class RecordingCleaner:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.recordings_dir = Path(self.config['dataset']['recordings_dir'])
        self.metadata_file = Path(self.config['dataset']['metadata_file'])
        self.sentences_file = Path("data/repository/sentences.json")
        self.used_sentences_file = Path("data/repository/used_sentences.json")
        self.trashed_sentences_file = Path("data/repository/trashed_sentences.json")
        
        # Load current state
        self.metadata = self._load_metadata()  # Actually load the metadata
        self.sentences = self._load_sentences()
        self.used_sentences = self._load_used_sentences()
        self.trashed_sentences = self._load_trashed_sentences()
        
    def run(self):
        """Main cleanup interface"""
        print("\nWelcome to Recording Cleanup Tool")
        
        while True:
            print("\nOptions:")
            print("1. Review recordings")
            print("2. Show statistics")
            print("3. Export cleanup report")
            print("4. Standardize British to American spellings in metadata.csv")
            print("5. Validate metadata & JSON")
            print("6. Cleanup duplicates & orphans")
            print("7. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                self.review_recordings()
            elif choice == '2':
                self.show_statistics()
            elif choice == '3':
                self.export_report()
            elif choice == '4':
                standardize_metadata_spelling(self.metadata_file)
            elif choice == '5':
                self.validate_metadata_and_json()
                # Return to menu afterwards
            elif choice == '6':
                self.cleanup_orphans_and_duplicates()
            elif choice == '7':
                print("\nExiting cleanup tool...")
                break
            else:
                print("\nInvalid option. Please try again.")

    def review_recordings(self):
        """Interactive recording review interface"""
        # Sort recordings numerically by the prefix before the underscore
        recordings = sorted(
            list(self.recordings_dir.glob("*.wav")),
            key=self._numeric_sort_key
        )
        
        if not recordings:
            print("\nNo recordings found!")
            return
                
        current_idx = 0
        while current_idx < len(recordings):
            # Recheck recordings list length in case of deletions
            if current_idx >= len(recordings):
                break
                    
            recording = recordings[current_idx]
            if not recording.exists():
                # Skip if file was deleted
                current_idx += 1
                continue
                    
            metadata_entry = self.metadata.get(recording.stem)
                
            if not metadata_entry:
                print(f"\nWarning: No metadata found for {recording.name}")
                choice = input("Delete orphaned recording? [y/N]: ").strip().lower()
                if choice == 'y':
                    recording.unlink()
                    print(f"Deleted orphaned recording {recording.name}")
                    current_idx += 1  # Continue to the next recording
                    continue  # Refresh the list without breaking the loop
                current_idx += 1
                continue
                
            print("\n" + "="*50)
            print(f"Recording {current_idx + 1}/{len(recordings)}: {recording.name}")
            print(f"Transcript: {metadata_entry['text']}")
            print("="*50)
                
            try:
                self._play_audio(recording)
            except (sd.PortAudioError, sf.SoundFileError) as e:
                print(f"\nError playing audio: {e}")
                choice = input("Try again? [Y/n]: ").strip().lower()
                if choice != 'n':
                    continue
                
            print("\nOptions:")
            print("1. Keep and continue")
            print("2. Delete recording")
            print("3. Replay audio")
            print("4. Previous recording")
            print("5. Return to menu")
            print("6. Jump to recording number")
                
            choice = input("\nChoice: ").strip()
                
            if choice == '1':
                current_idx += 1
            elif choice == '2':
                recording_num = int(recording.stem.split('_')[0])
                if self._delete_recording(recording):
                    print(f"\nDeleted {recording.name}")
                    # Don't break, just refresh the list and stay at current index to review replacement
                    recordings = sorted(list(self.recordings_dir.glob("*.wav")))
                    continue
                else:
                    print("\nFailed to delete recording")
            elif choice == '3':
                continue  # Will replay on next loop
            elif choice == '4':
                current_idx = max(0, current_idx - 1)
            elif choice == '5':
                return
            elif choice == '6':
                try:
                    target_num = int(input("Enter recording number to jump to: ").strip())
                    if target_num < 1:
                        print("\nInvalid recording number. Must be positive.")
                        continue
                            
                    # Find the recording with this number
                    target_recording = None
                    for idx, rec in enumerate(recordings):
                        try:
                            rec_num = int(rec.stem.split('_')[0])
                            if rec_num == target_num:
                                target_recording = rec
                                current_idx = idx
                                break
                        except ValueError:
                            continue
                                
                    if target_recording is None:
                        print(f"\nNo recording found with number {target_num}")
                except ValueError:
                    print("\nInvalid input. Please enter a number.")
            else:
                print("\nInvalid option. Please try again.")

    def _delete_recording(self, recording_path: Path) -> bool:
        """Delete a recording and update all related files"""
        try:
            recording_id = recording_path.stem
            deleted_num_format = recording_id.split('_')[0]
            try:
                recording_num = int(deleted_num_format)
            except ValueError:
                raise ValueError(f"Invalid recording number format in {recording_path.name}")
            
            # Verify metadata exists and is valid
            if recording_id not in self.metadata:
                raise ValueError(f"No metadata found for {recording_path.name}")
            
            metadata_entry = self.metadata[recording_id]
            required_fields = ['audio_filepath', 'text', 'duration']
            missing_fields = [f for f in required_fields if f not in metadata_entry]
            if missing_fields:
                raise ValueError(f"Missing required metadata fields: {missing_fields}")
            
            try:
                duration = float(metadata_entry['duration'])
                if duration <= 0:
                    raise ValueError("Duration must be positive")
            except (ValueError, TypeError):
                raise ValueError("Invalid duration in metadata")
            
            text = metadata_entry['text']
            if not text or not isinstance(text, str):
                raise ValueError("Invalid or missing text in metadata")
            
            # Delete the audio file - use try/except instead of exists check
            try:
                recording_path.unlink()
            except FileNotFoundError:
                raise FileNotFoundError(f"Recording file {recording_path.name} not found")
            
            # Handle all sentence list updates atomically
            original_sentences = self.sentences.copy()
            original_used = self.used_sentences.copy()
            original_trashed = self.trashed_sentences.copy()
            
            try:
                # Update all sentence lists
                if text in self.used_sentences:
                    self.used_sentences.remove(text)
                    if text not in self.trashed_sentences:
                        self.trashed_sentences.append(text)
                    if text in self.sentences:
                        self.sentences.remove(text)
                else:
                    print(f"Warning: Text '{text}' not found in used_sentences")
                
                # Save all sentence lists together
                self._save_sentences()
                self._save_used_sentences()
                self._save_trashed_sentences()
            except Exception as e:
                # Rollback sentence lists if saving fails
                self.sentences = original_sentences
                self.used_sentences = original_used
                self.trashed_sentences = original_trashed
                raise e
            
            # Create an ordered list of metadata entries and find the deleted position
            metadata_items = list(self.metadata.items())
            try:
                deleted_index = next(i for i, (k, _) in enumerate(metadata_items) if k == recording_id)
            except StopIteration:
                raise ValueError(f"Failed to find position of recording {recording_id} in metadata")
            
            # Store original metadata state for rollback
            original_metadata = self.metadata.copy()
            
            try:
                # Remove the old entry from metadata
                del self.metadata[recording_id]
                
                # Build numeric map of potential replacements
                new_entries = {}
                for k, v in self.metadata.items():
                    try:
                        prefix = k.split('_')[0]
                        num = int(prefix)
                        new_entries[num] = (k, v)
                    except ValueError:
                        continue
                
                if new_entries:
                    # 1) Select highest-numbered entry
                    max_num = max(new_entries.keys())
                    new_key, new_entry = new_entries[max_num]
                    
                    # 2) Remove it from current position in metadata
                    del self.metadata[new_key]
                    
                    # 3) Physically rename the .wav file to the old numeric prefix
                    old_audio_file = Path(new_entry["audio_filepath"])
                    if old_audio_file.exists():
                        suffix = old_audio_file.suffix  # e.g. ".wav"
                        # e.g. new_key = "00733_some_text", subprefix = "some_text"
                        subprefix = old_audio_file.stem.split('_', 1)[1]
                        new_audio_filename = f"{deleted_num_format}_{subprefix}{suffix}"
                        new_audio_file = self.recordings_dir / new_audio_filename
                        # Actually rename on disk
                        old_audio_file.rename(new_audio_file)
                        # Update new_entry to reflect the rename
                        new_entry["audio_filepath"] = str(Path("data/recordings") / new_audio_filename)
                    else:
                        print(f"Warning: Could not find file {old_audio_file} to rename.")
                    
                    # Create a new OrderedDict to place the replaced entry in the old spot
                    new_metadata = OrderedDict()
                    for i, (k, v) in enumerate(metadata_items):
                        if i == deleted_index:
                            # Rebuild the key to match "deleted_num_format" but keep the suffix text
                            replacement_key = f"{deleted_num_format}_{new_key.split('_', 1)[1]}"
                            new_metadata[replacement_key] = new_entry
                        if k != recording_id and k != new_key and k not in new_metadata:
                            new_metadata[k] = v
                    
                    self.metadata = new_metadata
                    
                # Save the updated metadata
                self._save_metadata_safe()
                
                # (Renumbering logic continues if new_entries is empty...)
                if not new_entries:
                    try:
                        self._renumber_recordings(recording_num)
                    except Exception as e:
                        # Restore original metadata state
                        self.metadata = original_metadata
                        self._save_metadata_safe()
                        print(f"Warning: Failed to renumber recordings: {e}")
                        print("The system may be in an inconsistent state. Please check the files manually.")
                        return False
                
                return True
                
            except Exception as e:
                # Restore original metadata state
                self.metadata = original_metadata
                self._save_metadata_safe()
                raise e
            
        except Exception as e:
            print(f"Error deleting recording: {e}")
            return False

    def _renumber_recordings(self, deleted_num: int):
        """Fill the gap by moving the last recording"""
        recordings = sorted(list(self.recordings_dir.glob("*.wav")))
        if not recordings:
            return
            
        last_recording = recordings[-1]
        last_num_str = last_recording.stem.split('_')[0]  # e.g., "002"
        
        # Prepare move operation
        old_text = '_'.join(last_recording.stem.split('_')[1:])
        new_name = last_recording.parent / f"{deleted_num:03d}_{old_text}.wav"
        new_stem = f"{deleted_num:03d}_{old_text}"
        
        # Keep original states for rollback
        original_metadata = self.metadata.copy()
        original_path = last_recording
        
        moved = False
        try:
            # Create ordered list and find target position
            metadata_items = list(self.metadata.items())
            target_index = deleted_num - 1  # Since numbers are 1-based
            
            # Update metadata first
            metadata_entry = self.metadata[last_recording.stem].copy()
            metadata_entry['audio_filepath'] = f"data/recordings/{new_name.name}"
            
            # Create new metadata with the entry in the correct position
            new_metadata = OrderedDict()
            for i, (k, v) in enumerate(metadata_items):
                if i == target_index:
                    new_metadata[new_stem] = metadata_entry
                if k != last_recording.stem:  # Skip the old key as we handle it separately
                    new_metadata[k] = v
            
            # Move the file
            last_recording.rename(new_name)
            moved = True
            
            # Update metadata
            self.metadata = new_metadata
            self._save_metadata_safe()
            
        except Exception as e:
            # Rollback file move if needed
            if moved:
                try:
                    new_name.rename(original_path)
                except Exception as move_error:
                    raise RuntimeError(f"Failed to rollback file move: {move_error}")
            
            # Restore metadata
            self.metadata = original_metadata
            self._save_metadata_safe()
            raise e

    def _save_metadata_safe(self):
        """Save metadata back to CSV file safely using atomic write"""
        import tempfile
        import shutil
        import os
        import fcntl
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(prefix='metadata_', suffix='.csv')
        try:
            # Close the file descriptor properly
            os.close(temp_fd)
            
            # Write to the temporary file
            with open(temp_path, 'w', newline='') as f:
                # Lock the file while writing
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                writer = csv.DictWriter(f, fieldnames=['audio_filepath', 'text', 'duration'])
                writer.writeheader()
                for recording_id, entry in self.metadata.items():
                    writer.writerow(entry)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Replace original file with new one
            shutil.move(temp_path, self.metadata_file)
        except Exception as e:
            # Clean up temp file in case of error
            try:
                Path(temp_path).unlink()
            except:
                pass
            raise e

    def _save_json_safe(self, data: List[str], filepath: Path):
        """Save JSON data safely using atomic write"""
        import tempfile
        import shutil
        import os
        import fcntl
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(prefix=filepath.stem + '_', suffix='.json')
        try:
            # Close the file descriptor properly
            os.close(temp_fd)
            
            # Write to the temporary file
            with open(temp_path, 'w') as f:
                # Lock the file while writing
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(data, f, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Replace original file with new one
            shutil.move(temp_path, filepath)
        except Exception as e:
            # Clean up temp file in case of error
            try:
                Path(temp_path).unlink()
            except:
                pass
            raise e

    def _save_sentences(self):
        """Save available sentences safely"""
        self._save_json_safe(self.sentences, self.sentences_file)

    def _save_used_sentences(self):
        """Save used sentences safely"""
        self._save_json_safe(self.used_sentences, self.used_sentences_file)

    def _save_trashed_sentences(self):
        """Save trashed sentences safely"""
        self._save_json_safe(self.trashed_sentences, self.trashed_sentences_file)

    def export_report(self):
        """Export cleanup report"""
        report = {
            'total_recordings': len(list(self.recordings_dir.glob("*.wav"))),
            'total_duration': sum(float(entry['duration']) for entry in self.metadata.values()),
            'available_sentences': len(self.sentences),
            'used_sentences': len(self.used_sentences),
            'trashed_sentences': len(self.trashed_sentences),
            'metadata_entries': len(self.metadata),
            'orphaned_recordings': [
                str(f.name) for f in self.recordings_dir.glob("*.wav")
                if f.stem not in self.metadata
            ],
            'missing_recordings': [
                entry['audio_filepath'] for entry in self.metadata.values()
                if not Path(entry['audio_filepath']).exists()
            ]
        }
        
        report_file = Path('cleanup_report.json')
        self._save_json_safe(report, report_file)
        print(f"\nExported cleanup report to {report_file}")

    def show_statistics(self):
        """Display current dataset statistics"""
        try:
            total_recordings = len(list(self.recordings_dir.glob("*.wav")))
            total_duration = 0
            invalid_durations = []
            
            for recording_id, entry in self.metadata.items():
                try:
                    duration = float(entry['duration'])
                    if duration <= 0:
                        invalid_durations.append(recording_id)
                    else:
                        total_duration += duration
                except (ValueError, TypeError):
                    invalid_durations.append(recording_id)
            
            if invalid_durations:
                print("\nWarning: Invalid durations found in metadata for recordings:", invalid_durations)
            
            print("\nDataset Statistics:")
            print(f"Total Recordings: {total_recordings}")
            print(f"Total Duration: {total_duration/60:.1f} minutes")
            print(f"Available Sentences: {len(self.sentences)}")
            print(f"Used Sentences: {len(self.used_sentences)}")
            print(f"Trashed Sentences: {len(self.trashed_sentences)}")
            
            # Verify data consistency
            orphaned = [f for f in self.recordings_dir.glob("*.wav") if f.stem not in self.metadata]
            missing = [k for k in self.metadata.keys() if not (self.recordings_dir / f"{k}.wav").exists()]
            
            if orphaned:
                print("\nWarning: Found recordings without metadata:", [f.name for f in orphaned])
            if missing:
                print("\nWarning: Found metadata entries without recordings:", missing)
                
        except Exception as e:
            print(f"Error calculating statistics: {e}")

    def _play_audio(self, audio_path: Path):
        """Play audio file with proper resource cleanup"""
        stream = None
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file {audio_path.name} not found")
            
            data, samplerate = sf.read(str(audio_path))
            stream = sd.OutputStream(samplerate=samplerate, channels=len(data.shape))
            stream.start()
            
            print("\nPlaying... (Press Enter to skip)")
            sd.play(data, samplerate)
            
            # Keep checking for input while audio is playing
            while sd.get_stream().active:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    sys.stdin.readline()
                    sd.stop()
                    break
                
        except (sd.PortAudioError, sf.SoundFileError) as e:
            raise e
        finally:
            if stream is not None:
                stream.stop()
                stream.close()
            sd.stop()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)

    @contextlib.contextmanager
    def _file_lock(self, file_obj, exclusive: bool = False):
        """Context manager for file locking"""
        try:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

    def _load_metadata(self) -> OrderedDict:
        """Load metadata from CSV file with proper ordering"""
        metadata = OrderedDict()
        if not self.metadata_file.exists():
            return metadata
            
        try:
            with open(self.metadata_file, 'r', newline='') as f:
                with self._file_lock(f):
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Use the stem (filename without extension) as the key
                        filepath = Path(row['audio_filepath'])
                        metadata[filepath.stem] = row.copy()  # Make a copy to avoid modifying the original
            return metadata
        except (IOError, csv.Error) as e:
            print(f"Error loading metadata: {e}")
            return OrderedDict()

    def _load_sentences(self) -> List[str]:
        """Load available sentences"""
        if self.sentences_file.exists():
            with open(self.sentences_file) as f:
                return json.load(f)
        return []

    def _load_used_sentences(self) -> List[str]:
        """Load used sentences"""
        if self.used_sentences_file.exists():
            with open(self.used_sentences_file) as f:
                return json.load(f)
        return []

    def _load_trashed_sentences(self) -> List[str]:
        """Load trashed sentences"""
        if self.trashed_sentences_file.exists():
            with open(self.trashed_sentences_file) as f:
                return json.load(f)
        return []

    def _validate_recording_number(self, num: int) -> bool:
        """Validate recording number"""
        if not isinstance(num, int):
            return False
        if num < 1:  # Only check that it's positive
            return False
        return True

    def _numeric_sort_key(self, path: Path) -> int:
        """Helper to convert the prefix of the filename to an integer for sorting."""
        # e.g., "00732_the_album.wav" -> prefix="00732" -> integer=732
        prefix = path.stem.split('_', 1)[0]
        try:
            return int(prefix)
        except ValueError:
            # If something doesn't parse, treat it as extremely large so it sorts last
            return 999999999

    def cleanup_orphans_and_duplicates(self):
        """
        Removes orphaned recordings (those not in metadata)
        and prints out any duplicated text entries in metadata.
        """
        print("\n--- Cleaning up orphans & checking for duplicates ---")
        
        # 1) Remove orphaned recordings
        orphans = []
        for wav_file in self.recordings_dir.glob("*.wav"):
            if wav_file.stem not in self.metadata:
                orphans.append(wav_file)
        
        if orphans:
            print(f"\nDeleting orphaned recordings: {[o.name for o in orphans]}")
            for orphan_file in orphans:
                orphan_file.unlink()
        else:
            print("\nNo orphaned recordings found.")

        # 2) Report duplicates in 'text' column
        text_map = {}
        for rec_id, row in self.metadata.items():
            text_val = row.get("text", "")
            if text_val not in text_map:
                text_map[text_val] = []
            text_map[text_val].append(rec_id)

        duplicates = {txt: recs for txt, recs in text_map.items() if len(recs) > 1}
        if duplicates:
            print("\nFound duplicates in 'text' column:")
            for txt, rec_ids in duplicates.items():
                print(f"  Text: {txt} => Recordings: {rec_ids}")
        else:
            print("\nNo duplicates found in 'text' column.")

    def validate_metadata_and_json(self):
        """
        Checks that:
          (1) All metadata entries have an incrementing numeric prefix with no duplicates.
          (2) The audio files exist for each metadata entry.
          (3) Each text in metadata is present in used_sentences.json.
        """
        print("\n--- Validating metadata & JSON ---")

        # 1) Check numeric prefixes for duplicates & missing indices
        prefixes = []
        for recording_id in self.metadata:
            # Example: recording_id is something like "00712_the_album..."
            # Here we parse the numeric portion before the underscore.
            prefix_str = recording_id.split("_", 1)[0]
            try:
                prefix = int(prefix_str)
                prefixes.append(prefix)
            except ValueError:
                print(f"  Warning: Could not parse numeric prefix from '{recording_id}'")

        # Sort and see if each prefix is unique
        prefixes_sorted = sorted(prefixes)
        duplicates = []
        missing = []

        # Identify duplicates
        seen = set()
        for p in prefixes_sorted:
            if p in seen:
                duplicates.append(p)
            seen.add(p)

        # Identify missing indices if we want them strictly consecutive from 1..max
        if prefixes_sorted:
            expected = set(range(1, prefixes_sorted[-1] + 1))
            actual = set(prefixes_sorted)
            missing = sorted(list(expected - actual))

        # 2) Check that files exist and text is in used_sentences
        missing_files = []
        missing_in_used_sentences = []
        for rec_id, row in self.metadata.items():
            audio_path = Path(row["audio_filepath"])
            if not audio_path.exists():
                missing_files.append(rec_id)
            text_val = row["text"]
            if text_val not in self.used_sentences:
                missing_in_used_sentences.append(rec_id)

        # Print final report
        if duplicates:
            print(f"\nDuplicate numeric prefixes found: {duplicates}")
        else:
            print("\nNo duplicate numeric prefixes found.")
        if missing:
            print(f"Missing numeric indices (gaps in sequence): {missing}")
        else:
            print("\nNo missing numeric indices found.")
        if missing_files:
            print(f"\nMetadata references missing audio files: {missing_files}")
        else:
            print("\nNo missing audio files referenced in metadata.")
        if missing_in_used_sentences:
            print(f"\nRecordings whose text is not in used_sentences.json: {missing_in_used_sentences}")
            # Also add them to used_sentences
            for rec_id in missing_in_used_sentences:
                text_val = self.metadata[rec_id]['text']
                if text_val not in self.used_sentences:
                    self.used_sentences.append(text_val)
            self._save_used_sentences()
            print(f"\nAdded {len(missing_in_used_sentences)} text(s) to used_sentences.json.")
        else:
            print("\nAll metadata text entries are present in used_sentences.json.")
        
        print("\n--- Validation complete ---\n")

def standardize_metadata_spelling(metadata_path: str):
    """Standardize British spellings to American in the metadata CSV."""
    # Load the metadata CSV
    df = pd.read_csv(metadata_path)

    # Initialize SentenceFilter for spelling standardization
    sentence_filter = SentenceFilter()

    # Standardize spellings in the 'text' column
    df['text'] = df['text'].apply(sentence_filter.standardize_spelling)

    # Save the updated DataFrame back to CSV
    df.to_csv(metadata_path, index=False)
    print(f"Updated British spellings to American in {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean up recorded dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cleaner = RecordingCleaner(args.config)
    cleaner.run()

if __name__ == "__main__":
    main()