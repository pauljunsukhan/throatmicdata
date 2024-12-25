#!/usr/bin/env python3
"""
Download and inspect the throat microphone dataset.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import io
import csv
import logging
import argparse
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import json

# Add src directory to path
src_dir = str(Path(__file__).parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.config import Config
from src.exceptions import DownloadError, DataError

# Initialize configuration
config = Config.load()
config.ensure_directories()

def setup_logging():
    """Configure logging based on config settings."""
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("data/throatmic.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_directories() -> tuple[Path, Path]:
    """Create necessary directories for dataset storage.
    
    Returns:
        tuple[Path, Path]: Tuple of (recordings_dir, metadata_dir)
    """
    data_dir = Path("data")
    recordings_dir = data_dir / "recordings"
    metadata_dir = data_dir / "metadata"
    
    for dir in [recordings_dir, metadata_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    
    return recordings_dir, metadata_dir

def get_existing_files(metadata_file: Path) -> tuple[set, int]:
    """Get set of existing file paths and texts from metadata, and the next index to use.
    
    Args:
        metadata_file: Path to metadata CSV file
        
    Returns:
        tuple[set, int]: Set of existing texts and next available index
        
    Raises:
        DataError: If metadata file cannot be read
    """
    if not metadata_file.exists():
        return set(), 1
    
    try:
        df = pd.read_csv(metadata_file)
        existing = {row['text'] for _, row in df.iterrows()}
        
        # Get the highest index from existing files
        indices = []
        for filepath in df['audio_filepath']:
            try:
                # Extract index from filename (e.g., "001_some_text.wav")
                index = int(Path(filepath).name.split('_')[0])
                indices.append(index)
            except (ValueError, IndexError):
                continue
        
        next_index = max(indices, default=0) + 1
        return existing, next_index
        
    except Exception as e:
        raise DataError(f"Error reading metadata file: {str(e)}")

def download_dataset(repo_id: str, token: str, metadata_file: Path) -> pd.DataFrame:
    """Download the dataset from Hugging Face and save to local format.
    
    Args:
        repo_id: Hugging Face repository ID
        token: Hugging Face API token
        metadata_file: Path to metadata CSV file
        
    Returns:
        pd.DataFrame: Downloaded dataset
        
    Raises:
        DownloadError: If dataset download fails
        DataError: If data processing fails
    """
    try:
        logger.info("Downloading dataset from Hugging Face...")
        try:
            parquet_path = hf_hub_download(
                repo_id=repo_id,
                filename="data/train-00000-of-00001.parquet",
                repo_type="dataset",
                token=token
            )
        except Exception as e:
            raise DownloadError(f"Failed to download dataset: {str(e)}")
            
        logger.info(f"Downloaded to: {parquet_path}")
        
        # Load dataset
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            raise DataError(f"Failed to read parquet file: {str(e)}")
            
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Create directories
        recordings_dir, metadata_dir = ensure_directories()
        
        # Check existing files
        existing_texts, next_index = get_existing_files(metadata_file)
        logger.info(f"Found {len(existing_texts)} existing recordings")
        
        # Create or append to metadata file
        file_exists = metadata_file.exists()
        mode = 'a' if file_exists else 'w'
        
        with open(metadata_file, mode, newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            if not file_exists:
                writer.writerow(["audio_filepath", "text", "duration"])
        
        # Track new files for logging
        new_files = 0
        skipped_files = 0
        
        logger.info("Processing recordings...")
        current_index = next_index
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Check if text already exists
                if row['text'] in existing_texts:
                    skipped_files += 1
                    continue
                
                # Create filename from text
                safe_text = "".join(c.lower() for c in row['text'][:30] if c.isalnum() or c == ' ')
                filename = f"{current_index:05d}_{safe_text.replace(' ', '_')}.wav"
                rel_path = str(Path("data/recordings") / filename)
                
                # Get audio data
                audio_data = row['audio']
                audio_bytes = audio_data['bytes']
                filepath = recordings_dir / filename
                
                # Read audio data from bytes
                with io.BytesIO(audio_bytes) as buf:
                    audio_array, sample_rate = sf.read(buf)
                    
                # Save as WAV
                sf.write(str(filepath), audio_array, sample_rate)
                
                # Add to metadata using csv writer
                with open(metadata_file, 'a', newline='') as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([rel_path, row['text'], row['duration']])
                
                existing_texts.add(row['text'])
                current_index += 1
                new_files += 1
                
            except Exception as e:
                logger.error(f"Error processing recording: {str(e)}")
                continue
        
        logger.info(f"Download complete! Added {new_files} new recordings, skipped {skipped_files} existing recordings")
        
        # Load existing used sentences
        used_sentences_path = Path('data/repository/used_sentences.json')
        if used_sentences_path.exists():
            with open(used_sentences_path, 'r') as f:
                used_sentences = json.load(f)
        else:
            used_sentences = []

        # Update used sentences
        used_sentences.extend(existing_texts)
        with open(used_sentences_path, 'w') as f:
            json.dump(used_sentences, f, indent=4)
        
        return df
        
    except (DownloadError, DataError) as e:
        logger.error(f"Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise DownloadError(f"Download failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Download and inspect the throat microphone dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download latest dataset
  python download_dataset.py
  
  # Download from custom repository
  python download_dataset.py --repo-id "your-username/your-dataset"
  
  # Save to custom metadata file
  python download_dataset.py --metadata "path/to/metadata.csv"
        """
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=config.dataset.repo_id,
        help=f"Hugging Face repository ID (default: {config.dataset.repo_id})"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get('HF_TOKEN'),
        help="Hugging Face API token (defaults to HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=config.dataset.metadata_file,
        help=f"Path to metadata CSV file (default: {config.dataset.metadata_file})"
    )
    parser.add_argument(
        "--help-more",
        action="store_true",
        help="Show detailed help about the download process"
    )
    
    args = parser.parse_args()
    
    if args.help_more:
        print("""
Detailed Download Process:
------------------------
1. Downloads the latest dataset from Hugging Face
2. Converts to local format (WAV files + metadata CSV)
3. Maintains sequential file numbering
4. Skips existing recordings automatically

File Organization:
----------------
- Recordings saved to: data/recordings/
- Metadata saved to: data/metadata/metadata.csv
- Files named as: 001_first_few_words.wav

Environment Variables:
--------------------
HF_TOKEN: Your Hugging Face API token
  - Get it from: https://huggingface.co/settings/tokens
  - Or use --token to provide directly
        """)
        return 0
    
    # Validate token
    if not args.token:
        logger.error("No Hugging Face token provided. Set HF_TOKEN environment variable or use --token")
        return 1
    
    try:
        df = download_dataset(
            repo_id=args.repo_id,
            token=args.token,
            metadata_file=Path(args.metadata)
        )
        logger.info("Download and conversion complete!")
        return 0
    except (DownloadError, DataError) as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    logger = setup_logging()
    exit(main()) 