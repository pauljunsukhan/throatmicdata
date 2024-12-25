#!/usr/bin/env python3
"""
Upload updated throat microphone dataset to Hugging Face.
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
from huggingface_hub import HfApi
from datasets import Dataset, Audio, load_dataset, concatenate_datasets, Features, Value

# Add src directory to path
src_dir = str(Path(__file__).parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.config import Config
from src.exceptions import UploadError, DataError

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

def load_metadata(metadata_file: Path) -> pd.DataFrame:
    """Load and validate the metadata CSV file.
    
    Args:
        metadata_file: Path to metadata CSV file
        
    Returns:
        pd.DataFrame: Loaded metadata
        
    Raises:
        DataError: If metadata file cannot be read or is invalid
    """
    logger.info(f"Loading metadata from {metadata_file}")
    try:
        df = pd.read_csv(metadata_file)
        
        # Validate required columns
        required_columns = ['audio_filepath', 'text', 'duration']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataError(f"Missing required columns in metadata: {missing_columns}")
        
        return df
    except Exception as e:
        raise DataError(f"Failed to load metadata: {str(e)}")

def prepare_dataset(metadata_df: pd.DataFrame, base_dir: Path, existing_paths: list[str] = None, replace: bool = False) -> Dataset:
    """Prepare the dataset in the format expected by Hugging Face.
    
    Args:
        metadata_df: DataFrame containing metadata
        base_dir: Base directory for audio files
        existing_paths: List of existing audio file paths to skip
        replace: If True, prepare all files without checking existing
        
    Returns:
        Dataset: Prepared dataset for upload
        
    Raises:
        DataError: If dataset preparation fails
    """
    logger.info("Preparing dataset for upload")
    
    try:
        # Skip files that are already in the dataset
        if existing_paths and not replace:
            # Normalize paths for comparison
            metadata_df['normalized_path'] = metadata_df['audio_filepath'].apply(
                lambda x: str(Path(x).name)
            )
            existing_paths = {str(Path(p).name) for p in existing_paths}
            
            # Filter out existing files
            new_files = metadata_df[~metadata_df['normalized_path'].isin(existing_paths)]
            if len(new_files) == 0:
                logger.info("No new recordings to upload")
                return None
            logger.info(f"Found {len(new_files)} new recordings to upload")
            metadata_df = new_files
        elif replace:
            logger.info(f"Preparing all {len(metadata_df)} recordings for upload")
        
        def generator():
            for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
                try:
                    audio_path = base_dir / row['audio_filepath']
                    yield {
                        'audio': str(audio_path),
                        'text': row['text'],
                        'duration': float(row['duration'])
                    }
                except Exception as e:
                    logger.error(f"Error processing {audio_path}: {str(e)}")
                    continue
        
        features = Features({
            'audio': Audio(sampling_rate=16000),
            'text': Value('string'),
            'duration': Value('float32')
        })
        
        return Dataset.from_generator(generator, features=features)
    except Exception as e:
        raise DataError(f"Failed to prepare dataset: {str(e)}")

def upload_to_huggingface(dataset: Dataset, repo_id: str, token: str, dataset_card: Path = Path("DATASET_CARD.md"), replace: bool = False):
    """Upload the dataset to Hugging Face.
    
    Args:
        dataset: Dataset to upload
        repo_id: Hugging Face repository ID
        token: Hugging Face API token
        dataset_card: Path to dataset card markdown file
        replace: If True, replace the entire dataset instead of concatenating
        
    Raises:
        UploadError: If upload fails
    """
    logger.info(f"Uploading dataset to {repo_id}")
    
    try:
        api = HfApi()
        
        # Upload dataset card first
        logger.info("Uploading dataset card...")
        try:
            api.upload_file(
                path_or_fileobj=str(dataset_card),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
            logger.info("Dataset card uploaded successfully")
        except Exception as e:
            raise UploadError(f"Failed to upload dataset card: {str(e)}")
        
        # Handle dataset upload
        dataset_to_upload = dataset
        try:
            if not replace:
                existing_dataset = load_dataset(repo_id, split='train', token=token)
                logger.info(f"Found existing dataset with {len(existing_dataset)} examples")
                
                # If no new data to upload, return early
                if dataset is None:
                    logger.info("No new data to upload")
                    return
                
                # Combine existing and new datasets
                dataset_to_upload = concatenate_datasets([existing_dataset, dataset])
                logger.info(f"Combined dataset will have {len(dataset_to_upload)} examples")
            else:
                logger.info("Replacing entire dataset with local version")
                if dataset is None:
                    logger.warning("No data to upload - this will clear the remote dataset")
            
        except Exception as e:
            if not replace:
                logger.info(f"Creating new dataset (no existing dataset found: {str(e)})")
        
        # Push to the Hub
        try:
            dataset_to_upload.push_to_hub(
                repo_id,
                token=token,
                private=False,
                split='train'
            )
        except Exception as e:
            raise UploadError(f"Failed to push dataset to hub: {str(e)}")
        
        logger.info("Successfully uploaded dataset!")
        
    except (UploadError, DataError) as e:
        logger.error(f"Error uploading to Hugging Face: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise UploadError(f"Upload failed: {str(e)}")

def deduplicate_dataset(dataset: Dataset) -> Dataset:
    """Remove duplicate recordings based on audio path.
    
    Args:
        dataset: Dataset to deduplicate
        
    Returns:
        Dataset: Deduplicated dataset
    """
    logger.info("Checking for duplicates...")
    
    # Get all paths and find duplicates
    paths = [str(Path(example['audio']['path']).name) for example in dataset]
    seen_paths = set()
    unique_indices = []
    
    # Keep only first occurrence of each path
    for i, path in enumerate(paths):
        if path not in seen_paths:
            seen_paths.add(path)
            unique_indices.append(i)
    
    # Create new dataset with only unique recordings
    unique_dataset = dataset.select(unique_indices)
    num_duplicates = len(dataset) - len(unique_dataset)
    
    if num_duplicates > 0:
        logger.info(f"Found and removed {num_duplicates} duplicate recordings")
    else:
        logger.info("No duplicates found")
    
    return unique_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Upload throat microphone dataset to Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload new recordings
  python upload_dataset.py
  
  # Upload with custom repository
  python upload_dataset.py --repo-id "your-username/your-dataset"
  
  # Replace entire dataset with local version
  python upload_dataset.py --replace
  
  # Clean up duplicates in dataset
  python upload_dataset.py --cleanup
  
  # Use custom metadata file
  python upload_dataset.py --metadata "path/to/metadata.csv"
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
        "--dataset-card",
        type=str,
        default=config.dataset.dataset_card,
        help=f"Path to dataset card markdown file (default: {config.dataset.dataset_card})"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up duplicates in the dataset"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace entire dataset with local version instead of concatenating"
    )
    parser.add_argument(
        "--help-more",
        action="store_true",
        help="Show detailed help about the upload process"
    )
    
    args = parser.parse_args()
    
    if args.help_more:
        print("""
Detailed Upload Process:
-----------------------
1. The tool first checks for existing recordings in your Hugging Face dataset
2. It then compares with your local metadata to find new recordings
3. New recordings are prepared and validated
4. The dataset card (README) is uploaded first
5. Finally, new recordings are uploaded and merged with existing data

Quality Checks:
--------------
- Audio files must be 16kHz mono WAV format
- Metadata must contain: audio_filepath, text, duration
- Duplicates are automatically detected and handled
- Failed uploads are logged for retry

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
        if args.cleanup:
            # Load existing dataset and deduplicate
            logger.info("Loading existing dataset for cleanup...")
            try:
                existing_dataset = load_dataset(args.repo_id, split='train', token=args.token)
                logger.info(f"Found dataset with {len(existing_dataset)} examples")
                
                # Deduplicate
                cleaned_dataset = deduplicate_dataset(existing_dataset)
                
                if len(cleaned_dataset) < len(existing_dataset):
                    # Use upload_to_huggingface with replace=True for cleanup
                    upload_to_huggingface(
                        dataset=cleaned_dataset,
                        repo_id=args.repo_id,
                        token=args.token,
                        dataset_card=Path(args.dataset_card),
                        replace=True
                    )
                    logger.info("Successfully cleaned and uploaded dataset!")
                
                return 0
                
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                return 1
        
        # Get existing dataset paths only if not replacing
        existing_paths = None
        if not args.replace:
            existing_dataset = load_dataset(args.repo_id, split='train', token=args.token)
            existing_paths = [example['audio']['path'] for example in existing_dataset]
            logger.info(f"Found {len(existing_paths)} existing recordings")
        
        # Load metadata
        metadata_df = load_metadata(Path(args.metadata))
        
        # Prepare dataset
        dataset = prepare_dataset(metadata_df, Path('.'), existing_paths, replace=args.replace)
        
        # Upload to Hugging Face
        upload_to_huggingface(
            dataset=dataset,
            repo_id=args.repo_id,
            token=args.token,
            dataset_card=Path(args.dataset_card),
            replace=args.replace
        )
        
    except (UploadError, DataError) as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    logger = setup_logging()
    exit(main()) 