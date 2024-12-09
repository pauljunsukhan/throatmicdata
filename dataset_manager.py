import os
import json
import csv
import librosa
import numpy as np
from pathlib import Path
from datasets import Dataset, Audio, load_dataset
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm
import shutil

class DatasetManager:
    def __init__(self, hf_token=None):
        self.data_dir = "throatmic_data"
        self.metadata_file = "metadata.csv"
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("Please provide a Hugging Face token via environment variable HF_TOKEN or as parameter")
        
        self.api = HfApi(token=self.hf_token)
        
    def prepare_dataset(self):
        """Prepare the dataset for upload to Hugging Face"""
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file {self.metadata_file} not found")
            
        # Read the metadata
        data = []
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_path = row['audio_filepath']
                if os.path.exists(audio_path):
                    # Get audio duration
                    duration = librosa.get_duration(path=audio_path)
                    data.append({
                        'audio': audio_path,
                        'text': row['text'],
                        'duration': duration
                    })
        
        # Create the dataset
        dataset = Dataset.from_list(data)
        
        # Cast the audio column to Audio feature
        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
        
        return dataset
    
    def upload_to_hub(self, repo_name, private=True):
        """Upload the dataset to Hugging Face Hub"""
        print("Preparing dataset...")
        dataset = self.prepare_dataset()
        
        # Create or get the repository
        try:
            repo_url = create_repo(
                repo_name,
                private=private,
                token=self.hf_token,
                repo_type="dataset"
            ).repo_id
        except Exception as e:
            if "already exists" not in str(e):
                raise e
            repo_url = f"{self.api.whoami()['name']}/{repo_name}"
        
        print(f"Pushing dataset to {repo_url}...")
        dataset.push_to_hub(
            repo_url,
            private=private,
            token=self.hf_token
        )
        
        print(f"Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_url}")
        return repo_url
    
    def setup_dvc(self):
        """Initialize DVC for the audio data"""
        import subprocess
        
        # Initialize DVC if not already initialized
        if not os.path.exists('.dvc'):
            subprocess.run(['dvc', 'init'], check=True)
            
        # Track the audio directory with DVC
        subprocess.run(['dvc', 'add', self.data_dir], check=True)
        
        # Create .gitignore if it doesn't exist
        gitignore_path = Path('.gitignore')
        if not gitignore_path.exists():
            gitignore_path.write_text(f"{self.data_dir}/\n.dvc/\n")
        else:
            current_content = gitignore_path.read_text()
            if self.data_dir not in current_content:
                with gitignore_path.open('a') as f:
                    f.write(f"\n{self.data_dir}/\n.dvc/\n")
        
        print("DVC setup complete. Now you can use:")
        print("1. 'dvc remote add' to add a remote storage")
        print("2. 'dvc push' to push your audio files")
        
    def validate_dataset(self):
        """Validate the dataset before upload"""
        if not os.path.exists(self.metadata_file):
            print("❌ Metadata file not found")
            return False
            
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        all_valid = True
        for row in tqdm(rows, desc="Validating files"):
            audio_path = row['audio_filepath']
            if not os.path.exists(audio_path):
                print(f"❌ Audio file not found: {audio_path}")
                all_valid = False
                continue
                
            try:
                duration = librosa.get_duration(path=audio_path)
                if duration < 1 or duration > 30:
                    print(f"⚠️ Unusual duration ({duration}s) for: {audio_path}")
            except Exception as e:
                print(f"❌ Error reading audio file {audio_path}: {e}")
                all_valid = False
                
        return all_valid

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manage throat mic dataset")
    parser.add_argument("--token", help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--repo", help="Repository name for the dataset")
    parser.add_argument("--private", action="store_true", help="Make the dataset private")
    parser.add_argument("--validate", action="store_true", help="Only validate the dataset")
    parser.add_argument("--setup-dvc", action="store_true", help="Setup DVC for the dataset")
    
    args = parser.parse_args()
    
    try:
        manager = DatasetManager(hf_token=args.token)
        
        if args.validate:
            print("Validating dataset...")
            is_valid = manager.validate_dataset()
            print("✅ Dataset is valid" if is_valid else "❌ Dataset has issues")
            return
            
        if args.setup_dvc:
            print("Setting up DVC...")
            manager.setup_dvc()
            return
            
        if args.repo:
            print("Validating dataset before upload...")
            if manager.validate_dataset():
                print("Uploading to Hugging Face Hub...")
                manager.upload_to_hub(args.repo, private=args.private)
            else:
                print("❌ Please fix dataset issues before uploading")
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 