# Throat Mic Recording Tool

This tool helps you create a dataset for fine-tuning Whisper using a throat microphone. It uses sentences from Mozilla's Common Voice dataset and guides you through the recording process.

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Download the Common Voice sentences:
```bash
python prepare_sentences.py
```

3. Start the recording tool:
```bash
python throatmicdata.py
```

## Usage

1. When you first run the tool, it will:
   - Create a `throatmic_data` directory for storing recordings
   - Create a `metadata.csv` file to track recordings and transcripts
   - Allow you to select your throat microphone from available audio devices

2. The recording process:
   - Each prompt will be displayed on screen
   - Press Enter to start recording (10 seconds per prompt)
   - Press 'q' at any time to return to the main menu
   - Progress is automatically saved between sessions

3. The tool will track your progress and show:
   - Number of completed recordings
   - Total recording time
   - Percentage complete
   - Next prompt to be recorded

## Dataset Management

### Using DVC (Data Version Control)

1. Initialize DVC for your dataset:
```bash
python dataset_manager.py --setup-dvc
```

2. Add a remote storage (e.g., S3, Google Drive):
```bash
dvc remote add -d myremote s3://mybucket/path
dvc remote modify myremote endpointurl https://...  # Optional
```

3. Push your dataset:
```bash
dvc push
```

### Uploading to Hugging Face

1. Get your Hugging Face token from https://huggingface.co/settings/tokens

2. Set your token (choose one method):
   ```bash
   # Method 1: Environment variable
   export HF_TOKEN=your_token_here
   
   # Method 2: Pass as parameter
   python dataset_manager.py --token your_token_here ...
   ```

3. Validate your dataset:
```bash
python dataset_manager.py --validate
```

4. Upload to Hugging Face:
```bash
python dataset_manager.py --repo your-dataset-name --private
```

## Output

- WAV files are saved in the `throatmic_data` directory
- All files are recorded in the format required by Whisper:
  - 16kHz sample rate
  - 16-bit PCM
  - Mono channel
- `metadata.csv` contains the mapping between audio files and transcripts
- Progress is saved in `recording_progress.json`

## Dataset Format

The dataset on Hugging Face will have the following structure:
```python
{
    'audio': {
        'path': 'path/to/audio.wav',
        'array': np.array(...),  # The audio signal
        'sampling_rate': 16000
    },
    'text': 'The transcription of the audio',
    'duration': 10.0  # Duration in seconds
}
```

This format is compatible with Whisper fine-tuning and can be loaded using:
```python
from datasets import load_dataset
dataset = load_dataset("your-username/your-dataset-name")
``` 