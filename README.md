# Throat Mic Recording Tool

A Python tool for creating a high-quality dataset for fine-tuning Whisper using throat microphone recordings. This tool helps you record and manage a dataset of throat microphone audio paired with transcriptions, using sentences from Mozilla's Common Voice dataset.

## Features

- 🎤 Easy-to-use recording interface
- 📝 Automatic prompt management from Common Voice
- ✨ Proper audio format for Whisper (16kHz, mono)
- 🔄 Progress saving and session management
- 🤗 Direct upload to Hugging Face Datasets
- ✅ Dataset validation and quality checks

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/throatmicdata.git
cd throatmicdata
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the Common Voice sentences:
```bash
python prepare_sentences.py
```

## Recording Process

1. Start the recording tool:
```bash
python throatmicdata.py
```

2. The tool will:
   - Create a `throatmic_data` directory for recordings
   - Create a `metadata.csv` file for transcripts
   - Let you select your throat microphone

3. During recording:
   - Each prompt is displayed clearly on screen
   - Press Enter to start a 10-second recording
   - Press 'q' to return to the main menu
   - Progress is automatically saved

4. Progress tracking shows:
   - Number of completed recordings
   - Total recording time
   - Percentage complete
   - Next prompt to record

## Dataset Management

### Uploading to Hugging Face

1. Get your token from https://huggingface.co/settings/tokens

2. Set your token:
```bash
export HF_TOKEN=your_token_here
```

3. Validate your dataset:
```bash
python dataset_manager.py --validate
```

4. Upload to Hugging Face:
```bash
python dataset_manager.py --repo your-dataset-name --private
```

## Using the Dataset

The dataset follows the Hugging Face audio dataset format, perfect for Whisper fine-tuning:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/your-dataset-name", use_auth_token=True)

# Access the data
for item in dataset['train']:
    audio = item['audio']  # Contains 'array' (audio signal) and 'sampling_rate'
    text = item['text']    # The transcript
    duration = item['duration']  # Audio length in seconds
```

## Dataset Format

Each item in the dataset has this structure:
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

## Technical Details

- Audio Format:
  - Sample Rate: 16kHz
  - Channels: Mono
  - Bit Depth: 16-bit PCM
  - Format: WAV

- File Organization:
  - `throatmic_data/`: Directory for WAV recordings
  - `metadata.csv`: Maps audio files to transcripts
  - `prompts.txt`: Recording prompts from Common Voice
  - `recording_progress.json`: Saves session progress

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 