# Throat Mic Recording Tool & Dataset

A high-quality dataset and recording tool for fine-tuning Whisper using throat microphone recordings. This dataset is specifically designed for training speech recognition models on throat microphone input.

## Dataset Characteristics

- **Audio Format**: 16kHz mono WAV files
- **Duration**: Each recording is approximately 10 seconds
- **Recording Device**: Throat microphone (laryngophone)
- **Language**: English
- **Total Recordings**: 499 utterances
- **Total Duration**: ~83 minutes
- **Audio Quality**: High-quality recordings with consistent volume levels and minimal background noise

### Sentence Characteristics

The dataset uses carefully selected sentences that are:
- Complex enough for meaningful speech recognition training (12-25 words)
- Include proper grammar and punctuation
- Contain a mix of statement types (declarations, questions, etc.)
- Include natural language patterns and varied vocabulary
- Balanced in terms of phonetic content

### Dataset Format

The dataset follows the standard format required for Whisper fine-tuning:
```
data/
├── recordings/           # WAV audio files (16kHz mono)
│   └── *.wav            # Format: {index}_{text_preview}.wav
└── metadata/
    └── metadata.csv     # Format: audio_filepath,text,duration
```

## Features

### Recording Tool
- 🎤 Easy-to-use recording interface
- 📝 Automatic prompt management
- ✨ Proper audio format for Whisper (16kHz, mono)
- 🔄 Progress saving and session management
- 📊 Dataset quality analysis
- 🔍 Real-time audio level monitoring
- ⚡ Clipping detection
- 🎵 Playback verification

### Dataset Management
- 🔄 Automatic synchronization with Hugging Face
- 📥 Smart dataset downloading with duplicate detection
- 📤 Efficient dataset uploading with change tracking
- 🧹 Duplicate cleanup functionality
- 📝 Automatic dataset card management
- ✅ Data validation and quality checks

## Setup

1. Use uv for project and venv management:
```bash
source .venv/bin/activate
```

2. Install dependencies using uv (recommended) or pip:
```bash
# Using uv (faster)
uv pip install -r requirements.txt

# Using pip
pip install -r requirements.txt
```

3. Set up Hugging Face token (for dataset management):
```bash
# Add to your shell configuration (e.g., .zshrc, .bashrc)
export HF_TOKEN="your_hugging_face_token"
```

## Usage

### Recording
```bash
# Start recording session
python record_audio.py

# List available audio devices
python record_audio.py --list-devices

# Analyze dataset quality
python record_audio.py --analyze --quality

# Check dataset coverage
python record_audio.py --analyze --coverage
```

### Dataset Management
```bash
# Download dataset (automatically handles duplicates)
python download_dataset.py

# Upload to Hugging Face (only uploads new/changed files)
python upload_dataset.py

# Clean up duplicates in the dataset
python upload_dataset.py --cleanup

# Use custom repository
python download_dataset.py --repo-id custom/repo
python upload_dataset.py --repo-id custom/repo
```

### Advanced Options
```bash
# Specify custom metadata file location
--metadata path/to/metadata.csv

# Use different dataset card
--dataset-card path/to/card.md

# Provide Hugging Face token directly
--token YOUR_TOKEN
```

## Dataset Quality Control

Each recording undergoes several quality checks:
1. Audio level monitoring during recording
2. Clipping detection
3. Playback verification
4. Option to re-record if quality is unsatisfactory
5. Automatic validation during dataset upload

## License

MIT License - see LICENSE file for details. 

## Configuration

The project settings can be customized by editing `config.yaml`:

```yaml
# Audio recording settings
audio:
  sample_rate: 16000    # Recording quality in Hz
  duration: 10.0        # Recording length in seconds
  clipping_threshold: 0.95   # Prevents audio distortion

# Dataset settings
dataset:
  repo_id: "pauljunsukhan/throatmic_codered"   # Your Hugging Face repository
  metadata_file: "data/metadata/metadata.csv"   # Where recordings are tracked
  min_words: 12    # Minimum words per sentence
  max_words: 25    # Maximum words per sentence

# Logging settings
logging:
  level: "INFO"    # Log detail (DEBUG, INFO, WARNING, ERROR)
  file: "data/throatmic.log"
```

Edit these values to match your needs. The configuration file is well-documented with comments explaining each option. 