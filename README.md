# Throat Mic Recording Tool & Dataset

Version: 0.1.0

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
├── metadata/            # Dataset tracking
│   └── metadata.csv     # Format: audio_filepath,text,duration
├── cache/               # Temporary files and download cache
└── repository/          # Sentence pool management
    ├── sentences.json   # Available sentences for recording
    └── used_sentences.json  # Tracked recorded sentences
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

### Sentence Management
- 🔄 Automatic sentence downloading from Common Voice
- 📋 Custom sentence addition support
- ✅ Quality filtering for suitable sentences
- 📊 Progress tracking and statistics
- 🎯 Prevents duplicate recordings
- 📝 Maintains available and used sentence pools

### Dataset Management
- 🔄 Automatic synchronization with Hugging Face
- 📥 Smart dataset downloading with duplicate detection
- 📤 Efficient dataset uploading with change tracking
- 🧹 Duplicate cleanup functionality
- 📝 Automatic dataset card management
- ✅ Data validation and quality checks

## Getting Started

### Initial Setup
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

### Recording Workflow

1. **Prepare Sentences**:
   First, you need sentences in your repository before recording:
   ```bash
   python record_audio.py
   # Select option 4 (Manage sentences), then:
   # - Option 1 to download sentences from Common Voice
   # - Option 2 to add your own custom sentences
   ```

2. **Sync with Existing Dataset** (if continuing previous work):
   ```bash
   # Download existing recordings (if any)
   python download_dataset.py
   ```
   The tool will automatically:
   - Download existing recordings
   - Skip any duplicates
   - Sync the sentence repository
   - Track what's been recorded

3. **Start Recording**:
   ```bash
   python record_audio.py
   # Select option 1 to start recording
   ```
   The tool will:
   - Show sentences from your repository
   - Track recording progress
   - Prevent duplicate recordings
   - Save recordings automatically

4. **Upload to Hugging Face**:
   ```bash
   python upload_dataset.py
   ```
   The upload process:
   - Detects new recordings
   - Skips existing files
   - Handles duplicates automatically
   - Updates the dataset card

### Additional Commands
```bash
# List available audio devices
python record_audio.py --list-devices

# Analyze dataset quality
python record_audio.py --analyze --quality

# Check dataset coverage
python record_audio.py --analyze --coverage

# Clean up duplicates in dataset
python upload_dataset.py --cleanup
```

### Advanced Options

**Custom Repository**:
```bash
# Use different Hugging Face repository
python download_dataset.py --repo-id custom/repo
python upload_dataset.py --repo-id custom/repo
```

**File Locations**:
```bash
# Custom metadata file
python download_dataset.py --metadata path/to/metadata.csv

# Custom dataset card
python upload_dataset.py --dataset-card path/to/card.md
```

**Authentication**:
```bash
# Provide token directly instead of environment variable
python download_dataset.py --token YOUR_TOKEN
python upload_dataset.py --token YOUR_TOKEN
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

The project settings can be customized by editing `config.yaml`. Key settings include:

```yaml
# Audio recording settings
audio:
  sample_rate: 16000      # Recording quality (Hz)
  channels: 1             # Mono recording
  duration: 10.0          # Recording length (seconds)
  clipping_threshold: 0.95 # Prevents audio distortion
  min_level_threshold: 0.1 # Minimum volume level

# Dataset settings
dataset:
  repo_id: "pauljunsukhan/throatmic_codered"
  metadata_file: "data/metadata/metadata.csv"
  min_words: 12          # Minimum words per sentence
  max_words: 25          # Maximum words per sentence
  min_duration: 8.0      # Minimum recording length
  max_duration: 12.0     # Maximum recording length

# Logging settings
logging:
  level: "INFO"          # Log detail (DEBUG, INFO, WARNING, ERROR)
  file: "data/throatmic.log"
```

See `config.yaml` for the complete list of configurable options with detailed comments. 

## How It Works

### Sentence Management
The tool uses a sophisticated sentence management system to ensure high-quality recordings:

1. **Sentence Sources**:
   - Downloads sentences from Common Voice (Wikipedia and community-contributed)
   - Supports adding custom sentences
   - Filters sentences based on length and complexity

2. **Quality Control**:
   - Ensures sentences are suitable length (12-25 words)
   - Validates grammar and punctuation
   - Maintains consistent complexity level
   - Filters out unsuitable content

3. **Recording Flow**:
   - Sentences are stored in `data/repository/sentences.json`
   - Each recorded sentence is moved to `data/repository/used_sentences.json`
   - Prevents accidental duplicate recordings
   - Tracks progress and completion statistics

4. **Management Features**:
   - Download more sentences when needed
   - Add custom sentences
   - View recording progress
   - Track total duration and completion

## Sentence Selection Criteria

The tool enforces strict criteria for sentence selection to ensure high-quality training data:

### Basic Requirements
1. **Length**: 12-25 words (targeting ~10 seconds of speech)
2. **Format**: Must start with capital letter and end with proper punctuation (., !, ?)
3. **Characters**: No numbers or special characters (except basic punctuation)
4. **Capitalization**: No all-caps words except common acronyms (US, UK, EU, UN, NASA, FBI, CIA, WHO)

### Complexity Requirements
Sentences must have at least two of the following complexity markers:
1. **Proper Comma Usage**: Contains appropriate comma placement
2. **Subordinating Conjunctions**: Uses words like:
   - because, although, though, unless
   - while, whereas, if, since
   - before, after, as, when
   - where, whether, which, who

3. **Coordinating Conjunctions**: Uses words like:
   - and, but, or, nor
   - for, yet, so

4. **Complex Phrases**: Contains relative pronouns or markers:
   - that, which, who, whom
   - whose, where, when

### Natural Language Rules
1. Maximum 4 commas per sentence (prevents overly complex structures)
2. No run-on sentences or excessive conjunctions
3. Must maintain natural speech rhythm
4. Balanced clause structure

These criteria ensure sentences are:
- Complex enough for meaningful training
- Natural and speakable within 10 seconds
- Suitable for speech recognition tasks
- Consistent in quality and structure