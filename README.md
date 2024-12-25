# 🎙 Throat Microphone Recording Tool

🤗 **Hugging Face Dataset**: [pauljunsukhan/throatmic_codered](https://huggingface.co/datasets/pauljunsukhan/throatmic_codered)  
📦 **GitHub Repository**: [pauljunsukhan/throatmicdata](https://github.com/pauljunsukhan/throatmicdata)

🚀 **Fine-tuned Whisper Model**:
- 🤗 **Model**: [pauljunsukhan/throatmic_subvocalization_whisper](https://huggingface.co/pauljunsukhan/throatmic_subvocalization_whisper)
- 📦 **Training Code**: [pauljunsukhan/whisper_finetuning](https://github.com/pauljunsukhan/whisper_finetuning)

🎧 **Hardware Used**:
- **Throat Microphone**: [CodeRed Assault MOD Tactical Throat Mic Headset](https://coderedheadsets.com/assault-mod-tactical-throat-mic-headset/)
  - Uses standard 3.5mm audio jack
  - Tested with MacBook Air's built-in 3.5mm port
  - ⚠️ Warning: Many USB-C to 3.5mm microphone adapters do not work with throat microphones

Create high-quality whisper fine-tuning datasets using a throat microphone! This tool helps you build clean, organized datasets by recording sentences from Common Voice or your own custom prompts.

## ✨ Features

- 🎤 Record audio using a throat microphone
- 🔄 Download sentences from Common Voice
- 📋 Add your own custom sentences
- 🎯 Prevent duplicates and track progress
- 🗑️ Trash difficult sentences with a single key
- 🔢 Clean, sequential file organization
- ⚡ Fast and easy dataset upload to Hugging Face

## 🚀 Quick Start

1. **Install Dependencies**
```bash
uv venv
source .venv/bin/activate    # Linux/Mac
# OR
.venv\Scripts\activate      # Windows

uv pip install -r requirements.txt
uv pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl"
```

2. **Configure Settings**
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your settings
```

3. **Get Some Sentences**
```bash
python record_audio.py
# Choose option 4 > 1 to download from Common Voice
# or option 4 > 2 to add your own
```

4. **Start Recording!**
```bash
python record_audio.py
# Choose option 1 and follow the prompts
```

5. **Upload to Hugging Face**
```bash
python upload_dataset.py
# Your dataset will be uploaded with proper metadata
```

## 🎯 How It Works

### Recording Flow

1. **See a sentence** → Read it in your head first
2. **Press SPACE** → Start recording
3. **Read clearly** → Take your time
4. **Press SPACE** → Stop recording
5. **Choose action:**
   - **ENTER** → Keep and continue
   - **r** → Retry recording
   - **t** → Trash difficult sentence
   - **q** → Take a break

### Smart Features

- 📝 **Sentence Management**
  - Sentences flow through three pools: available → used → (optionally) trashed
  - Hit 't' to move tricky sentences to the trash
  - Trashed sentences won't come back in future downloads

- 🔢 **Clean Organization**
  - Files numbered sequentially (001.wav, 002.wav, etc.)
  - No gaps even when sentences are trashed
  - Perfect for training datasets

- 📊 **Progress Tracking**
  - See your total recording time
  - Track completion percentage
  - Review trashed sentences

### File Structure

```
data/
├── recordings/     # Your audio files (001.wav, 002.wav, etc.)
├── metadata/
│   └── metadata.csv    # Dataset index (path,text,duration)
├── repository/
│   ├── sentences.json         # Available sentences
│   ├── used_sentences.json    # Recorded sentences
│   └── trashed_sentences.json # Difficult sentences
└── throatmic.log      # Activity log
```

## 🛠️ CLI Commands

### Recording Tool

```bash
python record_audio.py [OPTIONS]

Options:
  --analyze         Analyze existing recordings
  --quality         Run audio quality analysis
  --coverage        Analyze dataset coverage
  --list-devices    List available audio devices
  --output FILE     Output file for analysis results
  --debug          Enable debug logging
```

### Dataset Download

```bash
python download_dataset.py [OPTIONS]

Options:
  --repo-id TEXT    Hugging Face repository ID
                    (default: pauljunsukhan/throatmic_codered)
  --token TEXT      Hugging Face API token
                    (can also use HF_TOKEN environment variable)
  --metadata TEXT   Path to metadata CSV file
                    (default: data/metadata/metadata.csv)
  --help           Show this help message
  --help-more      Show detailed help about the download process
```

The download script provides several smart features:

1. **Incremental Downloads**
   - Automatically skips existing recordings
   - Maintains sequential file numbering (001, 002, etc.)
   - Continues from the last used index

2. **File Organization**
   - Saves recordings to `data/recordings/` as WAV files
   - Creates metadata CSV at `data/metadata/metadata.csv`
   - Names files using pattern: `NNNNN_first_few_words.wav` (5 digits) or `NNN_first_few_words.wav` (3 digits)
   - Supports both legacy 3-digit (001-999) and new 5-digit (00001-99999) formats
   - Automatically detects and maintains the appropriate format

3. **Data Handling**
   - Converts Hugging Face audio format to 16kHz WAV
   - Preserves all metadata (text, duration)
   - Handles special characters in filenames
   - Ensures consistent file structure

4. **Error Handling**
   - Validates downloaded data
   - Continues on individual file errors
   - Provides detailed error logging
   - Creates necessary directories automatically

Common Usage:
```bash
# Basic download (skips existing files)
python download_dataset.py

# Download with custom repository
python download_dataset.py --repo-id "username/dataset-name"

# Use custom metadata location
python download_dataset.py --metadata "path/to/metadata.csv"

# Show detailed help
python download_dataset.py --help-more
```

### Dataset Upload

```bash
python upload_dataset.py [OPTIONS]

Options:
  --repo-id TEXT     Hugging Face repository ID
                     (default: pauljunsukhan/throatmic_codered)
  --token TEXT       Hugging Face API token
                     (can also use HF_TOKEN environment variable)
  --metadata TEXT    Path to metadata CSV file
                     (default: data/metadata/metadata.csv)
  --dataset-card TEXT  Path to dataset card markdown file
                     (default: DATASET_CARD.md)
  --replace         Replace entire dataset with local version
  --cleanup         Clean up duplicates in the dataset
  --help            Show this help message
```

Common Usage:
- Basic upload (adds new files only):
  ```bash
  python upload_dataset.py
  ```

- Replace entire dataset with local version:
  ```bash
  python upload_dataset.py --replace
  ```

- Remove duplicate recordings from the remote dataset:
  ```bash
  python upload_dataset.py --cleanup
  ```
  The cleanup flag scans the remote dataset for duplicate recordings (based on filenames) and removes them, keeping only the first occurrence of each recording.




## 🧹 Dataset Cleanup Tool

The cleanup tool (`cleanup_dataset.py`) helps maintain dataset quality by allowing you to review, delete, and replace recordings interactively.

```bash
python cleanup_dataset.py
```

### Features

- 🎵 Interactive audio playback (press Enter to skip)
- 🗑️ Delete problematic recordings
- 🔄 Replace deleted recordings with new ones
- 📊 Automatic metadata management
- 🔢 Smart file renumbering (supports both 3-digit and 5-digit formats)

### Usage

1. **Review Recordings**
   - Listen to each recording
   - See sentence text and metadata
   - Press Enter to skip playback
   - Choose to keep or delete

2. **Replace Recordings**
   - After deleting, record a new version
   - Metadata is automatically updated
   - Files are renumbered to maintain sequence
   - Preserves existing numbering format (3 or 5 digits)

3. **View Statistics**
   - Total recordings
   - Dataset duration
   - Recent changes

The tool maintains dataset integrity by:
- Preserving sequential file numbering
- Updating metadata automatically
- Moving replacement recordings to correct positions
- Handling both 3-digit and 5-digit filename formats

## 🔍 Analysis Features

### Quality Analysis
```bash
python record_audio.py --analyze --quality
```

The analyzer checks and reports on recording quality:

1. **Audio Requirements**
   - Duration: 6-30 seconds (automatically adjusted based on complexity)
   - Audio levels: -50dB to 0dB
   - Silence ratio: <30% of recording
   - Signal-to-noise ratio: >10dB
   - No clipping or distortion

2. **Analysis Metrics**
   - Mean and max amplitude in dB
   - Silence distribution analysis
   - Background noise assessment
   - Signal clarity measurements
   - Detailed quality issue reports

### Coverage Analysis
```bash
python record_audio.py --analyze --coverage
```
Provides dataset-wide statistics:
- Total recordings and duration
- Sentence complexity distribution
- Vocabulary sophistication metrics
- Syntactic feature coverage
- Logical structure analysis

## 📝 Sentence Processing Pipeline

### Stage 1: Text Validation

1. **Basic Requirements**
   - Length: 12-25 words per sentence
   - Format:
     - Must start with capital letter
     - Must end with proper punctuation (., !, ?)
     - No special characters (except punctuation)
     - No numbers (write them out as words)
     - No all-caps (except common acronyms)

2. **Character Limits**
   - Total characters: 60-200
   - Average word length: 3-12 characters

### Stage 2: Linguistic Analysis
Uses spaCy's medium English model (en_core_web_md) for:

1. **Entity Validation**
   - Max 2 persons (PERSON)
   - Max 1 location (GPE)
   - Max 2 organizations (ORG)
   - Max 3 total entities

2. **Structural Requirements**
   - Clause count: 1-4 clauses
   - Parse tree depth: 2-5 levels
   - Part-of-speech distribution:
     - Verbs: 10-35% of content words
     - Nouns: 20-45% of content words
     - Adjectives: 5-25% of content words

3. **Complex Elements** (must have at least two):
   - **Proper Comma Usage**
     - Separating clauses
     - After introductory phrases
     - In lists (max 4 commas)
   
   - **Clause Types**
     - Subordinate clauses
     - Relative clauses
     - Adverbial clauses
   
   - **Conjunctions**
     - Subordinating (before, after, when, because, although, etc.)
     - Coordinating (and, but, or, nor, for, yet, so)
     - Relative pronouns (that, which, who, whom, whose, where, when)

### Stage 3: Complexity Scoring

Each sentence receives a weighted score (0-13) based on:

1. **Syntactic Complexity** (0-5 points)
   - Clause count: 1.0 points per clause
   - Tree depth: 0.5 points per level

2. **Vocabulary Sophistication** (0-5 points)
   - Rare words (frequency >10000): 0.5 points each
   - Technical terms: 1.0 points each
   - Abstract concepts: 1.0 points each

3. **Logical Complexity** (0-3 points)
   - Causal relationships: 1.0 point
   - Comparisons: 1.0 point
   - Conditional structures: 1.0 point

Final complexity ratings:
- <3: "Simple but clear"
- 3-5: "Moderately complex"
- 6-8: "Complex academic/technical"
- >8: "Highly complex"

### Stage 4: Duration Estimation

The tool automatically calculates expected duration using:

1. **Base Timing**
   - Syllable duration: 300ms per syllable
   - Word boundary pauses: 100ms
   
2. **Punctuation Pauses**
   - Period/Question/Exclamation: 700ms
   - Comma: 300ms
   - Semicolon/Colon: 400ms
   - Parentheses: 300ms

3. **Adjustments**
   - Long word penalty: +200ms for words >6 chars
   - Comprehension buffer: 20% extra time
   - Final range: 6-30 seconds

This ensures natural speech patterns without rushing or excessive pauses.

### Example Sentences
✅ Good Examples:
- "Although the harmonic series does diverge, it does so very slowly."
  - Has subordinate clause ("Although...")
  - Verb ratio: 0.25 (within 0.1-0.35 range)
  - Noun ratio: 0.25 (within 0.2-0.45 range)
  - 2 clauses (within 1-4 range)
  - Parse tree depth: 3 (within 2-5 range)

- "Nick confesses everything to his mother, who tells him that he can't 'control everything'."
  - Has relative clause ("who tells...")
  - Multiple clauses (3, within 1-4 range)
  - Good POS distribution
  - One PERSON entity (within limits)
  - Parse tree depth: 4 (within 2-5 range)

❌ Rejected Examples:
- Too simple: "Bronze tools were sharper and harder than those made of stone."
  - Only one clause (below minimum)
  - No required complex structures
  - Parse tree depth: 2 (at minimum)

- Too many entities: "Emperor Akbar granted them mansabs and their ancestral domains were treated as jagirs."
  - Multiple PERSON and LOC entities (exceeds limits)
  - Too many named entities total (>3)

- Poor POS distribution: "The big red dog ran quickly and jumped high and barked loudly."
  - Too many verbs (>35% of content words)
  - Too many adjectives (>25% of content words)
  - Too many coordinating conjunctions

### Sentence Sources
1. Common Voice Wikipedia Extracts
2. Common Voice Community Contributions
3. Custom user-added sentences

Each source goes through the same validation pipeline to ensure consistent quality and usability for throat mic recording.

### Configuration
All sentence selection criteria can be adjusted in `config.yaml`:
```yaml
sentence_filter:
  # Basic text constraints
  min_chars: 60
  max_chars: 200
  min_words: 5
  max_words: 30
  
  # Entity limits
  max_entities: 3
  max_persons: 2
  max_locations: 1
  max_organizations: 2
  
  # Complexity requirements
  min_complexity_score: 3
  min_total_complexity: 6
  
  # Part-of-speech ratios [min, max]
  pos_ratios:
    VERB: [0.1, 0.35]   # Verbs: 10-35% of content words
    NOUN: [0.2, 0.45]   # Nouns: 20-45% of content words
    ADJ: [0.05, 0.25]   # Adjectives: 5-25% of content words
  
  # Structural complexity ranges
  complexity_ranges:
    clause_count: [1, 4]      # Number of clauses
    word_length: [3, 12]      # Average word length
    tree_depth: [2, 5]        # Parse tree depth levels
  
  # Duration estimation parameters
  timing:
    syllable_duration: 0.3     # 300ms per syllable
    word_boundary_pause: 0.1   # 100ms between words
    long_word_penalty: 0.2     # Extra time for words >6 chars
    comprehension_buffer: 1.2  # Overall timing multiplier
    min_duration: 6.0         # Minimum recording duration
    max_duration: 30.0        # Maximum recording duration

audio:
  # Recording parameters
  sample_rate: 16000          # 16kHz
  channels: 1                 # Mono
  format: wav                 # File format
  
  # Quality thresholds
  min_level_threshold: -50    # Minimum amplitude (dB)
  clipping_threshold: 0       # Maximum amplitude (dB)
  min_duration: 6.0          # Minimum duration (seconds)
  max_duration: 30.0         # Maximum duration (seconds)
  buffer_ratio: 1.2          # Recording buffer multiplier
```

### Technical Implementation
- Uses spaCy's medium English model (en_core_web_md)
- Multi-stage validation for efficiency:
  1. Quick text filtering (regex/string operations)
  2. Entity and complexity validation (spaCy)
  3. Full linguistic analysis (spaCy)
- Comprehensive error handling and logging
- Automated duration estimation
- Detailed quality analysis

## 📋 License

MIT License - feel free to use and modify as needed!