# 🎙 Throat Microphone Recording Tool

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
  --cleanup         Clean up duplicates in the dataset
  --help            Show this help message
```

## 🔍 Analysis Features

### Quality Analysis
```bash
python record_audio.py --analyze --quality
```
Checks recordings for:
- Duration within limits (8-12 seconds)
- Proper amplitude levels (-50dB to 0dB)
- Silence ratio (<30%)
- Signal-to-noise ratio (>10dB)

### Coverage Analysis
```bash
python record_audio.py --analyze --coverage
```
Shows dataset statistics:
- Total recordings and duration
- Sentence length distribution
- Sentence complexity metrics
- Recording duration stats

## 📝 Sentence Selection Pipeline

### Stage 1: Quick Text Filtering
- Capitalization check (must start with capital letter)
- Punctuation check (must end with ., !, or ?)
- Character length (60-200 chars)
- Basic formatting validation

### Stage 2: spaCy-based Quick Validation
- Word count (5-30 words)
- Named Entity limits (using spaCy's NER):
  - Max 2 persons (PERSON)
  - Max 1 location (GPE)
  - Max 2 organizations (ORG)
  - Max 3 total entities

### Stage 3: Full Complexity Analysis
Uses spaCy's medium English model (en_core_web_md) with flexible ranges:

1. **Part-of-Speech Distribution**
   - Verbs: 10-35% of content words
   - Nouns: 20-45% of content words
   - Adjectives: 5-25% of content words

2. **Structural Complexity**
   - Clause count: 1-4 clauses
   - Parse tree depth: 2-5 levels
   - Average word length: 3-12 characters

3. **Required Elements** (must have at least one):
   - Coordinating conjunction (dep_ == 'cc')
   - Subordinating clause marker (dep_ == 'mark')
   - Relative clause (dep_ == 'relcl')

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
  min_chars: 60
  max_chars: 200
  min_words: 5
  max_words: 30
  max_entities: 3
  max_persons: 2
  max_locations: 1
  max_organizations: 2
  pos_ratios:
    VERB: [0.1, 0.35]
    NOUN: [0.2, 0.45]
    ADJ: [0.05, 0.25]
  complexity_ranges:
    clause_count: [1, 4]
    word_length: [3, 12]
    tree_depth: [2, 5]
```

### Technical Implementation
- Uses spaCy's medium English model (en_core_web_md)
- Multi-stage validation for efficiency:
  1. Quick text filtering (regex/string operations)
  2. Basic spaCy checks (entities, length)
  3. Full linguistic analysis (POS, complexity)
- Comprehensive error handling and logging
- Configurable ranges for natural language variation

## 📋 Sentence Criteria

### Basic Requirements

- **Length**: 12-25 words per sentence
- **Duration**: 8-12 seconds when spoken naturally
- **Format**:
  - Must start with capital letter
  - Must end with proper punctuation (., !, ?)
  - No special characters (except standard punctuation)
  - No numbers (write them out as words)
  - No all-caps (except common acronyms)

### Complexity Requirements

Each sentence should have at least two of these elements:

1. **Proper Comma Usage**
   - Separating clauses
   - After introductory phrases
   - In lists
   - Maximum 4 commas per sentence

2. **Complex Structure**
   - **Subordinating Conjunctions**:
     - Time: before, after, when, whenever, while, since
     - Cause/Effect: because, since, as, unless
     - Contrast: although, though, whereas
     - Condition: if, unless, whether
   - **Coordinating Conjunctions**:
     - and, but, or, nor, for, yet, so
   - **Relative Pronouns**:
     - that, which, who, whom, whose, where, when

3. **Natural Flow**
   - Balanced clause structure
   - Natural speaking rhythm
   - No run-on sentences
   - Clear subject-verb relationships

### Quality Thresholds

Audio recordings must meet these criteria:
- Duration: 6-30 seconds (automatically adjusted based on sentence complexity)
- Audio levels: -50dB to 0dB
- Silence ratio: <30% of recording
- Signal-to-noise ratio: >10dB

### Variable Recording Length

The tool automatically adjusts recording duration based on sentence complexity:

- Analyzes sentence structure, syllables, and pauses
- Estimates natural speaking time for each prompt
- Adds buffer time for comfortable pacing
- Supports sentences from 6-30 seconds (matching Whisper's capabilities)
- Smart timing features:
  - Syllable-based duration calculation
  - Pause detection for punctuation
  - Word boundary timing
  - Natural speech rhythm adaptation

This ensures recordings are neither rushed nor padded with silence, leading to more natural speech patterns and better training data.

## 🛠️ Technical Details

- Uses spaCy's medium English model (en_core_web_md) for linguistic analysis
- Multi-stage sentence validation pipeline:
  1. Quick text filtering (regex/string operations)
  2. Entity and complexity validation (spaCy)
  3. Full linguistic analysis (spaCy)
- Efficient batch processing with random sampling
- Comprehensive audio quality checks
- Automated metadata generation for Hugging Face datasets

## 📝 License

MIT License - feel free to use and modify as needed!