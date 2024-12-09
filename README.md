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

## Output

- WAV files are saved in the `throatmic_data` directory
- All files are recorded in the format required by Whisper:
  - 16kHz sample rate
  - 16-bit PCM
  - Mono channel
- `metadata.csv` contains the mapping between audio files and transcripts
- Progress is saved in `recording_progress.json` 