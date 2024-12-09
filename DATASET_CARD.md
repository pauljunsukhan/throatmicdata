---
language:
- en
license: mit
task_categories:
- automatic-speech-recognition
task_ids:
- audio-intent-classification
pretty_name: Throat Microphone Dataset
size_categories:
- 1K<n<10K
---

# Throat Microphone Dataset

## Dataset Description

- **Homepage:** [GitHub Repository](https://github.com/pauljunsukhan/throatmic_codered)
- **Repository:** [Hugging Face Repository](https://huggingface.co/datasets/pauljunsukhan/throatmic_codered)
- **Paper:** N/A
- **Point of Contact:** Paul Han

### Dataset Summary

A high-quality dataset of throat microphone (laryngophone) recordings specifically designed for fine-tuning Whisper and other speech recognition models. The dataset consists of carefully selected English sentences recorded using a throat microphone, which captures speech through vibrations in the throat rather than air-conducted sound.

### Supported Tasks

This dataset is suitable for:
- Automatic Speech Recognition (ASR)
- Speech-to-Text
- Voice Activity Detection
- Throat Microphone Adaptation

### Languages

The dataset contains English-language recordings only.

## Dataset Structure

### Data Instances

Each instance in the dataset contains:
- `audio`: A dictionary containing the audio data:
  - `bytes`: The raw audio bytes
  - `path`: Path to the audio file
  - `sampling_rate`: 16000 (16kHz)
- `text`: The transcription of the audio
- `duration`: Length of the audio in seconds

### Data Fields

- `audio`: Audio file in WAV format (16kHz mono)
- `text`: String containing the transcription
- `duration`: Float value representing duration in seconds

### Data Splits

The dataset is provided as a single training split.

## Dataset Creation

### Curation Rationale

This dataset was created to address the lack of high-quality throat microphone data for training speech recognition models. Throat microphones are particularly useful in noisy environments as they capture speech directly through throat vibrations.

### Source Data

#### Initial Data Collection and Normalization

Sentences were carefully selected to ensure:
- Complexity suitable for model training (12-25 words)
- Proper grammar and punctuation
- Mix of statement types
- Natural language patterns
- Varied vocabulary
- Balanced phonetic content

### Annotations

The annotations (transcriptions) are the original sentences used for recording, ensuring 100% accuracy.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset can help improve speech recognition in:
- High-noise environments
- Military and emergency services communications
- Industrial settings
- Assistive technology for voice disorders

### Discussion of Biases

The dataset:
- Contains only English language
- Uses standard English pronunciation
- May not represent all accents or dialects
- Recorded by a limited number of speakers

### Other Known Limitations

- Limited to throat microphone recordings
- May not generalize well to regular microphone input
- Fixed recording duration of approximately 10 seconds per utterance

## Additional Information

### Dataset Curators

This dataset was curated by Paul Han and contributors.

### Licensing Information

This dataset is released under the MIT License.

### Citation Information

If you use this dataset, please cite:

```
@misc{throatmic_dataset,
  title={Throat Microphone Dataset for Speech Recognition},
  author={Han, Paul},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/pauljunsukhan/throatmic_codered}}
}
```

### Contributions

Thanks to all contributors who helped create and maintain this dataset. 