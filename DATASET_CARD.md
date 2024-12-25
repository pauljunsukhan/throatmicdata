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
tags:
- audio
- speech
- throat-microphone
- whisper
- asr
dataset_info:
  features:
    - name: audio
      dtype: audio
    - name: text
      dtype: string
    - name: duration
      dtype: float32
---

# Throat Microphone Dataset

🤗 **Hugging Face**: [pauljunsukhan/throatmic_codered](https://huggingface.co/datasets/pauljunsukhan/throatmic_codered)  
📦 **GitHub**: [pauljunsukhan/throatmicdata](https://github.com/pauljunsukhan/throatmicdata)

🚀 **Fine-tuned Model**:
- 🤗 [pauljunsukhan/throatmic_subvocalization_whisper](https://huggingface.co/pauljunsukhan/throatmic_subvocalization_whisper)
- 📦 [pauljunsukhan/whisper_finetuning](https://github.com/pauljunsukhan/whisper_finetuning)

## Dataset Description

- **Homepage:** [GitHub Repository](https://github.com/pauljunsukhan/throatmicdata)
- **Repository:** [Hugging Face Repository](https://huggingface.co/datasets/pauljunsukhan/throatmic_codered)
- **Paper:** N/A
- **Point of Contact:** Paul Han

### Hardware Setup

- **Throat Microphone**: [CodeRed Assault MOD Tactical Throat Mic Headset](https://coderedheadsets.com/assault-mod-tactical-throat-mic-headset/)
  - Uses standard 3.5mm audio jack
  - Direct connection to MacBook Air's 3.5mm port
  - Note: Many USB-C to 3.5mm microphone adapters are not compatible with throat microphones

### Dataset Summary

A high-quality dataset of throat microphone (laryngophone) recordings specifically designed for fine-tuning Whisper and other speech recognition models. The dataset consists of 724 carefully selected English sentences recorded using a throat microphone, which captures speech through vibrations in the throat rather than air-conducted sound.

This dataset is particularly valuable for:
- Training ASR models for noisy environments
- Adapting speech recognition to throat microphone input
- Developing robust voice activity detection
- Research in alternative speech input methods

### Dataset Statistics

- **Total Recordings:** 724
- **Total Duration:** 120.1 minutes
- **Average Duration:** 9.9 seconds (median: 10.0s)
- **Duration Range:** 5.9s - 11.0s
- **Standard Deviation:** 0.3s

Duration Distribution:
- <6s: 0.1% (1 recording)
- 6-8s: 0.8% (6 recordings)
- 8-10s: 98.6% (714 recordings)
- 10-12s: 0.4% (3 recordings)
- 12s or greater: 0%

### Linguistic Characteristics

**Vocabulary Statistics:**
- Total Words: 9,334
- Unique Words: 3,629
- Vocabulary Density: 0.39
- Average Sentence Length: 13.0 words (range: 9-18)

**Part of Speech Distribution:**
- Nouns: 22.4%
- Proper Nouns: 12.1%
- Prepositions: 11.4%
- Verbs: 11.0%
- Determiners: 9.8%
- Adjectives: 9.1%
- Auxiliaries: 6.4%
- Pronouns: 5.2%
- Coordinating Conjunctions: 4.5%
- Adverbs: 4.3%
- Other: 13.8%

**Complexity Metrics:**
- Average Complexity Score: 8.0 (on a scale of 0-13)
- Average Tree Depth: 5.5
- Subordinate Clauses: 9.7%
- Relative Clauses: 9.7%
- Adverbial Clauses: 14.5%

**Complexity Distribution:**
- Simple: 0 sentences
- Moderate: 47 sentences (6.5%)
- Complex: 432 sentences (59.7%)
- Very Complex: 245 sentences (33.8%)

### Supported Tasks

This dataset is suitable for:
- **Automatic Speech Recognition (ASR)**: Training models to transcribe throat microphone audio
- **Speech-to-Text**: Converting throat microphone recordings to text
- **Voice Activity Detection**: Detecting speech in throat microphone signals
- **Domain Adaptation**: Adapting existing ASR models to throat microphone input

### Languages

The dataset contains English-language recordings only, with:
- Standard American English pronunciation
- Academic and technical vocabulary
- Complex sentence structures
- High-quality transcriptions

## Dataset Structure

### Data Instances

Each instance in the dataset contains:
```python
{
    'audio': {
        'path': str,          # Path to the audio file
        'array': np.array,    # The audio signal array
        'sampling_rate': int  # 16000 (16kHz)
    },
    'text': str,             # The transcription
    'duration': float        # Length in seconds
}
```

### Audio Quality

All recordings meet strict quality requirements:
- Sample Rate: 16kHz mono
- Duration: Typically 8-10 seconds (98.6% of recordings)
- Audio Levels: -50dB to 0dB
- Signal-to-Noise Ratio: >10dB
- Silence Ratio: <30%
- Clean, professional recording environment

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
- Complexity suitable for model training (9-18 words)
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
- Optimized for 8-10 second utterances

## Additional Information

### Dataset Curators

This dataset was curated by Paul Han

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
