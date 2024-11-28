# Audio Transcription Tool

## Overview
This Python script provides a versatile tool for transcribing audio from multiple sources using OpenAI's Whisper model. It supports:
- Recording audio directly from your microphone
- Transcribing existing audio files
- Extracting and transcribing audio from video files

## Prerequisites
- Python 3.10+
- CUDA-compatible GPU highly recommended (but not required)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LukeDitria/whisper_transcriber.git
cd whisper_transcriber
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies
- PyTorch: Deep learning framework
- Transformers: Hugging Face's model library
- PyAudio: Audio recording
- MoviePy: Video audio extraction
- Pynput: Keyboard event handling

## Usage

Run the script:
```bash
python to_transcript.py
```

### Input Options
1. **Record Audio**: Press spacebar to start and stop recording
2. **Transcribe Audio File**: Provide path to an existing .wav or .mp3 file
3. **Transcribe Video File**: Extract and transcribe audio from a video

### Outputs
- Transcription will be saved as a text file
- Filename based on source file with incremental numbering
