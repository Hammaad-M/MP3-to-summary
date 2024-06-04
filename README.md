# MP3 to Summary

## Installation

Run `pip install -r requirements.txt` in root folder.

## Usage

Change `audio_path` to match your audio input file. Run `python main.py` to launch program. Transcription is stored to `transcription.txt`, updated after processing each 30 second chunk. Afterwards, a full summary is generated and stored to `summary.txt`. 

## Models

[Whisper Medium](https://huggingface.co/openai/whisper-medium) used for transcription, [Facebook BART Large CNN](https://huggingface.co/facebook/bart-large-cnn) used for summarization. 
