from transformers import WhisperProcessor, TFWhisperForConditionalGeneration # For transcription
from transformers import pipeline  # For text summarization
import logging
logging.getLogger("transformers").setLevel(logging.ERROR) 
import librosa
import tensorflow as tf
from tqdm import tqdm

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

# # Load the local audio file
audio_path = "t.mp3"
transcript_file = "transcription.txt" # output for transcription
summary_file = "summary.txt" # output for summary
audio, sr = librosa.load(audio_path, sr=None)

# Resample the audio to 16000 Hz
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000


chunk_duration = 28  # seconds
chunk_size = chunk_duration * sr

# Open the text file for writing
with open(transcript_file, "w") as f:
    # Process the audio in chunks and transcribe
    with tqdm(total=(len(audio) // chunk_size ) + 1, desc="Processing chunks") as pbar:
        for i, start in enumerate(range(0, len(audio), chunk_size), 0):
            end = min(start + chunk_size, len(audio))
            audio_chunk = audio[start:end]

            # Process the audio chunk
            input_features = processor(audio_chunk, sampling_rate=sr, return_tensors="tf").input_features

            # Ensure the input features are float type
            input_features = tf.cast(input_features, tf.float32)

            # Generate token ids
            predicted_ids = model.generate(input_features)

            # Decode token ids to text
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            f.write(transcription + "\n")  # Write transcription to file
            f.flush()  # Flush the buffer to ensure the data is written to file immediately

            # Update progress bar
            pbar.update(1)

print("Full transcription saved to", transcript_file, "\nRunning summarization...")


# Generate summary

summary = []  # List to store summarized chunks

with open(transcript_file, 'r', encoding='utf-8') as input_file:
    with open(summary_file, 'w', encoding='utf-8') as output_file:
        text = input_file.read()  # Read input text
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Initialize summarizer model
        text = [text[i:i+2800] for i in range(0, len(text), 2800)]  # Break text into chunks of 1200 characters

        for chunk in tqdm(text, desc="Summarizing transcript chunks", unit="chunks"):  # Iterate over text chunks
            print(min(500, len(chunk)))
            summary = summarizer(chunk, max_length=min(350, len(chunk)), min_length=10, do_sample=True)  # Summarize each chunk, max_length is for the summary output
            output_file.write(summary[0]['summary_text'] + "\n") 

print("\nSummary saved to", summary_file)