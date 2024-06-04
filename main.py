from transformers import WhisperProcessor, TFWhisperForConditionalGeneration
import librosa
import tensorflow as tf
from tqdm import tqdm

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

# Load the local audio file
audio_path = "t.mp3"
audio, sr = librosa.load(audio_path, sr=None)

# Resample the audio to 16000 Hz
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000


chunk_duration = 28  # seconds
chunk_size = chunk_duration * sr

# Open the text file for writing
with open("transcription.txt", "w") as f:
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

print("Full transcription saved to transcription.txt")
