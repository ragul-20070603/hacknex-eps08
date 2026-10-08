import wave
import sherpa_onnx

MODEL_DIR = r"llm\models\asr\sherpa-onnx-zipformer-small-en-2023-06-26"

ENCODER = MODEL_DIR + r"\encoder-epoch-99-avg-1.int8.onnx"
DECODER = MODEL_DIR + r"\decoder-epoch-99-avg-1.int8.onnx"
JOINER = MODEL_DIR + r"\joiner-epoch-99-avg-1.int8.onnx"
TOKENS = MODEL_DIR + r"\tokens.txt"

WAV_FILE = MODEL_DIR + r"\test_wavs\0.wav"

print("Loading ASR model...")

recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=ENCODER,
    decoder=DECODER,
    joiner=JOINER,
    tokens=TOKENS,
    num_threads=2,
    decoding_method="greedy_search",
)

print("ASR model loaded.\n")

print("Reading audio...")

with wave.open(WAV_FILE, "rb") as f:
    sample_rate = f.getframerate()
    num_channels = f.getnchannels()
    sample_width = f.getsampwidth()
    num_frames = f.getnframes()

    audio_bytes = f.readframes(num_frames)

print("Sample rate:", sample_rate)
print("Channels:", num_channels)
print("Sample width:", sample_width)
print("Frames:", num_frames)

import numpy as np

audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

if num_channels > 1:
    audio = audio.reshape(-1, num_channels).mean(axis=1)

stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, audio)

print("\nDecoding...")

recognizer.decode_stream(stream)

print("\n==============================")
print("ASR RESULT")
print("==============================")
print(stream.result.text)
print("==============================")
