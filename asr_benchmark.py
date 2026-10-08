import wave
import time
import json
import os
import statistics
import numpy as np
import sherpa_onnx

MODEL_DIR = r"llm\models\asr\sherpa-onnx-zipformer-small-en-2023-06-26"

ENCODER = MODEL_DIR + r"\encoder-epoch-99-avg-1.int8.onnx"
DECODER = MODEL_DIR + r"\decoder-epoch-99-avg-1.int8.onnx"
JOINER = MODEL_DIR + r"\joiner-epoch-99-avg-1.int8.onnx"
TOKENS = MODEL_DIR + r"\tokens.txt"

WAV_FILE = MODEL_DIR + r"\test_wavs\0.wav"

RESULTS_DIR = r"results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "asr_timing.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

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

with wave.open(WAV_FILE, "rb") as f:
    sample_rate = f.getframerate()
    num_channels = f.getnchannels()
    num_frames = f.getnframes()
    audio_bytes = f.readframes(num_frames)

audio = np.frombuffer(
    audio_bytes,
    dtype=np.int16
).astype(np.float32) / 32768.0

if num_channels > 1:
    audio = audio.reshape(-1, num_channels).mean(axis=1)

audio_duration = len(audio) / sample_rate

print(f"Audio duration: {audio_duration:.3f}s")
print(f"Sample rate: {sample_rate} Hz")
print()
print("Running 5 ASR benchmark runs...\n")

times = []
transcript = ""

for i in range(5):

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)

    start = time.perf_counter()

    recognizer.decode_stream(stream)

    end = time.perf_counter()

    processing_time = end - start
    rtf = processing_time / audio_duration

    transcript = stream.result.text

    times.append(processing_time)

    print(
        f"Run {i + 1}: "
        f"{processing_time:.3f}s | "
        f"RTF: {rtf:.3f}"
    )

median_time = statistics.median(times)
median_rtf = median_time / audio_duration

results = {
    "component": "ASR",
    "model": "sherpa-onnx-zipformer-small-en-2023-06-26-int8",
    "audio_duration_seconds": round(audio_duration, 4),
    "sample_rate": sample_rate,
    "runs": [
        {
            "run": i + 1,
            "processing_time_seconds": round(times[i], 4),
            "rtf": round(times[i] / audio_duration, 4),
        }
        for i in range(5)
    ],
    "median_processing_time_seconds": round(median_time, 4),
    "median_rtf": round(median_rtf, 4),
    "transcript": transcript,
}

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)

print("\n==============================")
print("ASR BENCHMARK RESULTS")
print("==============================")

for i, value in enumerate(times):
    print(f"Run {i + 1}: {value:.3f}s")

print(f"\nMedian processing time: {median_time:.3f}s")
print(f"Audio duration:         {audio_duration:.3f}s")
print(f"Median RTF:             {median_rtf:.3f}")

print("\nTranscript:")
print(transcript)

print("==============================")
print(f"Saved to: {RESULTS_FILE}")
print("==============================")
