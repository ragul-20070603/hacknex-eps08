import re
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

MODEL = Path("llm/models/asr/sherpa-onnx-zipformer-large-en-2023-06-26")
TEST_DIR = Path("results/asr_tests")

REFERENCES = {
    1: "What are the advantages and disadvantages of artificial intelligence?",
    2: "Will artificial intelligence take over IT jobs?",
    3: "Explain the difference between machine learning and deep learning.",
    4: "Who is the CEO of Tesla?",
    5: "How can I prepare for a software engineering interview?",
}

def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def wer(reference, hypothesis):
    r, h = words(reference), words(hypothesis)
    dp = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        new = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            new[j] = min(
                dp[j] + 1,
                new[j - 1] + 1,
                dp[j - 1] + (rw != hw),
            )
        dp = new
    return dp[-1] / max(1, len(r)) * 100

def load_audio(path):
    with wave.open(str(path), "rb") as f:
        rate = f.getframerate()
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return rate, audio.astype(np.float32) / 32768.0

def trim_silence(audio, rate):
    frame_size = int(rate * 0.02)
    count = len(audio) // frame_size
    frames = audio[:count * frame_size].reshape(-1, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    active = np.where(rms > 0.01)[0]
    if len(active) == 0:
        return audio

    # Keep 200 ms before and after detected activity.
    start = max(0, active[0] * frame_size - int(rate * 0.2))
    end = min(len(audio), (active[-1] + 1) * frame_size + int(rate * 0.2))
    return audio[start:end]

def main():
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(MODEL / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(MODEL / "decoder-epoch-99-avg-1.int8.onnx"),
        joiner=str(MODEL / "joiner-epoch-99-avg-1.int8.onnx"),
        tokens=str(MODEL / "tokens.txt"),
        num_threads=2,
        decoding_method="greedy_search",
    )

    for i, reference in REFERENCES.items():
        rate, audio = load_audio(TEST_DIR / f"test_{i}.wav")
        trimmed = trim_silence(audio, rate)

        print(f"\nTEST {i}: {reference}")
        print(f"Original {len(audio)/rate:.2f}s | trimmed {len(trimmed)/rate:.2f}s")

        for label, signal in [("ORIGINAL", audio), ("TRIMMED", trimmed)]:
            stream = recognizer.create_stream()
            stream.accept_waveform(rate, signal)
            start = time.perf_counter()
            recognizer.decode_stream(stream)
            elapsed = time.perf_counter() - start
            text = stream.result.text

            print(f"{label}: {text}")
            print(f"WER={wer(reference, text):.1f}% | decode={elapsed:.3f}s")

if __name__ == "__main__":
    main()