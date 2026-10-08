
import re
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

ROOT = Path("llm/models/asr")
TEST_DIR = Path("results/asr_tests")

MODELS = {
    "SMALL_INT8": ROOT / "sherpa-onnx-zipformer-small-en-2023-06-26",
    "LARGE_INT8": ROOT / "sherpa-onnx-zipformer-large-en-2023-06-26",
}

REFERENCES = {
    1: "What are the advantages and disadvantages of artificial intelligence?",
    2: "Will artificial intelligence take over IT jobs?",
    3: "Explain the difference between machine learning and deep learning.",
    4: "Who is the CEO of Tesla?",
    5: "How can I prepare for a software engineering interview?",
}

def normalize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def word_errors(reference, hypothesis):
    r = normalize(reference)
    h = normalize(hypothesis)

    # Word-level Levenshtein distance
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]

    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    errors = dp[-1][-1]
    wer = errors / max(1, len(r)) * 100
    return errors, len(r), wer

def load_audio(path):
    with wave.open(str(path), "rb") as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit PCM WAV: {path}")
        sample_rate = f.getframerate()
        frames = f.getnframes()
        raw = f.readframes(frames)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return sample_rate, audio

def main():
    recognizers = {}

    for name, model_dir in MODELS.items():
        print(f"\nLoading {name}...")
        start = time.perf_counter()

        recognizers[name] = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_dir / "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(model_dir / "decoder-epoch-99-avg-1.int8.onnx"),
            joiner=str(model_dir / "joiner-epoch-99-avg-1.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=2,
            decoding_method="modified_beam_search",
        )

        print(f"Model load: {time.perf_counter() - start:.2f}s")

    totals = {
        name: {"errors": 0, "words": 0, "times": []}
        for name in MODELS
    }

    for i, reference in REFERENCES.items():
        path = TEST_DIR / f"test_{i}.wav"
        sample_rate, audio = load_audio(path)

        print(f"\n{'=' * 65}")
        print(f"TEST {i}: {reference}")
        print(f"Audio duration: {len(audio) / sample_rate:.2f}s")

        for name, recognizer in recognizers.items():
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio)

            start = time.perf_counter()
            recognizer.decode_stream(stream)
            elapsed = time.perf_counter() - start

            hypothesis = stream.result.text
            errors, words, wer = word_errors(reference, hypothesis)

            totals[name]["errors"] += errors
            totals[name]["words"] += words
            totals[name]["times"].append(elapsed)

            print(f"\n{name}")
            print(f"Transcript: {hypothesis}")
            print(f"Word errors: {errors}/{words} | WER: {wer:.1f}%")
            print(f"Decode time: {elapsed:.3f}s")

    print(f"\n{'=' * 65}")
    print("FINAL SUMMARY")
    for name, values in totals.items():
        overall_wer = values["errors"] / max(1, values["words"]) * 100
        median_time = float(np.median(values["times"]))
        print(
            f"{name}: overall WER={overall_wer:.1f}%, "
            f"median decode={median_time:.3f}s"
        )

if __name__ == "__main__":
    main()