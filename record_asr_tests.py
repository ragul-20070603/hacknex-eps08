
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

OUT = Path("results/asr_tests")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 8

sentences = [
    "What are the advantages and disadvantages of artificial intelligence?",
    "Will artificial intelligence take over IT jobs?",
    "Explain the difference between machine learning and deep learning.",
    "Who is the CEO of Tesla?",
    "How can I prepare for a software engineering interview?",
]

for i, sentence in enumerate(sentences, start=1):
    input(f"\nTest {i}/5: Press Enter, then speak the sentence below.\n"
          f"Sentence: {sentence}\n")

    print(f"Recording for {DURATION} seconds...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    path = OUT / f"test_{i}.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(audio.tobytes())

    print(f"Saved: {path}")

print("\nAll five recordings saved.")