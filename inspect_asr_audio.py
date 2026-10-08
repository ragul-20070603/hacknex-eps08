import wave
from pathlib import Path

import numpy as np

folder = Path("results/asr_tests")
threshold = 0.01  # normalized amplitude; rough diagnostic only

for path in sorted(folder.glob("test_*.wav")):
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        audio = np.frombuffer(
            wav.readframes(wav.getnframes()), dtype=np.int16
        ).astype(np.float32) / 32768.0

    frame_size = int(rate * 0.02)  # 20 ms
    usable = len(audio) // frame_size * frame_size
    frames = audio[:usable].reshape(-1, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    active = rms > threshold
    active_seconds = np.sum(active) * 0.02

    print(
        f"{path.name}: estimated active audio={active_seconds:.2f}s / "
        f"{len(audio) / rate:.2f}s, "
        f"threshold={threshold}"
    )