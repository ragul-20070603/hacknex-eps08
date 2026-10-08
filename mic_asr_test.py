import sherpa_onnx
import wave
from pathlib import Path

MODEL = Path(
    r"llm\models\asr\sherpa-onnx-zipformer-small-en-2023-06-26"
)
AUDIO = Path(r"results\mic_test.wav")

recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=str(MODEL / "encoder-epoch-99-avg-1.int8.onnx"),
    decoder=str(MODEL / "decoder-epoch-99-avg-1.int8.onnx"),
    joiner=str(MODEL / "joiner-epoch-99-avg-1.int8.onnx"),
    tokens=str(MODEL / "tokens.txt"),
    num_threads=2,
    decoding_method="greedy_search",
)

print("Reading microphone audio...")

with wave.open(str(AUDIO), "rb") as f:
    stream = recognizer.create_stream()
    samples = f.readframes(f.getnframes())
    import numpy as np
    samples = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0
    stream.accept_waveform(f.getframerate(), samples)

recognizer.decode_stream(stream)

print()
print("🎙️ Transcript:")
print(stream.result.text)