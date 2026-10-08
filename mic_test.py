import sounddevice as sd
import wave
import numpy as np

SAMPLE_RATE = 16000
DURATION = 5

print("🎙️ Recording for 5 seconds...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)
sd.wait()

audio = np.clip(audio, -1, 1)
audio_int16 = (audio * 32767).astype(np.int16)

with wave.open("results\\mic_test.wav", "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(SAMPLE_RATE)
    f.writeframes(audio_int16.tobytes())

print("✅ Saved: results\\mic_test.wav")