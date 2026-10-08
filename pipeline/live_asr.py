
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.1
SILENCE_DURATION = 1.2
ENERGY_THRESHOLD = 0.008
MIN_SPEECH_DURATION = 0.3
MAX_RECORDING_DURATION = 15

OUTPUT_DIR = Path("results")
AUDIO_PATH = OUTPUT_DIR / "live_input.wav"

MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

print("Loading faster-whisper...")
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)
print("Whisper ready.")


def record_until_silence():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    block_size = int(SAMPLE_RATE * BLOCK_DURATION)
    audio_blocks = []
    speech_started = False
    speech_duration = 0.0
    silence_duration = 0.0
    elapsed = 0.0

    print("\n🎙️ Listening... Speak now.")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=block_size
        ) as stream:

            while elapsed < MAX_RECORDING_DURATION:
                block, overflowed = stream.read(block_size)
                block = block[:, 0].copy()
                energy = float(np.sqrt(np.mean(block ** 2)))

                if not speech_started:
                    if energy >= ENERGY_THRESHOLD:
                        speech_started = True
                        speech_duration += BLOCK_DURATION
                        audio_blocks.append(block)
                        print("🗣️ Speech detected.")
                else:
                    audio_blocks.append(block)
                    speech_duration += BLOCK_DURATION

                    if energy < ENERGY_THRESHOLD:
                        silence_duration += BLOCK_DURATION
                    else:
                        silence_duration = 0.0

                    if silence_duration >= SILENCE_DURATION:
                        print("🔇 End of speech detected.")
                        break

                elapsed += BLOCK_DURATION

    except KeyboardInterrupt:
        print("\nRecording cancelled.")
        return None

    if not speech_started or speech_duration < MIN_SPEECH_DURATION:
        print("No speech detected.")
        return None

    audio = np.concatenate(audio_blocks).astype(np.float32)
    sf.write(AUDIO_PATH, audio, SAMPLE_RATE)

    print(f"Saved recording: {AUDIO_PATH}")
    return audio


def transcribe(audio_input):
    if audio_input is None or len(audio_input) == 0:
        return ""

    print("📝 Running Whisper ASR...")
    start = time.perf_counter()

    segments, info = model.transcribe(
        audio_input,
        language="en",
        beam_size=5,
        vad_filter=False
    )

    text = " ".join(segment.text.strip() for segment in segments).strip()

    print(f"Language: {info.language}")
    print(f"Whisper time: {time.perf_counter() - start:.3f}s")

    return text


if __name__ == "__main__":
    audio = record_until_silence()
    if audio is not None:
        print("Transcript:", transcribe(audio))