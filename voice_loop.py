import time
import wave
import requests
import numpy as np
import sounddevice as sd
import sherpa_onnx
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"E:\hacknex-eps08")

ASR_MODEL = BASE / r"llm\models\asr\sherpa-onnx-zipformer-small-en-2023-06-26"

TTS_MODEL_DIR = BASE / r"llm\models\vits-piper-en_US-amy-low"
TTS_MODEL = TTS_MODEL_DIR / "en_US-amy-low.onnx"
TTS_TOKENS = TTS_MODEL_DIR / "tokens.txt"
TTS_DATA = TTS_MODEL_DIR / "espeak-ng-data"

MIC_AUDIO = BASE / r"results\voice_input.wav"
OUTPUT_AUDIO = BASE / r"results\voice_output.wav"


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
RECORD_SECONDS = 5
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"


# ============================================================
# MICROPHONE
# ============================================================

print("\n==========================================")
print("       HNX26EPS08 VOICE LOOP")
print("==========================================")

print("\n🎙️ Recording...")
print("Speak now!")

audio = sd.rec(
    int(RECORD_SECONDS * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)

sd.wait()

audio = np.clip(audio, -1, 1)
audio_int16 = (audio * 32767).astype(np.int16)

with wave.open(str(MIC_AUDIO), "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(SAMPLE_RATE)
    f.writeframes(audio_int16.tobytes())

print("✅ Recording finished.")


# ============================================================
# LOAD ASR
# ============================================================

print("\n📝 Loading ASR...")

recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=str(ASR_MODEL / "encoder-epoch-99-avg-1.int8.onnx"),
    decoder=str(ASR_MODEL / "decoder-epoch-99-avg-1.int8.onnx"),
    joiner=str(ASR_MODEL / "joiner-epoch-99-avg-1.int8.onnx"),
    tokens=str(ASR_MODEL / "tokens.txt"),
    num_threads=2,
    decoding_method="greedy_search",
)

print("ASR loaded.")


# ============================================================
# ASR
# ============================================================

print("\n🔎 Running ASR...")

asr_start = time.perf_counter()

with wave.open(str(MIC_AUDIO), "rb") as f:
    stream = recognizer.create_stream()

    samples = f.readframes(f.getnframes())
    samples = (
        np.frombuffer(samples, dtype=np.int16)
        .astype(np.float32)
        / 32768.0
    )

    stream.accept_waveform(
        f.getframerate(),
        samples,
    )

recognizer.decode_stream(stream)

transcript = stream.result.text.strip()

asr_time = time.perf_counter() - asr_start

print("\n🎙️ Transcript:")
print(transcript)

print(f"ASR time: {asr_time:.3f}s")


if not transcript:
    print("\n❌ No speech detected.")
    raise SystemExit


# ============================================================
# LLM
# ============================================================

print("\n🧠 Sending transcript to Qwen...")

payload = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant. "
                "Reply in one or two short sentences."
            ),
        },
        {
            "role": "user",
            "content": transcript,
        },
    ],
    "temperature": 0,
    "max_tokens": 100,
    "stream": False,
}

llm_start = time.perf_counter()

response = requests.post(
    LLM_URL,
    json=payload,
    timeout=60,
)

response.raise_for_status()

answer = response.json()["choices"][0]["message"]["content"].strip()

llm_time = time.perf_counter() - llm_start

print("\n🤖 Qwen:")
print(answer)

print(f"LLM time: {llm_time:.3f}s")


# ============================================================
# SAFETY FILTER
# ============================================================

print("\n🛡️ Safety filter...")

BLOCKED_TERMS = [
    "malware",
    "ransomware",
    "keylogger",
    "phishing",
]

answer_lower = answer.lower()

blocked = any(
    term in answer_lower
    for term in BLOCKED_TERMS
)

if blocked:
    print("⚠️ Unsafe response detected.")
    answer = (
        "I can't provide instructions for that. "
        "I can help with a safe alternative."
    )

print("✅ Response passed safety gate.")


# ============================================================
# LOAD TTS
# ============================================================

print("\n🔊 Loading TTS...")

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(TTS_MODEL),
            tokens=str(TTS_TOKENS),
            data_dir=str(TTS_DATA),
        )
    ),
    max_num_sentences=1,
)

tts = sherpa_onnx.OfflineTts(tts_config)

print("TTS loaded.")


# ============================================================
# TTS
# ============================================================

print("\n🔊 Generating speech...")

tts_start = time.perf_counter()

tts_audio = tts.generate(
    text=answer,
    sid=0,
    speed=1.0,
)

tts_time = time.perf_counter() - tts_start

samples = np.asarray(
    tts_audio.samples,
    dtype=np.float32,
)

samples = np.clip(samples, -1.0, 1.0)

samples_int16 = (
    samples * 32767
).astype(np.int16)

with wave.open(str(OUTPUT_AUDIO), "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(tts_audio.sample_rate)
    f.writeframes(samples_int16.tobytes())

print(f"TTS time: {tts_time:.3f}s")
print(f"Saved: {OUTPUT_AUDIO}")


# ============================================================
# PLAYBACK
# ============================================================

print("\n🔈 Playing response...")

sd.play(
    samples,
    tts_audio.sample_rate,
    device=4,
)

sd.wait()

print("Playback finished.")


# ============================================================
# SUMMARY
# ============================================================

total_time = (
    asr_time
    + llm_time
    + tts_time
)

print("\n==========================================")
print("              VOICE LOOP")
print("==========================================")

print(f"ASR       : {asr_time:.3f}s")
print(f"LLM       : {llm_time:.3f}s")
print(f"TTS       : {tts_time:.3f}s")
print(f"Processing: {total_time:.3f}s")

print("------------------------------------------")
print("🎙️ Microphone → ASR → Qwen → Filter → TTS → 🔊")
print("==========================================")