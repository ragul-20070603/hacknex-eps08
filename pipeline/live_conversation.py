
import json
import time
import urllib.request
import wave

import numpy as np
import sounddevice as sd
import sherpa_onnx

from live_asr import record_until_silence, transcribe
from safety_filter import safety_filter


# =========================
# CONFIG
# =========================

LLM_URL = "http://127.0.0.1:11434/api/chat"

TTS_DIR = r"llm\models\vits-piper-en_US-amy-low"
TTS_MODEL = TTS_DIR + r"\en_US-amy-low.onnx"
TTS_TOKENS = TTS_DIR + r"\tokens.txt"
TTS_DATA = TTS_DIR + r"\espeak-ng-data"

OUTPUT_WAV = r"results\live_response.wav"


# =========================
# TTS INITIALIZATION
# =========================

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=TTS_MODEL,
            tokens=TTS_TOKENS,
            data_dir=TTS_DATA,
        ),
        num_threads=2,
    ),
    max_num_sentences=1,
)

tts = sherpa_onnx.OfflineTts(tts_config)


# =========================
# LLM
# =========================

def ask_llm(user_text):
    payload = {
        "model": "llama3.2:latest",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a real-time voice assistant. "
                    "Answer in 2-4 concise sentences. "
                    "Avoid long lists unless explicitly requested."
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_predict": 100,
        },
    }

    request = urllib.request.Request(
        LLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    # Ollama timing diagnostics
    print(
        f"Ollama total duration: "
        f"{result.get('total_duration', 0) / 1e9:.3f}s"
    )
    print(
        f"Ollama load duration: "
        f"{result.get('load_duration', 0) / 1e9:.3f}s"
    )
    print(
        f"Ollama prompt evaluation: "
        f"{result.get('prompt_eval_duration', 0) / 1e9:.3f}s"
    )
    print(
        f"Ollama generation duration: "
        f"{result.get('eval_duration', 0) / 1e9:.3f}s"
    )
    print(
        f"Generated tokens: {result.get('eval_count', 0)}"
    )

    return result["message"]["content"].strip()


# =========================
# TTS
# =========================

def speak(text):
    print("🔊 Generating speech...")

    start = time.perf_counter()
    audio = tts.generate(text)
    generation_time = time.perf_counter() - start

    if audio is None:
        print("❌ TTS generation failed.")
        return None, generation_time

    samples = np.asarray(audio.samples)

    if samples.size == 0:
        print("❌ TTS returned empty audio.")
        return None, generation_time

    samples_int16 = np.clip(
        samples * 32767,
        -32768,
        32767,
    ).astype(np.int16)

    # Save generated audio
    with wave.open(OUTPUT_WAV, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(samples_int16.tobytes())

    print("🔊 Playing response...")
    playback_start = time.perf_counter()

    try:
        sd.play(samples, audio.sample_rate)
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("\n⚠️ Playback interrupted by user.")
        return audio, generation_time
    finally:
        sd.stop()

    playback_time = time.perf_counter() - playback_start
    print(f"Playback time: {playback_time:.3f}s")
    print("🔊 Playback complete.")

    return audio, generation_time


# =========================
# MAIN
# =========================

def main():
    print("=" * 60)
    print("HNX26EPS08 LIVE CONVERSATION")
    print("=" * 60)

    # 1. Microphone recording
    audio_input = record_until_silence()

    if audio_input is None:
        print("No speech detected.")
        return

    # 2. ASR
    print("📝 Transcribing...")

    asr_start = time.perf_counter()
    text = transcribe(audio_input)
    asr_time = time.perf_counter() - asr_start

    print(f"User: {text}")
    print(f"ASR: {asr_time:.3f}s")

    if not text or not text.strip():
        print("No transcript produced.")
        return

    # 3. LLM
    print("🧠 Thinking...")

    llm_start = time.perf_counter()
    llm_response = ask_llm(text)
    llm_time = time.perf_counter() - llm_start

    print(f"LLM output: {llm_response}")
    print(f"LLM request time: {llm_time:.3f}s")

    # 4. Safety filter
    print("🛡️ Checking safety...")

    safe_response, blocked = safety_filter(llm_response)

    if blocked:
        print("⚠️ Unsafe response detected.")
    else:
        print("✅ Response passed safety filter.")

    print("Assistant:", safe_response)

    # 5. TTS and playback
    tts_audio, tts_time = speak(safe_response)
    print(f"TTS generation: {tts_time:.3f}s")

    if tts_audio is not None:
        print(f"🔊 Audio saved: {OUTPUT_WAV}")

    print()
    print("=" * 60)
    print("TURN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sd.stop()
        print("\n⚠️ Conversation interrupted.")
    except Exception as exc:
        sd.stop()
        print(f"\n❌ Pipeline error: {exc}")
        raise