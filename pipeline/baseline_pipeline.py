import json
import time
import wave
import requests
from pathlib import Path

import numpy as np
import sherpa_onnx


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

ASR_MODEL = (
    ROOT
    / "llm"
    / "models"
    / "asr"
    / "sherpa-onnx-zipformer-small-en-2023-06-26"
)

AUDIO_FILE = ASR_MODEL / "test_wavs" / "0.wav"

TTS_MODEL = (
    ROOT
    / "llm"
    / "models"
    / "vits-piper-en_US-amy-low"
)

TTS_MODEL_FILE = TTS_MODEL / "en_US-amy-low.onnx"
TTS_CONFIG_FILE = TTS_MODEL / "en_US-amy-low.onnx.json"

OUTPUT_FILE = ROOT / "results" / "baseline_output.wav"


# ============================================================
# LLM
# ============================================================

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Reply in one or two short sentences."
)

MAX_TOKENS = 100


# ============================================================
# ASR
# ============================================================

def load_audio(filename):

    with wave.open(str(filename), "rb") as f:
        sample_rate = f.getframerate()
        channels = f.getnchannels()
        sample_width = f.getsampwidth()
        frames = f.readframes(f.getnframes())

    if sample_width != 2:
        raise ValueError("Expected 16-bit WAV audio")

    audio = (
        np.frombuffer(frames, dtype=np.int16)
        .astype(np.float32)
        / 32768.0
    )

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return sample_rate, audio


def create_asr():

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(
            ASR_MODEL / "encoder-epoch-99-avg-1.int8.onnx"
        ),
        decoder=str(
            ASR_MODEL / "decoder-epoch-99-avg-1.int8.onnx"
        ),
        joiner=str(
            ASR_MODEL / "joiner-epoch-99-avg-1.int8.onnx"
        ),
        tokens=str(
            ASR_MODEL / "tokens.txt"
        ),
        num_threads=2,
        decoding_method="greedy_search",
    )


def run_asr(recognizer, sample_rate, audio):

    stream = recognizer.create_stream()

    stream.accept_waveform(
        sample_rate,
        audio,
    )

    recognizer.decode_stream(stream)

    return stream.result.text.strip()


# ============================================================
# LLM
# ============================================================

def run_llm(question):

    body = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    reply = ""
    first_token_time = None

    start = time.perf_counter()

    response = requests.post(
        LLM_URL,
        json=body,
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    for line in response.iter_lines():

        if not line:
            continue

        text = line.decode("utf-8")

        if not text.startswith("data: "):
            continue

        data = text[len("data: "):].strip()

        if data == "[DONE]":
            break

        chunk = json.loads(data)

        choices = chunk.get("choices")

        if not choices:
            continue

        piece = (
            choices[0]
            .get("delta", {})
            .get("content")
            or ""
        )

        if piece and first_token_time is None:
            first_token_time = (
                time.perf_counter() - start
            )

        reply += piece

    total_time = time.perf_counter() - start

    return (
        first_token_time,
        total_time,
        reply.strip(),
    )


# ============================================================
# TTS
# ============================================================

def create_tts():

    config = sherpa_onnx.OfflineTtsConfig()

    config.model.vits.model = str(
        TTS_MODEL_FILE
    )

    config.model.vits.tokens = str(
        TTS_MODEL / "tokens.txt"
    )

    config.model.vits.data_dir = str(
        TTS_MODEL / "espeak-ng-data"
    )

    config.model.vits.noise_scale = 0.667
    config.model.vits.noise_scale_w = 0.8
    config.model.vits.length_scale = 1.0

    config.max_num_sentences = 1

    return sherpa_onnx.OfflineTts(config)


def run_tts(tts, text):

    audio = tts.generate(
        text=text,
        sid=0,
        speed=1.0,
    )

    return audio


# ============================================================
# SAVE GENERATED AUDIO
# ============================================================

def save_audio(audio, filename):

    samples = np.asarray(
        audio.samples,
        dtype=np.float32,
    )

    # Keep samples inside valid audio range
    samples = np.clip(
        samples,
        -1.0,
        1.0,
    )

    # Convert float32 [-1, 1] → signed 16-bit PCM
    samples = (
        samples * 32767
    ).astype(np.int16)

    with wave.open(str(filename), "wb") as f:

        f.setnchannels(1)

        f.setsampwidth(2)

        f.setframerate(
            audio.sample_rate
        )

        f.writeframes(
            samples.tobytes()
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n==========================================")
    print("   HNX26EPS08 BASELINE VOICE PIPELINE")
    print("==========================================\n")

    Path(
        ROOT / "results"
    ).mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("Loading ASR...")

    asr = create_asr()

    print("Loading TTS...")

    tts = create_tts()

    print("Models loaded.\n")

    # --------------------------------------------------------
    # Load input audio
    # --------------------------------------------------------

    sample_rate, audio = load_audio(
        AUDIO_FILE
    )

    audio_duration = (
        len(audio) / sample_rate
    )

    print(
        f"Input audio duration: "
        f"{audio_duration:.3f} s"
    )

    print("------------------------------------------")

    # --------------------------------------------------------
    # ASR
    # --------------------------------------------------------

    start = time.perf_counter()

    transcript = run_asr(
        asr,
        sample_rate,
        audio,
    )

    asr_time = (
        time.perf_counter() - start
    )

    print("\nASR transcript:")
    print(transcript)

    print(
        f"ASR time: {asr_time:.3f} s"
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    first_token, llm_total, reply = run_llm(
        transcript
    )

    print("\nLLM response:")
    print(reply)

    print(
        f"LLM first token: {first_token:.3f} s"
    )

    print(
        f"LLM total:       {llm_total:.3f} s"
    )

    # --------------------------------------------------------
    # TTS
    # --------------------------------------------------------

    tts_start = time.perf_counter()

    audio_result = run_tts(
        tts,
        reply,
    )

    tts_time = (
        time.perf_counter() - tts_start
    )

    print(
        f"\nTTS time: {tts_time:.3f} s"
    )

    # --------------------------------------------------------
    # Save audio
    # --------------------------------------------------------

    save_audio(
        audio_result,
        OUTPUT_FILE,
    )

    print(
        f"Audio saved: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Total pipeline
    # --------------------------------------------------------

    total_pipeline_time = (
        asr_time
        + llm_total
        + tts_time
    )

    print("\n==========================================")
    print("              BASELINE")
    print("==========================================")

    print(
        f"ASR              : {asr_time:.3f} s"
    )

    print(
        f"LLM first token  : {first_token:.3f} s"
    )

    print(
        f"LLM total        : {llm_total:.3f} s"
    )

    print(
        f"TTS total        : {tts_time:.3f} s"
    )

    print("------------------------------------------")

    print(
        f"Pipeline total   : "
        f"{total_pipeline_time:.3f} s"
    )

    print("------------------------------------------")

    print(
        f"Output audio     : {OUTPUT_FILE}"
    )

    print("==========================================\n")

    # --------------------------------------------------------
    # Save benchmark
    # --------------------------------------------------------

    result = {
        "input_audio": str(
            AUDIO_FILE
        ),

        "input_audio_duration_s": (
            audio_duration
        ),

        "transcript": transcript,

        "llm_response": reply,

        "timing": {
            "asr_s": asr_time,
            "llm_first_token_s": first_token,
            "llm_total_s": llm_total,
            "tts_total_s": tts_time,
            "pipeline_total_s": (
                total_pipeline_time
            ),
        },
    }

    output_json = (
        ROOT
        / "results"
        / "baseline_pipeline.json"
    )

    output_json.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved benchmark: {output_json}"
    )


if __name__ == "__main__":
    main()