import json
import time
from pathlib import Path

import requests
import sherpa_onnx
import numpy as np
import wave


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

AUDIO_FILE = (
    ASR_MODEL
    / "test_wavs"
    / "0.wav"
)

TTS_MODEL = (
    ROOT
    / "llm"
    / "models"
    / "vits-piper-en_US-amy-low"
)

TTS_MODEL_FILE = (
    TTS_MODEL
    / "en_US-amy-low.onnx"
)

LLM_URL = (
    "http://127.0.0.1:8080/v1/chat/completions"
)


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(filename):

    with wave.open(str(filename), "rb") as f:

        sample_rate = f.getframerate()
        channels = f.getnchannels()
        sample_width = f.getsampwidth()
        frames = f.readframes(f.getnframes())

    if sample_width != 2:
        raise ValueError(
            "Expected 16-bit WAV audio"
        )

    audio = (
        np.frombuffer(
            frames,
            dtype=np.int16
        )
        .astype(np.float32)
        / 32768.0
    )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels
        ).mean(axis=1)

    return sample_rate, audio


# ============================================================
# ASR
# ============================================================

def create_asr():

    return sherpa_onnx.OfflineRecognizer.from_transducer(

        encoder=str(
            ASR_MODEL
            / "encoder-epoch-99-avg-1.int8.onnx"
        ),

        decoder=str(
            ASR_MODEL
            / "decoder-epoch-99-avg-1.int8.onnx"
        ),

        joiner=str(
            ASR_MODEL
            / "joiner-epoch-99-avg-1.int8.onnx"
        ),

        tokens=str(
            ASR_MODEL
            / "tokens.txt"
        ),

        num_threads=2,

        decoding_method="greedy_search",
    )


def run_asr(
    recognizer,
    sample_rate,
    audio
):

    stream = recognizer.create_stream()

    stream.accept_waveform(
        sample_rate,
        audio
    )

    recognizer.decode_stream(
        stream
    )

    return stream.result.text.strip()


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

    return sherpa_onnx.OfflineTts(
        config
    )


def run_tts(tts, text):

    start = time.perf_counter()

    audio = tts.generate(
        text=text,
        sid=0,
        speed=1.0,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return audio, elapsed


# ============================================================
# STREAMING LLM
# ============================================================

def stream_llm(question):

    body = {

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
                "content": question,
            },
        ],

        "temperature": 0,

        "max_tokens": 100,

        "stream": True,
    }

    response = requests.post(

        LLM_URL,

        json=body,

        stream=True,

        timeout=120,
    )

    response.raise_for_status()

    full_response = ""

    first_token_time = None

    first_sentence_time = None

    first_sentence = None

    start = time.perf_counter()

    for line in response.iter_lines():

        if not line:
            continue

        text = line.decode(
            "utf-8"
        )

        if not text.startswith(
            "data: "
        ):
            continue

        data = text[
            len("data: "):
        ].strip()

        if data == "[DONE]":
            break

        chunk = json.loads(data)

        choices = chunk.get(
            "choices"
        )

        if not choices:
            continue

        piece = (
            choices[0]
            .get("delta", {})
            .get("content")
            or ""
        )

        if not piece:
            continue

        if first_token_time is None:

            first_token_time = (
                time.perf_counter()
                - start
            )

        full_response += piece

        # ----------------------------------------------------
        # Detect first complete sentence
        # ----------------------------------------------------

        if first_sentence is None:

            for punctuation in [
                ".",
                "?",
                "!",
            ]:

                if punctuation in full_response:

                    first_sentence = (
                        full_response
                        .split(
                            punctuation,
                            1
                        )[0]
                        + punctuation
                    )

                    first_sentence_time = (
                        time.perf_counter()
                        - start
                    )

                    break

    total_time = (
        time.perf_counter()
        - start
    )

    return (
        first_token_time,
        first_sentence_time,
        first_sentence,
        total_time,
        full_response.strip(),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("     STREAMING TTS EXPERIMENT")
    print("==========================================\n")

    print("Loading ASR...")

    asr = create_asr()

    print("Loading TTS...")

    tts = create_tts()

    print("Models loaded.\n")

    # --------------------------------------------------------
    # ASR
    # --------------------------------------------------------

    sample_rate, audio = load_audio(
        AUDIO_FILE
    )

    print(
        f"Input audio: "
        f"{len(audio) / sample_rate:.3f}s"
    )

    asr_start = time.perf_counter()

    transcript = run_asr(
        asr,
        sample_rate,
        audio
    )

    asr_time = (
        time.perf_counter()
        - asr_start
    )

    print("\nASR:")
    print(transcript)

    print(
        f"ASR time: {asr_time:.3f}s"
    )

    # --------------------------------------------------------
    # STREAMING LLM
    # --------------------------------------------------------

    print("\nStarting streaming LLM...")

    (
        first_token,
        first_sentence_time,
        first_sentence,
        llm_total,
        full_response,
    ) = stream_llm(
        transcript
    )

    print(
        f"\nLLM first token: "
        f"{first_token:.3f}s"
    )

    print(
        f"First sentence ready: "
        f"{first_sentence_time:.3f}s"
    )

    print(
        f"First sentence:\n"
        f"{first_sentence}"
    )

    print(
        f"\nFull response:\n"
        f"{full_response}"
    )

    print(
        f"\nLLM total: "
        f"{llm_total:.3f}s"
    )

    # --------------------------------------------------------
    # TTS FIRST SENTENCE
    # --------------------------------------------------------

    print(
        "\nStarting TTS on first sentence..."
    )

    tts_start_global = time.perf_counter()

    audio_result, tts_time = run_tts(
        tts,
        first_sentence
    )

    first_audio_time = (
        time.perf_counter()
        - tts_start_global
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n==========================================")
    print("          STREAMING RESULTS")
    print("==========================================")

    print(
        f"ASR time                 : "
        f"{asr_time:.3f}s"
    )

    print(
        f"LLM first token          : "
        f"{first_token:.3f}s"
    )

    print(
        f"First sentence ready     : "
        f"{first_sentence_time:.3f}s"
    )

    print(
        f"TTS first audio generated: "
        f"{first_audio_time:.3f}s"
    )

    print(
        f"LLM total                : "
        f"{llm_total:.3f}s"
    )

    print(
        f"TTS generation           : "
        f"{tts_time:.3f}s"
    )

    print("==========================================")

    print(
        "\nNOTE:"
    )

    print(
        "This experiment measures "
        "time to generate the first "
        "TTS audio object, not speaker playback."
    )


if __name__ == "__main__":
    main()