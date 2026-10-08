import streamlit as st
import requests
import json
import time
import wave
import threading
import numpy as np
from pathlib import Path
import sherpa_onnx

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

ASR_DIR = (
    ROOT
    / "llm"
    / "models"
    / "asr"
    / "sherpa-onnx-zipformer-small-en-2023-06-26"
)

ASR_ENCODER = ASR_DIR / "encoder-epoch-99-avg-1.int8.onnx"
ASR_DECODER = ASR_DIR / "decoder-epoch-99-avg-1.int8.onnx"
ASR_JOINER = ASR_DIR / "joiner-epoch-99-avg-1.int8.onnx"
ASR_TOKENS = ASR_DIR / "tokens.txt"

TEST_AUDIO = ASR_DIR / "test_wavs" / "0.wav"

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

TTS_MODEL = (
    ROOT
    / "llm"
    / "models"
    / "vits-piper-en_US-amy-low"
    / "en_US-amy-low.onnx"
)

TTS_TOKENS = (
    ROOT
    / "llm"
    / "models"
    / "vits-piper-en_US-amy-low"
    / "tokens.txt"
)

TTS_DATA = (
    ROOT
    / "llm"
    / "models"
    / "vits-piper-en_US-amy-low"
    / "espeak-ng-data"
)

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="HNX26EPS08 Live Demo",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ HNX26EPS08 — LIVE Offline AI Pipeline")

st.caption(
    "CPU-only • Offline • INT8 ASR • Q4 LLM • Offline TTS"
)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(ASR_ENCODER),
        decoder=str(ASR_DECODER),
        joiner=str(ASR_JOINER),
        tokens=str(ASR_TOKENS),
        num_threads=2,
        decoding_method="greedy_search",
    )

    tts_config = sherpa_onnx.OfflineTtsConfig()

    tts_config.model.vits.model = str(TTS_MODEL)
    tts_config.model.vits.tokens = str(TTS_TOKENS)
    tts_config.model.vits.data_dir = str(TTS_DATA)

    tts_config.model.vits.noise_scale = 0.667
    tts_config.model.vits.noise_scale_w = 0.8
    tts_config.model.vits.length_scale = 1.0

    tts_config.max_num_sentences = 1

    tts = sherpa_onnx.OfflineTts(tts_config)

    return recognizer, tts


# ============================================================
# AUDIO SAVE
# ============================================================

def save_audio(audio, filename):

    samples = np.asarray(
        audio.samples,
        dtype=np.float32
    )

    samples = np.clip(samples, -1.0, 1.0)

    samples = (
        samples * 32767
    ).astype(np.int16)

    with wave.open(str(filename), "wb") as f:

        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(audio.sample_rate)

        f.writeframes(
            samples.tobytes()
        )


# ============================================================
# UI HELPERS
# ============================================================

def stage_box(container, title, status, timing=None):

    if status == "waiting":
        container.info(f"⚪ {title} — Waiting")

    elif status == "running":
        container.warning(f"🟡 {title} — Running...")

    elif status == "done":

        if timing is not None:
            container.success(
                f"🟢 {title} — Complete  |  ⏱ {timing:.3f} s"
            )
        else:
            container.success(
                f"🟢 {title} — Complete"
            )


# ============================================================
# HEADER STATUS
# ============================================================

st.subheader("Pipeline")

p1, p2, p3, p4 = st.columns(4)

p1.metric("ASR", "INT8")
p2.metric("LLM", "Q4_K_M")
p3.metric("TTS", "Piper VITS")
p4.metric("Execution", "OFFLINE")


# ============================================================
# INPUT
# ============================================================

st.subheader("🎤 Input")

st.audio(
    str(TEST_AUDIO),
    format="audio/wav"
)

run = st.button(
    "▶ RUN LIVE PIPELINE",
    type="primary",
    use_container_width=True
)


# ============================================================
# PLACEHOLDERS
# ============================================================

asr_area = st.empty()
llm_area = st.empty()
tts_area = st.empty()

transcript_area = st.empty()
llm_output_area = st.empty()

timeline_area = st.empty()

m1, m2, m3, m4, m5 = st.columns(5)

metric_asr = m1.empty()
metric_first = m2.empty()
metric_llm = m3.empty()
metric_tts = m4.empty()
metric_eos = m5.empty()


# ============================================================
# RUN
# ============================================================

if run:

    recognizer, tts = load_models()

    # --------------------------------------------------------
    # INITIAL
    # --------------------------------------------------------

    stage_box(
        asr_area,
        "ASR",
        "running"
    )

    stage_box(
        llm_area,
        "LLM",
        "waiting"
    )

    stage_box(
        tts_area,
        "TTS",
        "waiting"
    )

    # --------------------------------------------------------
    # ASR
    # --------------------------------------------------------

    with wave.open(
        str(TEST_AUDIO),
        "rb"
    ) as f:

        sample_rate = f.getframerate()

        frames = f.readframes(
            f.getnframes()
        )

    samples = (
        np.frombuffer(
            frames,
            dtype=np.int16
        ).astype(np.float32)
        / 32768.0
    )

    stream = recognizer.create_stream()

    stream.accept_waveform(
        sample_rate,
        samples
    )

    asr_start = time.perf_counter()

    recognizer.decode_stream(stream)

    asr_end = time.perf_counter()

    transcript = stream.result.text.strip()

    asr_time = asr_end - asr_start

    transcript_area.markdown(
        f"### 📝 ASR Output\n`{transcript}`"
    )

    metric_asr.metric(
        "ASR",
        f"{asr_time:.3f} s"
    )

    stage_box(
        asr_area,
        "ASR",
        "done",
        asr_time
    )

    # --------------------------------------------------------
    # EoS
    # --------------------------------------------------------

    eos_time = time.perf_counter()

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    stage_box(
        llm_area,
        "LLM",
        "running"
    )

    messages = [
        {
            "role": "system",
            "content":
                "You are a helpful voice assistant. "
                "Reply in one or two short sentences."
        },
        {
            "role": "user",
            "content": transcript
        }
    ]

    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": 100,
        "stream": True
    }

    llm_start = time.perf_counter()

    first_token_time = None

    full_response = ""

    first_sentence = ""

    tts_started = False
    tts_finished = False

    tts_start_time = None
    tts_end_time = None

    tts_audio = None

    # --------------------------------------------------------
    # TTS FUNCTION
    # --------------------------------------------------------

    def run_tts(text):

        nonlocal_dummy = None

        global tts_audio
        global tts_started
        global tts_finished
        global tts_start_time
        global tts_end_time

        tts_started = True

        tts_start_time = time.perf_counter()

        audio = tts.generate(
            text=text,
            sid=0,
            speed=1.0
        )

        tts_end_time = time.perf_counter()

        tts_audio = audio

        tts_finished = True

    # --------------------------------------------------------
    # STREAM LLM
    # --------------------------------------------------------

    with requests.post(
        LLM_URL,
        json=payload,
        stream=True,
        timeout=120
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines():

            if not line:
                continue

            decoded = line.decode(
                "utf-8"
            )

            if not decoded.startswith(
                "data:"
            ):
                continue

            data = decoded[
                5:
            ].strip()

            if data == "[DONE]":
                break

            try:

                obj = json.loads(
                    data
                )

            except Exception:
                continue

            choices = obj.get(
                "choices",
                []
            )

            if not choices:
                continue

            delta = choices[0].get(
                "delta",
                {}
            )

            token = delta.get(
                "content",
                ""
            )

            if not token:
                continue

            # First token
            if first_token_time is None:

                first_token_time = (
                    time.perf_counter()
                    - llm_start
                )

                metric_first.metric(
                    "LLM First Token",
                    f"{first_token_time:.3f} s"
                )

            full_response += token

            # Live token output
            llm_output_area.markdown(
                f"### 🤖 LLM Streaming\n"
                f"**{full_response}▌**"
            )

            # ------------------------------------------------
            # FIRST COMPLETE SENTENCE
            # ------------------------------------------------

            if (
                not tts_started
                and any(
                    p in full_response
                    for p in [".", "?", "!"]
                )
            ):

                sentence_end = max(
                    full_response.rfind("."),
                    full_response.rfind("?"),
                    full_response.rfind("!")
                )

                first_sentence = (
                    full_response[
                        :sentence_end + 1
                    ].strip()
                )

                # TTS starts immediately
                tts_thread = threading.Thread(
                    target=run_tts,
                    args=(first_sentence,),
                    daemon=True
                )

                tts_thread.start()

                tts_area.warning(
                    "🟡 TTS STARTED — "
                    "while LLM is still streaming"
                )

                st.toast(
                    "🔊 TTS started concurrently!",
                    icon="🎙️"
                )

    # --------------------------------------------------------
    # LLM COMPLETE
    # --------------------------------------------------------

    llm_end = time.perf_counter()

    llm_total = (
        llm_end - llm_start
    )

    # Wait for TTS
    if tts_started:

        tts_thread.join()

    # --------------------------------------------------------
    # FINAL UI
    # --------------------------------------------------------

    llm_output_area.markdown(
        f"### 🤖 Final LLM Output\n"
        f"**{full_response}**"
    )

    metric_llm.metric(
        "LLM Total",
        f"{llm_total:.3f} s"
    )

    stage_box(
        llm_area,
        "LLM",
        "done",
        llm_total
    )

    # --------------------------------------------------------
    # TTS
    # --------------------------------------------------------

    if tts_finished:

        tts_time = (
            tts_end_time
            - tts_start_time
        )

        eos_to_audio = (
            tts_end_time
            - eos_time
        )

        output_file = (
            RESULTS
            / "live_demo_output.wav"
        )

        save_audio(
            tts_audio,
            output_file
        )

        metric_tts.metric(
            "TTS Generation",
            f"{tts_time:.3f} s"
        )

        metric_eos.metric(
            "EoS → Audio",
            f"{eos_to_audio:.3f} s"
        )

        tts_area.success(
            "🟢 TTS Complete\n\n"
            f"Generated from: `{first_sentence}`"
        )

        st.audio(
            str(output_file),
            format="audio/wav"
        )

        st.success(
            "🔊 Audio generated successfully."
        )

    # --------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------

    st.subheader(
        "⏱️ Live Execution Timeline"
    )

    st.code(
f"""
EoS
│
├── ASR             {asr_time:.3f} s
│
├── LLM first token {first_token_time:.3f} s
│
├── LLM streaming
│   ├── "{first_sentence}"
│   │
│   └──────► TTS START
│             │
│             └── {tts_time:.3f} s
│
└── LLM total       {llm_total:.3f} s

EoS → audio generated
= {eos_to_audio:.3f} s

🟢 TTS overlapped with LLM streaming
""",
        language="text"
    )

    st.success(
        "🎯 LIVE RUN COMPLETE"
    )