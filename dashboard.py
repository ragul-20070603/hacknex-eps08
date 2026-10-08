import streamlit as st
import json
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="HNX26EPS08 Dashboard",
    page_icon="🎙️",
    layout="wide"
)

# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

def load_json(filename):
    path = RESULTS / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

asr_data = load_json("asr_timing.json")
baseline_data = load_json("baseline_pipeline.json")
concurrent_data = load_json("concurrent_experiment.json")


# --------------------------------------------------
# CURRENT VALUES
# --------------------------------------------------

# Current quantized ASR
current_asr = 0.101

# Current LLM warm median
current_llm_first = 0.094
current_llm_total = 0.529

# Current TTS measured from concurrent runs
current_tts = 0.560

# Concurrent EoS -> generated audio
concurrent_eos = 2.128

# Baseline integrated pipeline
baseline_pipeline = 2.209


# --------------------------------------------------
# PREVIOUS / NON-QUANTIZED VALUES
# --------------------------------------------------

# Values measured before moving to the current
# quantized / lightweight configuration.

previous_asr = 0.144
previous_llm_first = 0.148
previous_llm_total = 1.514

# Earlier TTS measurement before the current setup
previous_tts = 0.618


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎙️ HNX26EPS08 — On-Device Conversational Stack")

st.caption(
    "Fully offline • CPU-only • Quantized models • "
    "ASR → LLM → TTS"
)

st.info(
    "📍 Current Stage: Concurrent LLM + TTS validation → "
    "optimizing EoS → first generated audio"
)

# --------------------------------------------------
# SYSTEM STATUS
# --------------------------------------------------

st.subheader("⚙️ System Configuration")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Execution", "Offline")
c2.metric("Hardware", "CPU Only")
c3.metric("ASR", "INT8")
c4.metric("LLM", "Q4_K_M")


# --------------------------------------------------
# ARCHITECTURE
# --------------------------------------------------

st.subheader("🔄 Current Architecture")

st.code(
"""User stops speaking
        ↓
      ASR
        ↓
  LLM token streaming
        ↓
First complete sentence
        ↓
     TTS starts
        ↓
🔊 Audio generation

LLM continues streaming while TTS runs
""",
language="text"
)

st.success(
    "✅ Verified: TTS starts in a background thread while "
    "the LLM is still streaming."
)


# --------------------------------------------------
# CURRENT BENCHMARK
# --------------------------------------------------

st.subheader("📊 Current Measured Performance")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("ASR", f"{current_asr:.3f} s")
c2.metric("LLM First Token", f"{current_llm_first:.3f} s")
c3.metric("LLM Total", f"{current_llm_total:.3f} s")
c4.metric("TTS Generation", f"{current_tts:.3f} s")
c5.metric("EoS → Audio Generated", f"{concurrent_eos:.3f} s")


st.caption(
    "⚠️ EoS → Audio Generated currently means completion of the "
    "first TTS generation, not true speaker/playback first-audio latency."
)


# --------------------------------------------------
# BEFORE QUANTIZATION VS CURRENT
# --------------------------------------------------

st.subheader("📈 Before Quantization vs Current")

st.caption(
    "Controlled component-level comparison of the earlier configuration "
    "against the current lightweight/quantized configuration."
)

comparison = {
    "Metric": [
        "ASR processing",
        "LLM first token",
        "LLM total",
        "TTS generation"
    ],
    "Before Quantization": [
        previous_asr,
        previous_llm_first,
        previous_llm_total,
        previous_tts
    ],
    "Current": [
        current_asr,
        current_llm_first,
        current_llm_total,
        current_tts
    ]
}

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# SPEEDUP
# --------------------------------------------------

st.subheader("⚡ Improvement")

def improvement(old, new):
    return ((old - new) / old) * 100

i1, i2, i3, i4 = st.columns(4)

i1.metric(
    "ASR",
    f"{improvement(previous_asr, current_asr):.1f}% faster"
)

i2.metric(
    "LLM First Token",
    f"{improvement(previous_llm_first, current_llm_first):.1f}% faster"
)

i3.metric(
    "LLM Total",
    f"{improvement(previous_llm_total, current_llm_total):.1f}% faster"
)

i4.metric(
    "TTS",
    f"{improvement(previous_tts, current_tts):.1f}% faster"
)


# --------------------------------------------------
# BASELINE PIPELINE
# --------------------------------------------------

st.subheader("🧪 Integrated Pipeline Baseline")

b1, b2, b3, b4 = st.columns(4)

b1.metric("ASR", "0.144 s")
b2.metric("LLM Total", "1.514 s")
b3.metric("TTS", "0.550 s")
b4.metric("Pipeline Total", "2.209 s")

st.caption(
    "This is the original sequential integrated pipeline. "
    "It is useful as the baseline architecture, but it is not "
    "the final conversational EoS → first-audio metric."
)


# --------------------------------------------------
# CONCURRENT EXPERIMENT
# --------------------------------------------------

st.subheader("🚀 Concurrent LLM + TTS Experiment")

runs = [
    2.687,
    2.128,
    2.078
]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Run 1", "2.687 s")
c2.metric("Run 2", "2.128 s")
c3.metric("Run 3", "2.078 s")
c4.metric("Median", "2.128 s")

st.success(
    "🎯 Concurrency is working: TTS generation overlaps with "
    "continued LLM streaming."
)


# --------------------------------------------------
# COMPLETED WORK
# --------------------------------------------------

st.subheader("✅ Completed")

completed = [
    "Offline CPU-only architecture",
    "Sherpa-ONNX INT8 ASR",
    "ASR benchmark",
    "Local Qwen2.5 1.5B Q4_K_M LLM",
    "LLM streaming",
    "Piper/Sherpa-ONNX offline TTS",
    "Integrated ASR → LLM → TTS baseline",
    "Concurrent LLM + TTS experiment",
    "Background TTS execution",
    "Before-quantization component measurements"
]

for item in completed:
    st.write("✅", item)


# --------------------------------------------------
# CURRENT STEP
# --------------------------------------------------

st.subheader("🔬 Current Step")

st.warning(
"""Optimize EoS → first generated audio

Current strategy:
EoS → ASR → LLM streaming → first complete sentence → TTS

Next:
• Start TTS from earlier usable chunks
• Reduce TTS startup/generation latency
• Measure true first-audio playback latency
• Compare different chunking strategies
"""
)


# --------------------------------------------------
# ROADMAP
# --------------------------------------------------

st.subheader("🗺️ Project Roadmap")

roadmap = [
    ("Baseline pipeline", "✅ Complete"),
    ("Component benchmarking", "✅ Complete"),
    ("Quantized models", "✅ Complete"),
    ("LLM streaming", "✅ Complete"),
    ("Concurrent LLM + TTS", "✅ Complete"),
    ("Earlier TTS chunking", "🔄 Current"),
    ("VAD / end-of-speech", "⏳ Pending"),
    ("CPU/thread optimization", "⏳ Pending"),
    ("RAM measurement", "⏳ Pending"),
    ("Energy measurement", "⏳ Pending"),
    ("Ablation study", "⏳ Pending"),
    ("Graceful degradation", "⏳ Pending"),
    ("Multilingual / smaller-device stretch", "⏳ Pending")
]

for stage, status in roadmap:
    st.write(f"**{stage}** — {status}")


# --------------------------------------------------
# RESEARCH CONTRIBUTION
# --------------------------------------------------

st.subheader("🧠 Research Contribution")

st.write(
"""The project does not claim quantization itself as the novelty.

The contribution is the systematic optimization and evaluation of a
fully offline conversational stack under strict CPU/resource limits.

The study focuses on:

• End-of-speech → first-audio latency
• Cross-stage optimization
• Streaming and computation overlap
• Quantization/resource trade-offs
• TTS chunking strategies
• Ablation of individual optimizations
• Graceful degradation under constrained hardware
"""
)


# --------------------------------------------------
# RESOURCE CONSTRAINTS
# --------------------------------------------------

st.subheader("💻 Resource Constraints")

st.write(
"""Target environment:

• CPU-only execution
• No cloud inference
• No GPU requirement
• Small quantized LLM
• INT8 ASR
• Lightweight offline TTS
• Limited RAM
• Low-end consumer hardware
"""
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "HNX26EPS08 • On-Device Conversational Stack • "
    "Current numbers are experimental benchmark results."
)