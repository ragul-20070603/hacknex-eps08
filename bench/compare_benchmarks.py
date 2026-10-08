import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Previously validated 5-run baseline
baseline = {
    "median_asr_seconds": 0.133,
    "median_llm_first_token_seconds": 0.065,
    "median_llm_total_seconds": 0.795,
    "median_tts_seconds": 0.476,
    "median_eos_to_audio_seconds": 1.209,
    "median_peak_cpu_percent": 545.0,
    "median_peak_ram_mb": 1235.8,
}

with open(RESULTS / "optimized.json", "r", encoding="utf-8") as f:
    optimized = json.load(f)["summary"]


def improvement(old, new):
    return ((old - new) / old) * 100


comparison = {
    "project": "HNX26EPS08",

    "baseline": baseline,

    "optimized": optimized,

    "improvement_percent": {
        "asr": improvement(
            baseline["median_asr_seconds"],
            optimized["median_asr_seconds"],
        ),

        "llm_ttft": improvement(
            baseline["median_llm_first_token_seconds"],
            optimized["median_llm_first_token_seconds"],
        ),

        "llm_total": improvement(
            baseline["median_llm_total_seconds"],
            optimized["median_llm_total_seconds"],
        ),

        "tts": improvement(
            baseline["median_tts_seconds"],
            optimized["median_tts_seconds"],
        ),

        "eos_to_audio": improvement(
            baseline["median_eos_to_audio_seconds"],
            optimized["median_eos_to_audio_seconds"],
        ),

        "peak_cpu": improvement(
            baseline["median_peak_cpu_percent"],
            optimized["median_peak_cpu_percent"],
        ),

        "peak_ram": improvement(
            baseline["median_peak_ram_mb"],
            optimized["median_peak_ram_mb"],
        ),
    },
}

output = RESULTS / "comparison.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)

print("=" * 60)
print("HNX26EPS08 BASELINE vs OPTIMIZED")
print("=" * 60)

print(
    f"EoS → audio: "
    f"{baseline['median_eos_to_audio_seconds']:.3f}s → "
    f"{optimized['median_eos_to_audio_seconds']:.3f}s "
    f"({comparison['improvement_percent']['eos_to_audio']:.1f}% faster)"
)

print(
    f"LLM total: "
    f"{baseline['median_llm_total_seconds']:.3f}s → "
    f"{optimized['median_llm_total_seconds']:.3f}s "
    f"({comparison['improvement_percent']['llm_total']:.1f}% faster)"
)

print(
    f"TTS: "
    f"{baseline['median_tts_seconds']:.3f}s → "
    f"{optimized['median_tts_seconds']:.3f}s "
    f"({comparison['improvement_percent']['tts']:.1f}% faster)"
)

print(
    f"Peak CPU: "
    f"{baseline['median_peak_cpu_percent']:.1f}% → "
    f"{optimized['median_peak_cpu_percent']:.1f}%"
)

print(
    f"Peak RAM: "
    f"{baseline['median_peak_ram_mb']:.1f} MB → "
    f"{optimized['median_peak_ram_mb']:.1f} MB"
)

print("\nSaved:")
print(output)