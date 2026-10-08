import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PIPELINE = ROOT / "pipeline" / "concurrent_experiment.py"

RESULTS.mkdir(exist_ok=True)


def run_once():
    process = subprocess.run(
        [sys.executable, str(PIPELINE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
        raise RuntimeError("Benchmark run failed")

    result_file = RESULTS / "concurrent_experiment.json"

    if not result_file.exists():
        raise FileNotFoundError(result_file)

    with open(result_file, "r", encoding="utf-8") as f:
        return json.load(f)


def median(values):
    return statistics.median(values)


def build_summary(runs):
    return {
        "runs": len(runs),
        "median_asr_seconds": median(
            [r["asr_time"] for r in runs]
        ),
        "median_llm_first_token_seconds": median(
            [r["llm_first_token"] for r in runs]
        ),
        "median_llm_total_seconds": median(
            [r["llm_total"] for r in runs]
        ),
        "median_tts_seconds": median(
            [r["tts_generation"] for r in runs]
        ),
        "median_eos_to_audio_seconds": median(
            [r["eos_to_audio_generated"] for r in runs]
        ),
        "median_avg_cpu_percent": median(
            [r["avg_cpu_percent"] for r in runs]
        ),
        "median_peak_cpu_percent": median(
            [r["peak_cpu_percent"] for r in runs]
        ),
        "median_avg_ram_mb": median(
            [r["avg_ram_mb"] for r in runs]
        ),
        "median_peak_ram_mb": median(
            [r["peak_ram_mb"] for r in runs]
        ),
        "median_pipeline_peak_cpu_percent": median(
            [r["peak_pipeline_cpu_percent"] for r in runs]
        ),
        "median_llm_peak_cpu_percent": median(
            [r["peak_llm_cpu_percent"] for r in runs]
        ),
        "median_pipeline_peak_ram_mb": median(
            [r["pipeline_peak_ram_mb"] for r in runs]
        ),
        "median_llm_peak_ram_mb": median(
            [r["llm_peak_ram_mb"] for r in runs]
        ),
    }


def main():
    print("=" * 60)
    print("HNX26EPS08 OPTIMIZED BENCHMARK")
    print("=" * 60)

    runs = []

    for i in range(5):
        print(f"\nRun {i + 1}/5...")

        result = run_once()
        runs.append(result)

        print(
            f"ASR: {result['asr_time']:.3f}s | "
            f"LLM TTFT: {result['llm_first_token']:.3f}s | "
            f"LLM total: {result['llm_total']:.3f}s | "
            f"TTS: {result['tts_generation']:.3f}s | "
            f"EoS→audio: {result['eos_to_audio_generated']:.3f}s"
        )

    summary = build_summary(runs)

    output = {
        "project": "HNX26EPS08",
        "benchmark": "optimized",
        "configuration": {
            "asr": "Sherpa-ONNX Zipformer small English INT8",
            "llm": "Qwen2.5-1.5B-Instruct Q4_K_M GGUF",
            "llm_threads": 2,
            "llm_parallel_slots": 4,
            "llm_max_tokens": 48,
            "llm_response_style": "one short sentence",
            "tts": "Sherpa-ONNX VITS/Piper en_US-amy-low",
            "tts_threads": 2,
            "safety_filter": True,
            "streaming_sentence_to_tts": True,
        },
        "summary": summary,
        "runs": runs,
        "notes": [
            "Five-run median is the primary benchmark statistic.",
            "EoS-to-audio measures generated audio availability.",
            "It does not represent speaker playback latency.",
            "This benchmark uses the safety-test mode.",
            "Microphone/ASR quality is deferred to the final checkpoint.",
        ],
    }

    output_file = RESULTS / "optimized.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("MEDIAN RESULTS")
    print("=" * 60)

    print(f"ASR:              {summary['median_asr_seconds']:.3f}s")
    print(
        f"LLM TTFT:         "
        f"{summary['median_llm_first_token_seconds']:.3f}s"
    )
    print(
        f"LLM total:        "
        f"{summary['median_llm_total_seconds']:.3f}s"
    )
    print(f"TTS:              {summary['median_tts_seconds']:.3f}s")
    print(
        f"EoS → audio:      "
        f"{summary['median_eos_to_audio_seconds']:.3f}s"
    )
    print(
        f"Average CPU:      "
        f"{summary['median_avg_cpu_percent']:.1f}%"
    )
    print(
        f"Peak CPU:         "
        f"{summary['median_peak_cpu_percent']:.1f}%"
    )
    print(
        f"Average RAM:      "
        f"{summary['median_avg_ram_mb']:.1f} MB"
    )
    print(
        f"Peak RAM:         "
        f"{summary['median_peak_ram_mb']:.1f} MB"
    )

    print("\nSaved:")
    print(output_file)


if __name__ == "__main__":
    main()