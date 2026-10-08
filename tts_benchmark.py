import time
import json
import os
import statistics
import sherpa_onnx

MODEL_DIR = r"llm\models\vits-piper-en_US-amy-low"
MODEL = MODEL_DIR + r"\en_US-amy-low.onnx"
TOKENS = MODEL_DIR + r"\tokens.txt"
DATA_DIR = MODEL_DIR + r"\espeak-ng-data"

RESULTS_DIR = r"results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "tts_timing.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading TTS model...")

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=MODEL,
            tokens=TOKENS,
            data_dir=DATA_DIR,
        )
    ),
    max_num_sentences=1,
)

tts = sherpa_onnx.OfflineTts(tts_config)

print("TTS model loaded.\n")

text = (
    "Hello. This is a low latency voice AI system "
    "running on a resource constrained computer."
)

print("Test text:")
print(text)
print()

times = []
durations = []

for i in range(5):
    print(f"Run {i + 1}/5...")

    start = time.perf_counter()

    audio = tts.generate(
        text=text,
        sid=0,
        speed=1.0,
    )

    end = time.perf_counter()

    generation_time = end - start

    sample_rate = audio.sample_rate
    audio_duration = len(audio.samples) / sample_rate

    rtf = generation_time / audio_duration

    times.append(generation_time)
    durations.append(audio_duration)

    print(
        f"  Generation: {generation_time:.3f}s"
        f" | Audio: {audio_duration:.3f}s"
        f" | RTF: {rtf:.3f}"
    )

median_time = statistics.median(times)
median_duration = statistics.median(durations)
median_rtf = median_time / median_duration

results = {
    "component": "TTS",
    "model": "vits-piper-en_US-amy-low",
    "text": text,
    "runs": [
        {
            "run": i + 1,
            "generation_time_seconds": round(times[i], 4),
            "audio_duration_seconds": round(durations[i], 4),
            "rtf": round(times[i] / durations[i], 4),
        }
        for i in range(5)
    ],
    "median_generation_time_seconds": round(median_time, 4),
    "median_audio_duration_seconds": round(median_duration, 4),
    "median_rtf": round(median_rtf, 4),
}

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)

print("\n==============================")
print("TTS BENCHMARK RESULTS")
print("==============================")

print("Runs:", len(times))

print("\nGeneration times:")
for i, value in enumerate(times):
    print(f"  Run {i + 1}: {value:.3f}s")

print(f"\nMedian generation time: {median_time:.3f}s")
print(f"Median audio duration:  {median_duration:.3f}s")
print(f"Median RTF:             {median_rtf:.3f}")

print("==============================")
print(f"Saved to: {RESULTS_FILE}")
print("==============================")