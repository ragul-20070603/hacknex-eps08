"""
Time the local llama-server: how long until the first piece of text arrives
(time to first token) and how long until the reply is complete.

Run with the server already started:
    python llm\\test_llm.py

Results and all settings are saved to results\\llm_timing.json
"""
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests

# ---- Settings (all of these go into the results file) ----
URL = "http://127.0.0.1:8080/v1/chat/completions"
SYSTEM = "You are a helpful voice assistant. Reply in one or two short sentences."
QUESTION = "What is the capital of France?"
TEMPERATURE = 0
MAX_TOKENS = 100
RUNS = 5


def run_once():
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    reply = ""
    first_token_time = None

    # Timer starts right before the request is sent.
    start = time.perf_counter()
    response = requests.post(URL, json=body, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if not text.startswith("data: "):
            continue
        data = text[len("data: "):].strip()

        # The stream ends with "data: [DONE]", which is not JSON.
        if data == "[DONE]":
            break

        chunk = json.loads(data)
        choices = chunk.get("choices")
        if not choices:
            continue
        piece = choices[0].get("delta", {}).get("content") or ""

        # First chunk with real text = time to first token.
        if piece and first_token_time is None:
            first_token_time = time.perf_counter() - start
        reply += piece

    total_time = time.perf_counter() - start
    return first_token_time, total_time, reply


if __name__ == "__main__":
    runs = []
    for i in range(RUNS):
        first, total, reply = run_once()
        runs.append({"run": i + 1, "first_token_s": first, "total_s": total, "reply": reply})
        print(f"Run {i + 1}: first token {first:.3f} s, total {total:.3f} s")
        if i == 0:
            print("Reply:", reply)

    # Run 1 is a cold start, so summarise runs 2..N separately.
    warm = [r for r in runs[1:] if r["first_token_s"] is not None]
    if warm:
        print(
            "Warm runs (2..{}): median first token {:.3f} s, median total {:.3f} s".format(
                RUNS,
                statistics.median(r["first_token_s"] for r in warm),
                statistics.median(r["total_s"] for r in warm),
            )
        )

    out = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "url": URL,
            "system_prompt": SYSTEM,
            "question": QUESTION,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "runs": RUNS,
        },
        "runs": runs,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/llm_timing.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved results\\llm_timing.json")