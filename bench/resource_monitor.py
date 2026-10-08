# ============================================================
# HNX26EPS08 - SYSTEM RESOURCE MONITOR
# ============================================================

import os
import time
import psutil


PIPELINE_PROCESS = psutil.Process(os.getpid())

# Qwen / llama.cpp server
LLM_PID = 18480

try:
    LLM_PROCESS = psutil.Process(LLM_PID)
except psutil.NoSuchProcess:
    LLM_PROCESS = None


def monitor_resources(samples, interval=0.1):

    PIPELINE_PROCESS.cpu_percent(interval=None)

    if LLM_PROCESS is not None:
        try:
            LLM_PROCESS.cpu_percent(interval=None)
        except psutil.NoSuchProcess:
            pass

    start_time = time.perf_counter()

    while True:

        now = time.perf_counter() - start_time

        # Pipeline process
        try:
            pipeline_cpu = PIPELINE_PROCESS.cpu_percent(interval=None)

            pipeline_ram = (
                PIPELINE_PROCESS.memory_info().rss
                / (1024 * 1024)
            )

        except psutil.NoSuchProcess:
            pipeline_cpu = 0
            pipeline_ram = 0

        # LLM server
        llm_cpu = 0
        llm_ram = 0

        if LLM_PROCESS is not None:
            try:
                llm_cpu = LLM_PROCESS.cpu_percent(interval=None)

                llm_ram = (
                    LLM_PROCESS.memory_info().rss
                    / (1024 * 1024)
                )

            except psutil.NoSuchProcess:
                pass

        # Combined
        total_cpu = pipeline_cpu + llm_cpu
        total_ram = pipeline_ram + llm_ram

        samples.append({
            "time": now,

            "pipeline_cpu_percent": pipeline_cpu,
            "pipeline_ram_mb": pipeline_ram,

            "llm_cpu_percent": llm_cpu,
            "llm_ram_mb": llm_ram,

            "total_cpu_percent": total_cpu,
            "total_ram_mb": total_ram,
        })

        time.sleep(interval)


def summarize(samples):

    if not samples:
        return {
            "avg_cpu_percent": 0,
            "peak_cpu_percent": 0,
            "avg_ram_mb": 0,
            "peak_ram_mb": 0,

            "avg_pipeline_cpu_percent": 0,
            "peak_pipeline_cpu_percent": 0,

            "avg_llm_cpu_percent": 0,
            "peak_llm_cpu_percent": 0,

            "pipeline_peak_ram_mb": 0,
            "llm_peak_ram_mb": 0,
        }

    total_cpu = [
        x["total_cpu_percent"]
        for x in samples
    ]

    total_ram = [
        x["total_ram_mb"]
        for x in samples
    ]

    pipeline_cpu = [
        x["pipeline_cpu_percent"]
        for x in samples
    ]

    llm_cpu = [
        x["llm_cpu_percent"]
        for x in samples
    ]

    pipeline_ram = [
        x["pipeline_ram_mb"]
        for x in samples
    ]

    llm_ram = [
        x["llm_ram_mb"]
        for x in samples
    ]

    return {
        "avg_cpu_percent": (
            sum(total_cpu) / len(total_cpu)
        ),

        "peak_cpu_percent": max(total_cpu),

        "avg_ram_mb": (
            sum(total_ram) / len(total_ram)
        ),

        "peak_ram_mb": max(total_ram),

        "avg_pipeline_cpu_percent": (
            sum(pipeline_cpu) / len(pipeline_cpu)
        ),

        "peak_pipeline_cpu_percent": max(
            pipeline_cpu
        ),

        "avg_llm_cpu_percent": (
            sum(llm_cpu) / len(llm_cpu)
        ),

        "peak_llm_cpu_percent": max(
            pipeline_cpu
        ) if False else max(llm_cpu),

        "pipeline_peak_ram_mb": max(
            pipeline_ram
        ),

        "llm_peak_ram_mb": max(
            llm_ram
        ),
    }