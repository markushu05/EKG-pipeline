from dataclasses import dataclass
import time
import tracemalloc


@dataclass
class EmbeddedMetrics:
    runtime_ms: float
    peak_memory_kb: float


def start_measurement():
    """
    Startet Runtime- und RAM-Messung.
    """

    tracemalloc.start()
    start_time = time.perf_counter()

    return start_time


def stop_measurement(start_time):
    """
    Beendet Runtime- und RAM-Messung.
    """

    runtime_ms = (
        time.perf_counter() - start_time
    ) * 1000

    _, peak_memory = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    peak_memory_kb = peak_memory / 1024

    return EmbeddedMetrics(
        runtime_ms=float(runtime_ms),
        peak_memory_kb=float(peak_memory_kb)
    )