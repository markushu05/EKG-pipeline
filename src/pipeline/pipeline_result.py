from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PipelineSegmentResult:
    segment_id: int
    quality_score: float
    decision: str
    selected_method: str
    input_samples: np.ndarray
    output_samples: np.ndarray | None
    quality_metrics: Any
    runtime_ms: float
    peak_memory_kb: float


@dataclass
class PipelineResult:
    record_name: str
    sampling_rate: float
    segments: list[PipelineSegmentResult]

    @property
    def total_segments(self):
        return len(self.segments)

    @property
    def kept_segments(self):
        return sum(item.decision == "KEEP" for item in self.segments)

    @property
    def filtered_segments(self):
        return sum(item.decision == "FILTER" for item in self.segments)

    @property
    def rejected_segments(self):
        return sum(item.decision == "REJECT" for item in self.segments)

    @property
    def coverage(self):
        if self.total_segments == 0:
            return 0.0
        return (
            self.kept_segments + self.filtered_segments
        ) / self.total_segments

    @property
    def keep_rate(self):
        if self.total_segments == 0:
            return 0.0
        return self.kept_segments / self.total_segments

    @property
    def filter_rate(self):
        if self.total_segments == 0:
            return 0.0
        return self.filtered_segments / self.total_segments

    @property
    def reject_rate(self):
        if self.total_segments == 0:
            return 0.0
        return self.rejected_segments / self.total_segments

    @property
    def total_runtime_ms(self):
        return float(sum(item.runtime_ms for item in self.segments))

    @property
    def mean_runtime_ms_per_segment(self):
        if self.total_segments == 0:
            return 0.0
        return self.total_runtime_ms / self.total_segments

    @property
    def peak_memory_kb(self):
        if self.total_segments == 0:
            return 0.0
        return float(max(item.peak_memory_kb for item in self.segments))

    def reconstructed_signal(self, rejected_value=np.nan):
        parts = []

        for item in self.segments:
            if item.output_samples is None:
                parts.append(
                    np.full_like(
                        item.input_samples,
                        rejected_value,
                        dtype=float,
                    )
                )
            else:
                parts.append(np.asarray(item.output_samples))

        if not parts:
            return np.array([], dtype=float)

        return np.concatenate(parts)
