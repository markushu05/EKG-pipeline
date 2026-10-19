from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.embedded import start_measurement, stop_measurement
from src.metrics.signal_quality_metrics import create_signal_quality_metrics
from src.segmentation.segmenter import Segmenter
from src.pipeline.decision_policy import DecisionPolicy, SegmentDecision
from src.pipeline.pipeline_result import PipelineResult, PipelineSegmentResult


class EmpiricalQualityScorer:
    SNR_COLUMN = "quality_estimated_snr"
    CORRELATION_COLUMN = "quality_beat_template_correlation"

    def __init__(
        self,
        snr_reference,
        correlation_reference,
        snr_weight=0.70,
        correlation_weight=0.30,
    ):
        self.snr_reference = self._clean_reference(snr_reference)
        self.correlation_reference = self._clean_reference(
            correlation_reference
        )
        self.snr_weight = float(snr_weight)
        self.correlation_weight = float(correlation_weight)

        if self.snr_reference.size == 0:
            raise ValueError("Keine gueltigen SNR-Referenzwerte vorhanden.")

        if self.correlation_reference.size == 0:
            raise ValueError(
                "Keine gueltigen Beat-Correlation-Referenzwerte vorhanden."
            )

    @classmethod
    def from_analysis_file(
        cls,
        analysis_file,
        snr_weight=0.70,
        correlation_weight=0.30,
    ):
        df = pd.read_excel(
            Path(analysis_file),
            sheet_name="improvements",
            engine="openpyxl",
        )

        required = [cls.SNR_COLUMN, cls.CORRELATION_COLUMN]
        missing = [column for column in required if column not in df.columns]

        if missing:
            raise ValueError(
                "In der globalen Analysedatei fehlen: "
                + ", ".join(missing)
            )

        return cls(
            snr_reference=df[cls.SNR_COLUMN],
            correlation_reference=df[cls.CORRELATION_COLUMN],
            snr_weight=snr_weight,
            correlation_weight=correlation_weight,
        )

    def calculate(self, quality_metrics):
        estimated_snr = self._metric_value(
            quality_metrics,
            ["estimated_snr", "quality_estimated_snr"],
        )
        beat_correlation = self._metric_value(
            quality_metrics,
            [
                "beat_template_correlation",
                "quality_beat_template_correlation",
            ],
        )

        values = []
        weights = []

        if estimated_snr is not None and np.isfinite(estimated_snr):
            values.append(
                self._higher_is_better_badness(
                    estimated_snr,
                    self.snr_reference,
                )
            )
            weights.append(self.snr_weight)

        if beat_correlation is not None and np.isfinite(beat_correlation):
            values.append(
                self._higher_is_better_badness(
                    beat_correlation,
                    self.correlation_reference,
                )
            )
            weights.append(self.correlation_weight)

        if not weights or sum(weights) <= 0.0:
            return np.nan

        return float(np.average(values, weights=weights))

    @staticmethod
    def _clean_reference(values):
        array = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
        return np.sort(array[np.isfinite(array)])

    @staticmethod
    def _higher_is_better_badness(value, reference):
        position = np.searchsorted(reference, value, side="right")
        return float(1.0 - position / reference.size)

    @staticmethod
    def _metric_value(metrics, names):
        if isinstance(metrics, dict):
            for name in names:
                if name in metrics:
                    return metrics[name]
            return None

        for name in names:
            if hasattr(metrics, name):
                return getattr(metrics, name)

        return None


class AdaptivePipeline:
    """Segmentiert einen Record und entscheidet KEEP, FILTER oder REJECT.

    runtime_ms und peak_memory_kb umfassen pro Segment:
        Peak Detection + SQI + Score + Entscheidung + optional Bandpass.
    """

    def __init__(
        self,
        peak_detector,
        denoiser,
        quality_scorer,
        decision_policy=None,
        segment_length_s=10,
    ):
        self.peak_detector = peak_detector
        self.denoiser = denoiser
        self.quality_scorer = quality_scorer
        self.decision_policy = decision_policy or DecisionPolicy()
        self.segment_length_s = segment_length_s

    def run(self, signal, sampling_rate, record_name="unknown"):
        segment_objects = Segmenter.split(
            signal=signal,
            sampling_rate=sampling_rate,
            segment_length_s=self.segment_length_s,
        )

        results = []

        for segment_object in segment_objects:
            input_samples = np.asarray(segment_object.samples)
            measurement_start = start_measurement()

            waves = self.peak_detector.detect_waves(input_samples)
            quality_metrics = create_signal_quality_metrics(
                input_samples,
                waves.r_peaks,
                sampling_rate,
            )
            quality_score = self.quality_scorer.calculate(quality_metrics)
            decision = self.decision_policy.decide(quality_score)

            if decision == SegmentDecision.KEEP:
                output_samples = input_samples.copy()
                selected_method = "RAW"
            elif decision == SegmentDecision.FILTER:
                output_samples = self._apply_denoiser(
                    input_samples,
                    sampling_rate,
                )
                selected_method = "Bandpass"
            else:
                output_samples = None
                selected_method = "REJECT"

            embedded = stop_measurement(measurement_start)

            results.append(
                PipelineSegmentResult(
                    segment_id=segment_object.segment_id,
                    quality_score=quality_score,
                    decision=decision.value,
                    selected_method=selected_method,
                    input_samples=input_samples,
                    output_samples=output_samples,
                    quality_metrics=quality_metrics,
                    runtime_ms=self._embedded_value(
                        embedded,
                        "runtime_ms",
                    ),
                    peak_memory_kb=self._embedded_value(
                        embedded,
                        "peak_memory_kb",
                    ),
                )
            )

        return PipelineResult(
            record_name=record_name,
            sampling_rate=sampling_rate,
            segments=results,
        )

    def _apply_denoiser(self, signal, sampling_rate):
        if hasattr(self.denoiser, "denoise"):
            try:
                return self.denoiser.denoise(signal, sampling_rate)
            except TypeError:
                return self.denoiser.denoise(signal)

        if hasattr(self.denoiser, "filter"):
            return self.denoiser.filter(signal, sampling_rate)

        raise ValueError(
            f"Unbekannter Denoiser-Typ: {type(self.denoiser)}"
        )

    @staticmethod
    def _embedded_value(embedded_metrics, name):
        if isinstance(embedded_metrics, dict):
            value = embedded_metrics.get(name, 0.0)
        else:
            value = getattr(embedded_metrics, name, 0.0)

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
