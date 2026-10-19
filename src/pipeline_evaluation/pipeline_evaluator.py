from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd

from src.metrics.medical_metrics import create_medical_metrics
from src.metrics.morphology_metrics import create_morphology_metrics
from src.metrics.fidelity import create_fidelity_metrics
from src.segmentation.segmenter import Segmenter


from src.metrics.embedded import (
    start_measurement,
    stop_measurement,
)


class PipelineEvaluator:
    """Bewertet genau einen Record.

    Verglichen werden RAW, Always Bandpass, Selective und Oracle.
    Intern wird segmentweise gerechnet. Nach außen wird nur eine kompakte
    Zusammenfassung pro Strategie zurueckgegeben.
    """

    ERROR_METRICS = [
        "medical_hr_error_bpm",
        "medical_rr_error_ms",
        "morphology_qrs_amplitude_error",
        "morphology_qrs_duration_error",
        "morphology_st_segment_error",
        "morphology_p_amplitude_error",
        "morphology_t_amplitude_error",
        "morphology_qt_interval_error",
        "fidelity_rmse",
        "fidelity_prd",
    ]

    F1_METRICS = [
        "medical_sensitivity",
        "medical_precision",
        "medical_f1_score",
    ]

    CORRELATION_METRICS = [
        "morphology_beat_correlation",
        "fidelity_full_segment_corr",
    ]

    SNR_METRICS = ["fidelity_snr_after_db"]
    EPSILON = 1e-12

    def __init__(
        self,
        adaptive_pipeline,
        peak_detector,
        bandpass_filter,
        segment_length_s=10,
    ):
        self.adaptive_pipeline = adaptive_pipeline
        self.peak_detector = peak_detector
        self.bandpass_filter = bandpass_filter
        self.segment_length_s = segment_length_s

    def evaluate(
        self,
        clean_signal,
        noisy_signal,
        sampling_rate,
        record_name,
    ):
        clean_segments = Segmenter.split(
            signal=clean_signal,
            sampling_rate=sampling_rate,
            segment_length_s=self.segment_length_s,
        )
        noisy_segments = Segmenter.split(
            signal=noisy_signal,
            sampling_rate=sampling_rate,
            segment_length_s=self.segment_length_s,
        )

        pipeline_result = self.adaptive_pipeline.run(
            signal=noisy_signal,
            sampling_rate=sampling_rate,
            record_name=record_name,
        )

        strategy_improvements = {
            "RAW": [],
            "Always Bandpass": [],
            "Selective": [],
            "Oracle": [],
        }

        selective_used = 0
        total = min(
            len(clean_segments),
            len(noisy_segments),
            len(pipeline_result.segments),
        )

        bandpass_runtime_values = []
        bandpass_memory_values = []

        for index in range(total):
            clean = np.asarray(clean_segments[index].samples)
            raw = np.asarray(noisy_segments[index].samples)
            reference_waves = self.peak_detector.detect_waves(clean)

            bandpass_start = start_measurement()

            bandpass = self._apply_bandpass(
                raw,
                sampling_rate,
            )

            bandpass_embedded = stop_measurement(
                bandpass_start
            )

            bandpass_runtime_ms = self._embedded_value(
                bandpass_embedded,
                "runtime_ms",
            )

            bandpass_peak_memory_kb = self._embedded_value(
                bandpass_embedded,
                "peak_memory_kb",
            )

            bandpass_runtime_values.append(
                bandpass_runtime_ms
            )

            bandpass_memory_values.append(
                bandpass_peak_memory_kb
            )

            raw_metrics = self._calculate_metrics(
                clean,
                raw,
                raw,
                reference_waves,
                sampling_rate,
            )
            bandpass_metrics = self._calculate_metrics(
                clean,
                raw,
                bandpass,
                reference_waves,
                sampling_rate,
            )
            reference_metrics = self._calculate_metrics(
                clean,
                clean,
                clean,
                reference_waves,
                sampling_rate,
            )

            raw_improvements = self._zero_improvements(raw_metrics)
            bandpass_improvements = self._improvements(
                raw_metrics,
                bandpass_metrics,
                reference_metrics,
            )

            strategy_improvements["RAW"].append(raw_improvements)
            strategy_improvements["Always Bandpass"].append(
                bandpass_improvements
            )

            pipeline_segment = pipeline_result.segments[index]

            if pipeline_segment.selected_method == "RAW":
                strategy_improvements["Selective"].append(
                    raw_improvements
                )
                selective_used += 1
            elif pipeline_segment.selected_method == "Bandpass":
                strategy_improvements["Selective"].append(
                    bandpass_improvements
                )
                selective_used += 1

            raw_overall = self._group_scores(raw_improvements)[
                "overall_score"
            ]
            bandpass_overall = self._group_scores(
                bandpass_improvements
            )["overall_score"]

            if bandpass_overall > raw_overall:
                strategy_improvements["Oracle"].append(
                    bandpass_improvements
                )
            else:
                strategy_improvements["Oracle"].append(
                    raw_improvements
                )

        always_total_runtime_ms = sum(
            bandpass_runtime_values
        )

        always_mean_runtime_ms = (
            always_total_runtime_ms / total
            if total > 0
            else 0.0
        )

        always_peak_memory_kb = (
            max(bandpass_memory_values)
            if bandpass_memory_values
            else 0.0
        )

        selective_total_runtime_ms = (
            pipeline_result.total_runtime_ms
        )

        selective_mean_runtime_ms = (
            pipeline_result.mean_runtime_ms_per_segment
        )

        selective_peak_memory_kb = (
            pipeline_result.peak_memory_kb
        )

        rows = []

        for strategy, segment_values in strategy_improvements.items():
            segment_scores = [
                self._group_scores(values)
                for values in segment_values
            ]
            score_df = pd.DataFrame(segment_scores)

            n_used = len(segment_values)
            coverage = n_used / total if total else 0.0

            if strategy == "RAW":
                total_runtime_ms = 0.0
                mean_runtime_ms = 0.0
                peak_memory_kb = 0.0
                resource_value_type = "baseline_no_denoising"

            elif strategy == "Always Bandpass":
                total_runtime_ms = always_total_runtime_ms
                mean_runtime_ms = always_mean_runtime_ms
                peak_memory_kb = always_peak_memory_kb
                resource_value_type = "measured"

            elif strategy == "Selective":
                total_runtime_ms = selective_total_runtime_ms
                mean_runtime_ms = selective_mean_runtime_ms
                peak_memory_kb = selective_peak_memory_kb
                resource_value_type = "measured"

            else:
                total_runtime_ms = np.nan
                mean_runtime_ms = np.nan
                peak_memory_kb = np.nan
                resource_value_type = "not_runtime_capable"

            row = {
                "record": record_name,
                "strategy": strategy,
                "n_total_segments": total,
                "n_used_segments": n_used,
                "coverage": coverage,
                "keep_rate": (
                    pipeline_result.keep_rate
                    if strategy == "Selective"
                    else (1.0 if strategy in ["RAW", "Oracle"] else 0.0)
                ),
                "filter_rate": (
                    pipeline_result.filter_rate
                    if strategy == "Selective"
                    else (1.0 if strategy == "Always Bandpass" else np.nan)
                ),
                "reject_rate": (
                    pipeline_result.reject_rate
                    if strategy == "Selective"
                    else 0.0
                ),
                "total_runtime_ms": total_runtime_ms,
                "mean_runtime_ms_per_segment": mean_runtime_ms,
                "peak_memory_kb": peak_memory_kb,
                "resource_value_type": resource_value_type,
            }

            for column in [
                "medical_score",
                "morphology_score",
                "fidelity_score",
                "overall_score",
                "worst_improvement",
                "negative_metric_count",
                "strong_negative_metric_count",
            ]:
                row[column] = (
                    score_df[column].mean()
                    if column in score_df.columns and not score_df.empty
                    else np.nan
                )

            rows.append(row)

        result_df = pd.DataFrame(
            rows
        )

        always_row = result_df[
            result_df["strategy"] == "Always Bandpass"
        ]

        if not always_row.empty:
            runtime_reference = always_row.iloc[0][
                "total_runtime_ms"
            ]

            memory_reference = always_row.iloc[0][
                "peak_memory_kb"
            ]

            result_df[
                "runtime_percent_of_always"
            ] = np.where(
                runtime_reference > 0.0,
                (
                    result_df["total_runtime_ms"]
                    / runtime_reference
                    * 100.0
                ),
                np.nan,
            )

            result_df[
                "runtime_saving_percent"
            ] = np.where(
                result_df[
                    "runtime_percent_of_always"
                ].notna(),
                (
                    100.0
                    - result_df[
                        "runtime_percent_of_always"
                    ]
                ),
                np.nan,
            )

            result_df[
                "memory_percent_of_always"
            ] = np.where(
                memory_reference > 0.0,
                (
                    result_df["peak_memory_kb"]
                    / memory_reference
                    * 100.0
                ),
                np.nan,
            )

            result_df[
                "memory_saving_percent"
            ] = np.where(
                result_df[
                    "memory_percent_of_always"
                ].notna(),
                (
                    100.0
                    - result_df[
                        "memory_percent_of_always"
                    ]
                ),
                np.nan,
            )

        else:
            result_df[
                "runtime_percent_of_always"
            ] = np.nan

            result_df[
                "runtime_saving_percent"
            ] = np.nan

            result_df[
                "memory_percent_of_always"
            ] = np.nan

            result_df[
                "memory_saving_percent"
            ] = np.nan

        return result_df, pipeline_result

    def _calculate_metrics(
        self,
        clean_signal,
        noisy_signal,
        processed_signal,
        reference_waves,
        sampling_rate,
    ):
        test_waves = self.peak_detector.detect_waves(processed_signal)

        medical = create_medical_metrics(
            reference_waves.r_peaks,
            test_waves.r_peaks,
            sampling_rate,
        )
        morphology = create_morphology_metrics(
            clean_signal,
            processed_signal,
            reference_waves,
            test_waves,
            sampling_rate,
        )
        fidelity = create_fidelity_metrics(
            clean_signal=clean_signal,
            noisy_signal=noisy_signal,
            denoised_signal=processed_signal,
        )

        metrics = {}
        metrics.update(self._flatten_metrics(medical, "medical"))
        metrics.update(self._flatten_metrics(morphology, "morphology"))
        metrics.update(self._flatten_metrics(fidelity, "fidelity"))
        return metrics

    def _improvements(self, raw, filtered, reference):
        result = {}

        for metric in self.ERROR_METRICS:
            result[f"improvement_{metric}"] = self._relative_error(
                raw.get(metric),
                filtered.get(metric),
                reference.get(metric),
            )

        for metric in self.F1_METRICS + self.CORRELATION_METRICS:
            result[f"improvement_{metric}"] = self._bounded_improvement(
                raw.get(metric),
                filtered.get(metric),
            )

        for metric in self.SNR_METRICS:
            result[f"improvement_{metric}"] = self._snr_improvement(
                raw.get(metric),
                filtered.get(metric),
            )

        return result

    @staticmethod
    def _zero_improvements(metrics):
        return {
            f"improvement_{metric}": 0.0
            for metric in metrics
        }

    def _group_scores(self, improvements):
        medical = [
            value
            for key, value in improvements.items()
            if "improvement_medical_" in key and self._valid(value)
        ]
        morphology = [
            value
            for key, value in improvements.items()
            if "improvement_morphology_" in key and self._valid(value)
        ]
        fidelity = [
            value
            for key, value in improvements.items()
            if "improvement_fidelity_" in key and self._valid(value)
        ]
        all_values = medical + morphology + fidelity

        medical_score = np.mean(medical) if medical else np.nan
        morphology_score = np.mean(morphology) if morphology else np.nan
        fidelity_score = np.mean(fidelity) if fidelity else np.nan
        group_values = [
            value
            for value in [
                medical_score,
                morphology_score,
                fidelity_score,
            ]
            if self._valid(value)
        ]

        return {
            "medical_score": medical_score,
            "morphology_score": morphology_score,
            "fidelity_score": fidelity_score,
            "overall_score": (
                np.mean(group_values) if group_values else np.nan
            ),
            "worst_improvement": (
                min(all_values) if all_values else np.nan
            ),
            "negative_metric_count": sum(
                value < -0.02 for value in all_values
            ),
            "strong_negative_metric_count": sum(
                value < -0.30 for value in all_values
            ),
        }

    def _relative_error(self, raw, filtered, reference):
        if not all(self._valid(value) for value in [raw, filtered, reference]):
            return np.nan
        raw_distance = abs(raw - reference)
        if raw_distance < self.EPSILON:
            return 0.0
        return (raw_distance - abs(filtered - reference)) / raw_distance

    def _bounded_improvement(self, raw, filtered):
        if not all(self._valid(value) for value in [raw, filtered]):
            return np.nan
        raw_error = 1.0 - raw
        if raw_error < self.EPSILON:
            return 0.0
        return (raw_error - (1.0 - filtered)) / raw_error

    def _snr_improvement(self, raw, filtered):
        if not all(self._valid(value) for value in [raw, filtered]):
            return np.nan
        if abs(raw) < self.EPSILON:
            return 0.0
        return (filtered - raw) / abs(raw)

    @staticmethod
    def _flatten_metrics(metrics, prefix):
        if is_dataclass(metrics):
            values = asdict(metrics)
        elif isinstance(metrics, dict):
            values = metrics
        elif hasattr(metrics, "__dict__"):
            values = vars(metrics)
        else:
            raise TypeError(
                f"Metrikobjekt kann nicht gelesen werden: {type(metrics)}"
            )

        result = {}
        for key, value in values.items():
            full_key = key if key.startswith(f"{prefix}_") else f"{prefix}_{key}"
            result[full_key] = value
        return result

    @staticmethod
    def _valid(value):
        try:
            return value is not None and np.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _apply_bandpass(self, signal, sampling_rate):
        if hasattr(self.bandpass_filter, "filter"):
            return self.bandpass_filter.filter(signal, sampling_rate)
        if hasattr(self.bandpass_filter, "denoise"):
            try:
                return self.bandpass_filter.denoise(signal, sampling_rate)
            except TypeError:
                return self.bandpass_filter.denoise(signal)
        raise ValueError("BandpassFilter besitzt weder filter noch denoise.")

    @staticmethod
    def _embedded_value(
        embedded_metrics,
        name,
    ):
        if isinstance(
            embedded_metrics,
            dict,
        ):
            value = embedded_metrics.get(
                name,
                0.0,
            )
        else:
            value = getattr(
                embedded_metrics,
                name,
                0.0,
            )

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0