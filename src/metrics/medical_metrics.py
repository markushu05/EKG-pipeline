from dataclasses import dataclass

import numpy as np


@dataclass
class MedicalMetrics:

    hr_test_bpm: float
    hr_error_bpm: float

    rr_test_ms: float
    rr_error_ms: float

    sensitivity: float
    precision: float
    f1_score: float


def calculate_hr(rpeaks, fs):
    """
    Herzfrequenz in BPM aus R-Peaks.
    """

    if len(rpeaks) < 2:
        return np.nan

    rr_intervals = np.diff(rpeaks) / fs

    return float(
        60.0 / np.mean(rr_intervals)
    )


def calculate_rr_mean(rpeaks, fs):
    """
    Mittleres RR-Intervall in ms.
    """

    if len(rpeaks) < 2:
        return np.nan

    rr_intervals = np.diff(rpeaks) / fs

    return float(
        np.mean(rr_intervals) * 1000
    )


def calculate_hr_error(
        hr_ref_bpm,
        hr_test_bpm):
    """
    Absoluter Herzfrequenzfehler.
    """

    if np.isnan(hr_ref_bpm) or np.isnan(hr_test_bpm):
        return np.nan

    return float(
        abs(
            hr_ref_bpm -
            hr_test_bpm
        )
    )


def calculate_rr_error(
        rr_ref_ms,
        rr_test_ms):
    """
    Absoluter RR-Fehler.
    """

    if np.isnan(rr_ref_ms) or np.isnan(rr_test_ms):
        return np.nan

    return float(
        abs(
            rr_ref_ms -
            rr_test_ms
        )
    )


def match_rpeaks(
        ref_rpeaks,
        test_rpeaks,
        fs,
        tolerance_ms=100):
    """
    Vergleicht Referenz-R-Peaks mit Test-R-Peaks.

    Ein Peak gilt als True Positive,
    wenn er innerhalb der Toleranz liegt.
    """

    ref_rpeaks = np.asarray(ref_rpeaks)
    test_rpeaks = np.asarray(test_rpeaks)

    tolerance_samples = int(
        tolerance_ms / 1000 * fs
    )

    matched_test_indices = set()

    tp = 0
    fn = 0

    for ref_peak in ref_rpeaks:

        if len(test_rpeaks) == 0:
            fn += 1
            continue

        distances = np.abs(
            test_rpeaks - ref_peak
        )

        closest_index = int(
            np.argmin(distances)
        )

        closest_distance = distances[
            closest_index
        ]

        if (
            closest_distance <= tolerance_samples
            and closest_index not in matched_test_indices
        ):
            tp += 1
            matched_test_indices.add(
                closest_index
            )

        else:
            fn += 1

    fp = len(test_rpeaks) - len(matched_test_indices)

    return tp, fp, fn


def calculate_sensitivity(
        tp,
        fn):
    """
    Recall / Sensitivity.
    """

    denominator = tp + fn

    if denominator == 0:
        return np.nan

    return float(
        tp / denominator
    )


def calculate_precision(
        tp,
        fp):
    """
    Precision.
    """

    denominator = tp + fp

    if denominator == 0:
        return np.nan

    return float(
        tp / denominator
    )


def calculate_f1_score(
        sensitivity,
        precision):
    """
    F1 Score.
    """

    if np.isnan(sensitivity) or np.isnan(precision):
        return np.nan

    denominator = (
        sensitivity +
        precision
    )

    if denominator == 0:
        return np.nan

    return float(
        2 *
        sensitivity *
        precision /
        denominator
    )


def create_medical_metrics(
        ref_rpeaks,
        test_rpeaks,
        fs,
        tolerance_ms=100):
    """
    Erstellt alle medizinischen Metriken direkt aus den R-Peaks.
    Der Runner muss dadurch nur ref_rpeaks, test_rpeaks und fs übergeben.
    """

    hr_ref_bpm = calculate_hr(
        ref_rpeaks,
        fs
    )

    hr_test_bpm = calculate_hr(
        test_rpeaks,
        fs
    )

    rr_ref_ms = calculate_rr_mean(
        ref_rpeaks,
        fs
    )

    rr_test_ms = calculate_rr_mean(
        test_rpeaks,
        fs
    )

    hr_error = calculate_hr_error(
        hr_ref_bpm,
        hr_test_bpm
    )

    rr_error = calculate_rr_error(
        rr_ref_ms,
        rr_test_ms
    )

    tp, fp, fn = match_rpeaks(
        ref_rpeaks=ref_rpeaks,
        test_rpeaks=test_rpeaks,
        fs=fs,
        tolerance_ms=tolerance_ms
    )

    sensitivity = calculate_sensitivity(
        tp,
        fn
    )

    precision = calculate_precision(
        tp,
        fp
    )

    f1 = calculate_f1_score(
        sensitivity,
        precision
    )

    return MedicalMetrics(
        #hr_ref_bpm=hr_ref_bpm,
        hr_test_bpm=hr_test_bpm,
        hr_error_bpm=hr_error,
        #rr_ref_ms=rr_ref_ms,
        rr_test_ms=rr_test_ms,
        rr_error_ms=rr_error,
        sensitivity=sensitivity,
        precision=precision,
        f1_score=f1
    )