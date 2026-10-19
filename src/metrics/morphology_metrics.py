from dataclasses import dataclass

import numpy as np


@dataclass
class MorphologyMetrics:

    beat_correlation: float

    qrs_amplitude_error: float
    qrs_duration_error: float

    st_segment_error: float

    p_amplitude_error: float
    t_amplitude_error: float

    qt_interval_error: float


def extract_beats(
        signal,
        rpeaks,
        fs,
        pre_ms=250,
        post_ms=400):
    """
    Extrahiert Herzschläge um die R-Peaks.
    """

    signal = np.asarray(signal)
    rpeaks = np.asarray(rpeaks)

    pre_samples = int(
        pre_ms / 1000 * fs
    )

    post_samples = int(
        post_ms / 1000 * fs
    )

    beats = []

    for rpeak in rpeaks:

        start = int(rpeak) - pre_samples
        end = int(rpeak) + post_samples

        if start < 0:
            continue

        if end > len(signal):
            continue

        beat = signal[start:end]

        beats.append(
            beat
        )

    return beats


def calculate_beat_correlation(
        ref_signal,
        test_signal,
        ref_rpeaks,
        test_rpeaks,
        fs):
    """
    Mittlere Korrelation zwischen Referenz- und Test-Beats.
    """

    ref_beats = extract_beats(
        ref_signal,
        ref_rpeaks,
        fs
    )

    test_beats = extract_beats(
        test_signal,
        test_rpeaks,
        fs
    )

    if len(ref_beats) == 0 or len(test_beats) == 0:
        return np.nan

    correlations = []

    for ref_beat, test_beat in zip(
            ref_beats,
            test_beats):

        min_len = min(
            len(ref_beat),
            len(test_beat)
        )

        ref_beat = ref_beat[:min_len]
        test_beat = test_beat[:min_len]

        if np.std(ref_beat) == 0 or np.std(test_beat) == 0:
            continue

        correlation = np.corrcoef(
            ref_beat,
            test_beat
        )[0, 1]

        correlations.append(
            correlation
        )

    if len(correlations) == 0:
        return np.nan

    return float(
        np.nanmean(correlations)
    )


def calculate_qrs_amplitudes(
        signal,
        rpeaks,
        fs,
        window_ms=80):
    """
    QRS-Amplitude als Peak-to-Peak-Wert um den R-Peak.
    """

    signal = np.asarray(signal)
    rpeaks = np.asarray(rpeaks)

    half_window = int(
        window_ms / 1000 * fs
    )

    amplitudes = []

    for rpeak in rpeaks:

        start = int(rpeak) - half_window
        end = int(rpeak) + half_window

        if start < 0:
            continue

        if end > len(signal):
            continue

        qrs_window = signal[start:end]

        amplitude = (
            np.max(qrs_window)
            -
            np.min(qrs_window)
        )

        amplitudes.append(
            amplitude
        )

    return amplitudes


def calculate_qrs_amplitude_error(
        ref_signal,
        test_signal,
        ref_rpeaks,
        test_rpeaks,
        fs):
    """
    Mittlerer absoluter Fehler der QRS-Amplitude.
    """

    ref_amplitudes = calculate_qrs_amplitudes(
        ref_signal,
        ref_rpeaks,
        fs
    )

    test_amplitudes = calculate_qrs_amplitudes(
        test_signal,
        test_rpeaks,
        fs
    )

    if len(ref_amplitudes) == 0 or len(test_amplitudes) == 0:
        return np.nan

    min_len = min(
        len(ref_amplitudes),
        len(test_amplitudes)
    )

    ref_amplitudes = np.asarray(
        ref_amplitudes[:min_len]
    )

    test_amplitudes = np.asarray(
        test_amplitudes[:min_len]
    )

    return float(
        np.mean(
            np.abs(
                ref_amplitudes
                -
                test_amplitudes
            )
        )
    )


def calculate_qrs_durations(
        signal,
        rpeaks,
        fs,
        window_ms=120,
        threshold_ratio=0.5):
    """
    Einfache QRS-Dauer-Schätzung über Amplitudenschwelle.
    """

    signal = np.asarray(signal)
    rpeaks = np.asarray(rpeaks)

    half_window = int(
        window_ms / 1000 * fs
    )

    durations = []

    for rpeak in rpeaks:

        start = int(rpeak) - half_window
        end = int(rpeak) + half_window

        if start < 0:
            continue

        if end > len(signal):
            continue

        qrs_window = signal[start:end]

        baseline = np.median(
            qrs_window
        )

        centered = np.abs(
            qrs_window - baseline
        )

        max_value = np.max(
            centered
        )

        if max_value == 0:
            continue

        threshold = (
            max_value * threshold_ratio
        )

        indices = np.where(
            centered >= threshold
        )[0]

        if len(indices) == 0:
            continue

        duration_samples = (
            indices[-1]
            -
            indices[0]
        )

        duration_ms = (
            duration_samples
            /
            fs
            *
            1000
        )

        durations.append(
            duration_ms
        )

    return durations


def calculate_qrs_duration_error(
        ref_signal,
        test_signal,
        ref_rpeaks,
        test_rpeaks,
        fs):
    """
    Mittlerer absoluter Fehler der geschätzten QRS-Dauer.
    """

    ref_durations = calculate_qrs_durations(
        ref_signal,
        ref_rpeaks,
        fs
    )

    test_durations = calculate_qrs_durations(
        test_signal,
        test_rpeaks,
        fs
    )

    if len(ref_durations) == 0 or len(test_durations) == 0:
        return np.nan

    min_len = min(
        len(ref_durations),
        len(test_durations)
    )

    ref_durations = np.asarray(
        ref_durations[:min_len]
    )

    test_durations = np.asarray(
        test_durations[:min_len]
    )

    return float(
        np.mean(
            np.abs(
                ref_durations
                -
                test_durations
            )
        )
    )


def calculate_st_values(
        signal,
        rpeaks,
        fs,
        start_ms=60,
        end_ms=120):
    """
    Einfache ST-Strecken-Schätzung nach dem R-Peak.
    """

    signal = np.asarray(signal)
    rpeaks = np.asarray(rpeaks)

    start_offset = int(
        start_ms / 1000 * fs
    )

    end_offset = int(
        end_ms / 1000 * fs
    )

    st_values = []

    for rpeak in rpeaks:

        start = int(rpeak) + start_offset
        end = int(rpeak) + end_offset

        if start < 0:
            continue

        if end > len(signal):
            continue

        st_window = signal[start:end]

        st_values.append(
            np.mean(st_window)
        )

    return st_values


def calculate_st_segment_error(
        ref_signal,
        test_signal,
        ref_rpeaks,
        test_rpeaks,
        fs):
    """
    Mittlerer Fehler der ST-Strecke.
    """

    ref_st_values = calculate_st_values(
        ref_signal,
        ref_rpeaks,
        fs
    )

    test_st_values = calculate_st_values(
        test_signal,
        test_rpeaks,
        fs
    )

    if len(ref_st_values) == 0 or len(test_st_values) == 0:
        return np.nan

    min_len = min(
        len(ref_st_values),
        len(test_st_values)
    )

    ref_st_values = np.asarray(
        ref_st_values[:min_len]
    )

    test_st_values = np.asarray(
        test_st_values[:min_len]
    )

    return float(
        np.mean(
            np.abs(
                ref_st_values
                -
                test_st_values
            )
        )
    )


def calculate_peak_amplitudes(
        signal,
        peaks):
    """
    Liest die Amplituden an gegebenen Peak-Positionen aus.
    Für P- und T-Wellen.
    """

    signal = np.asarray(signal)
    peaks = np.asarray(peaks)

    amplitudes = []

    for peak in peaks:

        peak = int(peak)

        if peak < 0:
            continue

        if peak >= len(signal):
            continue

        amplitudes.append(
            signal[peak]
        )

    return amplitudes


def calculate_peak_amplitude_error(
        ref_signal,
        test_signal,
        ref_peaks,
        test_peaks):
    """
    Mittlerer absoluter Amplitudenfehler zwischen
    Referenz-Peaks und Test-Peaks.
    """

    ref_amplitudes = calculate_peak_amplitudes(
        ref_signal,
        ref_peaks
    )

    test_amplitudes = calculate_peak_amplitudes(
        test_signal,
        test_peaks
    )

    if len(ref_amplitudes) == 0 or len(test_amplitudes) == 0:
        return np.nan

    min_len = min(
        len(ref_amplitudes),
        len(test_amplitudes)
    )

    ref_amplitudes = np.asarray(
        ref_amplitudes[:min_len]
    )

    test_amplitudes = np.asarray(
        test_amplitudes[:min_len]
    )

    return float(
        np.mean(
            np.abs(
                ref_amplitudes
                -
                test_amplitudes
            )
        )
    )


def calculate_p_amplitude_error(
        ref_signal,
        test_signal,
        ref_p_peaks,
        test_p_peaks):
    """
    Mittlerer absoluter Fehler der P-Wellen-Amplitude.
    """

    return calculate_peak_amplitude_error(
        ref_signal=ref_signal,
        test_signal=test_signal,
        ref_peaks=ref_p_peaks,
        test_peaks=test_p_peaks
    )


def calculate_t_amplitude_error(
        ref_signal,
        test_signal,
        ref_t_peaks,
        test_t_peaks):
    """
    Mittlerer absoluter Fehler der T-Wellen-Amplitude.
    """

    return calculate_peak_amplitude_error(
        ref_signal=ref_signal,
        test_signal=test_signal,
        ref_peaks=ref_t_peaks,
        test_peaks=test_t_peaks
    )


def calculate_qt_intervals(
        q_peaks,
        t_peaks,
        fs):
    """
    Einfache QT-Schätzung von Q-Peak bis T-Peak.

    Hinweis:
    Das ist keine klinisch perfekte QT-Messung,
    aber als vergleichende Metrik zwischen REF, RAW
    und gefilterten Signalen sinnvoll.
    """

    q_peaks = np.asarray(q_peaks)
    t_peaks = np.asarray(t_peaks)

    if len(q_peaks) == 0 or len(t_peaks) == 0:
        return []

    intervals = []

    min_len = min(
        len(q_peaks),
        len(t_peaks)
    )

    q_peaks = q_peaks[:min_len]
    t_peaks = t_peaks[:min_len]

    for q_peak, t_peak in zip(
            q_peaks,
            t_peaks):

        if t_peak <= q_peak:
            continue

        interval_ms = (
            (t_peak - q_peak)
            /
            fs
            *
            1000
        )

        intervals.append(
            interval_ms
        )

    return intervals


def calculate_qt_interval_error(
        ref_q_peaks,
        ref_t_peaks,
        test_q_peaks,
        test_t_peaks,
        fs):
    """
    Mittlerer absoluter Fehler der QT-Intervall-Schätzung.
    """

    ref_qt = calculate_qt_intervals(
        ref_q_peaks,
        ref_t_peaks,
        fs
    )

    test_qt = calculate_qt_intervals(
        test_q_peaks,
        test_t_peaks,
        fs
    )

    if len(ref_qt) == 0 or len(test_qt) == 0:
        return np.nan

    min_len = min(
        len(ref_qt),
        len(test_qt)
    )

    ref_qt = np.asarray(
        ref_qt[:min_len]
    )

    test_qt = np.asarray(
        test_qt[:min_len]
    )

    return float(
        np.mean(
            np.abs(
                ref_qt
                -
                test_qt
            )
        )
    )

def create_morphology_metrics(
        ref_signal,
        test_signal,
        ref_waves,
        test_waves,
        fs):
    """
    Erstellt alle Morphologie-Metriken direkt aus Signal,
    ECG-Wellen und Samplingrate.

    ref_waves und test_waves müssen folgende Attribute besitzen:
    p_peaks, q_peaks, r_peaks, s_peaks, t_peaks.
    """

    beat_correlation = calculate_beat_correlation(
        ref_signal,
        test_signal,
        ref_waves.r_peaks,
        test_waves.r_peaks,
        fs
    )

    qrs_amplitude_error = calculate_qrs_amplitude_error(
        ref_signal,
        test_signal,
        ref_waves.r_peaks,
        test_waves.r_peaks,
        fs
    )

    qrs_duration_error = calculate_qrs_duration_error(
        ref_signal,
        test_signal,
        ref_waves.r_peaks,
        test_waves.r_peaks,
        fs
    )

    st_segment_error = calculate_st_segment_error(
        ref_signal,
        test_signal,
        ref_waves.r_peaks,
        test_waves.r_peaks,
        fs
    )

    p_amplitude_error = calculate_p_amplitude_error(
        ref_signal,
        test_signal,
        ref_waves.p_peaks,
        test_waves.p_peaks
    )

    t_amplitude_error = calculate_t_amplitude_error(
        ref_signal,
        test_signal,
        ref_waves.t_peaks,
        test_waves.t_peaks
    )

    qt_interval_error = calculate_qt_interval_error(
        ref_waves.q_peaks,
        ref_waves.t_peaks,
        test_waves.q_peaks,
        test_waves.t_peaks,
        fs
    )

    return MorphologyMetrics(
        beat_correlation=beat_correlation,
        qrs_amplitude_error=qrs_amplitude_error,
        qrs_duration_error=qrs_duration_error,
        st_segment_error=st_segment_error,
        p_amplitude_error=p_amplitude_error,
        t_amplitude_error=t_amplitude_error,
        qt_interval_error=qt_interval_error
    )