from dataclasses import dataclass

import numpy as np
import neurokit2 as nk

from scipy.signal import butter
from scipy.signal import filtfilt

EPSILON = 1e-12

@dataclass
class SignalQualityMetrics:

    beat_template_correlation: float

    bsqi: float

    baseline_drift: float

    estimated_snr: float

    rr_plausibility: float


def _prepare_signal(signal):
    """
    Bereitet das Signal auf, ohne die Länge zu verändern.

    Nicht endliche Werte werden linear interpoliert.
    Das ist wichtig, damit vorhandene R-Peak-Indizes weiterhin
    zum Signal passen.
    """

    signal = np.asarray(
        signal,
        dtype=float
    ).reshape(-1)

    if len(signal) == 0:
        return signal

    finite_mask = np.isfinite(
        signal
    )

    if np.all(finite_mask):
        return signal

    if not np.any(finite_mask):
        return np.array(
            [],
            dtype=float
        )

    valid_indices = np.where(
        finite_mask
    )[0]

    invalid_indices = np.where(
        ~finite_mask
    )[0]

    signal[invalid_indices] = np.interp(
        invalid_indices,
        valid_indices,
        signal[valid_indices]
    )

    return signal


def _clean_peaks(
        rpeaks,
        signal_length):
    """
    Entfernt ungültige, nicht endliche und außerhalb des
    Signals liegende R-Peak-Indizes.
    """

    if rpeaks is None:
        return np.array(
            [],
            dtype=int
        )

    cleaned_peaks = []

    for peak in np.asarray(
            rpeaks
    ).reshape(-1):

        if peak is None:
            continue

        if not np.isfinite(
                peak):
            continue

        peak = int(
            peak
        )

        if peak < 0:
            continue

        if peak >= signal_length:
            continue

        cleaned_peaks.append(
            peak
        )

    if len(cleaned_peaks) == 0:
        return np.array(
            [],
            dtype=int
        )

    return np.asarray(
        sorted(
            set(
                cleaned_peaks
            )
        ),
        dtype=int
    )


def _safe_filtfilt(
        b,
        a,
        signal):
    """
    Führt filtfilt nur aus, wenn das Signal lang genug ist.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    minimum_length = (
        3
        *
        max(
            len(a),
            len(b)
        )
    )

    if len(signal) <= minimum_length:
        return signal.copy()

    try:

        return filtfilt(
            b,
            a,
            signal
        )

    except Exception:

        return signal.copy()


def calculate_beat_template_correlation(
        signal,
        rpeaks,
        fs,
        pre_ms=250,
        post_ms=400):
    """
    Berechnet die Konsistenz der Herzschläge.

    Vorgehen:
    1. Alle gültigen Beats um die R-Peaks ausschneiden.
    2. Aus allen Beats einen robusten Median-Beat bilden.
    3. Jeden Beat mit diesem Template korrelieren.
    4. Median der Korrelationen zurückgeben.

    Wertebereich näherungsweise:
    1.0  = sehr konsistente Beats
    0.0  = keine lineare Ähnlichkeit
    < 0  = starke Abweichung oder inverse Ähnlichkeit

    Höher = besser.
    """

    signal = _prepare_signal(
        signal
    )

    if len(signal) == 0:
        return np.nan

    rpeaks = _clean_peaks(
        rpeaks,
        len(signal)
    )

    if len(rpeaks) < 2:
        return np.nan

    pre_samples = int(
        pre_ms
        /
        1000
        *
        fs
    )

    post_samples = int(
        post_ms
        /
        1000
        *
        fs
    )

    beat_length = (
        pre_samples
        +
        post_samples
    )

    if beat_length < 2:
        return np.nan

    beats = []

    for rpeak in rpeaks:

        start = (
            int(rpeak)
            -
            pre_samples
        )

        end = (
            int(rpeak)
            +
            post_samples
        )

        if start < 0:
            continue

        if end > len(signal):
            continue

        beat = signal[
            start:end
        ]

        if len(beat) != beat_length:
            continue

        beats.append(
            beat
        )

    if len(beats) < 2:
        return np.nan

    beats = np.asarray(
        beats,
        dtype=float
    )

    template = np.median(
        beats,
        axis=0
    )

    template_std = np.std(
        template
    )

    if template_std == 0:
        return np.nan

    correlations = []

    for beat in beats:

        beat_std = np.std(
            beat
        )

        if beat_std == 0:
            continue

        correlation = np.corrcoef(
            beat,
            template
        )[0, 1]

        if not np.isfinite(
                correlation):
            continue

        correlations.append(
            correlation
        )

    if len(correlations) == 0:
        return np.nan

    return float(
        np.median(
            correlations
        )
    )


def _count_matching_peaks(
        reference_peaks,
        comparison_peaks,
        tolerance_samples):
    """
    Ordnet Peaks beider Detektoren innerhalb eines Zeitfensters
    eins zu eins einander zu.
    """

    reference_peaks = np.asarray(
        reference_peaks,
        dtype=int
    )

    comparison_peaks = np.asarray(
        comparison_peaks,
        dtype=int
    )

    if len(reference_peaks) == 0:
        return 0

    if len(comparison_peaks) == 0:
        return 0

    used_comparison_indices = set()

    matching_count = 0

    for reference_peak in reference_peaks:

        distances = np.abs(
            comparison_peaks
            -
            reference_peak
        )

        ordered_indices = np.argsort(
            distances
        )

        for comparison_index in ordered_indices:

            comparison_index = int(
                comparison_index
            )

            if comparison_index in used_comparison_indices:
                continue

            if distances[comparison_index] > tolerance_samples:
                break

            used_comparison_indices.add(
                comparison_index
            )

            matching_count += 1

            break

    return matching_count


def calculate_bsqi(
        signal,
        primary_rpeaks,
        fs,
        tolerance_ms=100):
    """
    Berechnet einen vereinfachten bSQI über die Übereinstimmung
    zweier QRS-Detektoren.

    Detektor 1:
    Die bereits vom Runner übergebenen R-Peaks.

    Detektor 2:
    NeuroKit mit Pan-Tompkins 1985.

    Formel:
    bSQI = 2 * matches / (n_primary + n_secondary)

    Wertebereich:
    1.0 = vollständige Übereinstimmung
    0.0 = keine Übereinstimmung

    Höher = besser.
    """

    signal = _prepare_signal(
        signal
    )

    if len(signal) == 0:
        return np.nan

    primary_rpeaks = _clean_peaks(
        primary_rpeaks,
        len(signal)
    )

    if len(primary_rpeaks) == 0:
        return 0.0

    try:

        _, secondary_info = nk.ecg_peaks(
            signal,
            sampling_rate=fs,
            method="pantompkins1985"
        )

        secondary_rpeaks = _clean_peaks(
            secondary_info.get(
                "ECG_R_Peaks"
            ),
            len(signal)
        )

    except Exception:

        return np.nan

    if len(secondary_rpeaks) == 0:
        return 0.0

    tolerance_samples = max(
        1,
        int(
            tolerance_ms
            /
            1000
            *
            fs
        )
    )

    matching_count = _count_matching_peaks(
        reference_peaks=primary_rpeaks,
        comparison_peaks=secondary_rpeaks,
        tolerance_samples=tolerance_samples
    )

    denominator = (
        len(primary_rpeaks)
        +
        len(secondary_rpeaks)
    )

    if denominator == 0:
        return np.nan

    bsqi = (
        2.0
        *
        matching_count
        /
        denominator
    )

    return float(
        np.clip(
            bsqi,
            0.0,
            1.0
        )
    )


def calculate_baseline_drift(
        signal,
        fs,
        cutoff_hz=0.5):
    """
    Berechnet einen normalisierten Baseline-Drift-Wert.

    Ein Lowpass unter 0.5 Hz schätzt die langsame Grundlinie.
    Die Streuung der Baseline wird durch die Streuung des
    Gesamtsignals geteilt.

    Dadurch ist der Wert weniger von der absoluten
    Signalamplitude abhängig.

    Höher = schlechter.
    """

    signal = _prepare_signal(
        signal
    )

    if len(signal) == 0:
        return np.nan

    if fs <= 0:
        return np.nan

    nyquist = (
        fs
        /
        2
    )

    if cutoff_hz >= nyquist:
        return np.nan

    b, a = butter(
        2,
        cutoff_hz,
        btype="low",
        fs=fs
    )

    baseline = _safe_filtfilt(
        b,
        a,
        signal
    )

    signal_std = np.std(
        signal
    )

    if signal_std < EPSILON:
        return np.nan

    baseline_std = np.std(
        baseline
    )

    return float(
        baseline_std
        /
        signal_std
    )


def calculate_estimated_snr(
        signal,
        fs,
        lowcut_hz=0.5,
        highcut_hz=40.0):
    """
    Berechnet eine referenzfreie SNR-Näherung.

    Der Anteil zwischen 0.5 und 40 Hz wird als grobe
    Schätzung des ECG-Nutzsignals verwendet.

    Die Differenz zwischen Rohsignal und bandbegrenztem
    Signal wird als geschätzter Störanteil verwendet.

    Rückgabe:
    geschätztes SNR in dB.

    Höher = besser.

    Hinweis:
    Das ist keine Referenz-SNR und damit nur ein SQI-Proxy.
    """

    signal = _prepare_signal(
        signal
    )

    if len(signal) == 0:
        return np.nan

    if fs <= 0:
        return np.nan

    nyquist = (
        fs
        /
        2
    )

    adjusted_highcut_hz = min(
        highcut_hz,
        nyquist
        -
        0.1
    )

    if adjusted_highcut_hz <= lowcut_hz:
        return np.nan

    b, a = butter(
        2,
        [
            lowcut_hz,
            adjusted_highcut_hz
        ],
        btype="bandpass",
        fs=fs
    )

    estimated_ecg = _safe_filtfilt(
        b,
        a,
        signal
    )

    estimated_noise = (
        signal
        -
        estimated_ecg
    )

    ecg_power = np.mean(
        estimated_ecg ** 2
    )

    noise_power = np.mean(
        estimated_noise ** 2
    )

    if ecg_power < EPSILON:
        return np.nan

    if noise_power < EPSILON:
        return float(
            60.0
        )

    return float(
        10.0
        *
        np.log10(
            ecg_power
            /
            noise_power
        )
    )


def calculate_rr_plausibility(
        rpeaks,
        fs,
        minimum_rr_ms=300.0,
        maximum_rr_ms=2000.0,
        minimum_adjacent_ratio=0.5,
        maximum_adjacent_ratio=2.0):
    """
    Bewertet die Plausibilität der detektierten RR-Intervalle.

    Zwei Kriterien werden kombiniert:

    1. Physiologischer Wertebereich:
       RR zwischen 300 und 2000 ms.

    2. Verhältnis aufeinanderfolgender RR-Intervalle:
       Das Nachbarverhältnis muss zwischen 0.5 und 2.0 liegen.

    Der Rückgabewert liegt zwischen 0 und 1.

    1.0 = alle geprüften RR-Intervalle plausibel
    0.0 = keine plausiblen RR-Intervalle

    Höher = besser.

    Die Grenzen sind konfigurierbare Startwerte und sollten
    für die Arbeit dokumentiert und empirisch geprüft werden.
    """

    if fs <= 0:
        return np.nan

    rpeaks = np.asarray(
        rpeaks
    ).reshape(-1)

    cleaned_peaks = []

    for peak in rpeaks:

        if peak is None:
            continue

        if not np.isfinite(
                peak):
            continue

        cleaned_peaks.append(
            int(peak)
        )

    if len(cleaned_peaks) < 2:
        return 0.0

    cleaned_peaks = np.asarray(
        sorted(
            set(
                cleaned_peaks
            )
        ),
        dtype=int
    )

    rr_intervals_ms = (
        np.diff(
            cleaned_peaks
        )
        /
        fs
        *
        1000.0
    )

    if len(rr_intervals_ms) == 0:
        return 0.0

    range_plausibility = np.mean(
        (
            rr_intervals_ms
            >=
            minimum_rr_ms
        )
        &
        (
            rr_intervals_ms
            <=
            maximum_rr_ms
        )
    )

    if len(rr_intervals_ms) < 2:

        ratio_plausibility = (
            range_plausibility
        )

    else:

        previous_rr = rr_intervals_ms[
            :-1
        ]

        next_rr = rr_intervals_ms[
            1:
        ]

        valid_denominator = (
            previous_rr
            >
            EPSILON
        )

        if not np.any(
                valid_denominator):

            ratio_plausibility = 0.0

        else:

            rr_ratios = (
                next_rr[
                    valid_denominator
                ]
                /
                previous_rr[
                    valid_denominator
                ]
            )

            ratio_plausibility = np.mean(
                (
                    rr_ratios
                    >=
                    minimum_adjacent_ratio
                )
                &
                (
                    rr_ratios
                    <=
                    maximum_adjacent_ratio
                )
            )

    rr_plausibility = np.mean(
        [
            range_plausibility,
            ratio_plausibility
        ]
    )

    return float(
        np.clip(
            rr_plausibility,
            0.0,
            1.0
        )
    )


def create_signal_quality_metrics(
        signal,
        rpeaks,
        fs):
    """
    Erstellt alle fünf Signalqualitätsmetriken.

    Die Funktion behält dieselbe Signatur wie vorher.
    Daher müssen FullSignalRunner und ExperimentRunner
    nicht angepasst werden.
    """

    return SignalQualityMetrics(
        beat_template_correlation=(
            calculate_beat_template_correlation(
                signal=signal,
                rpeaks=rpeaks,
                fs=fs
            )
        ),
        bsqi=calculate_bsqi(
            signal=signal,
            primary_rpeaks=rpeaks,
            fs=fs
        ),
        baseline_drift=calculate_baseline_drift(
            signal=signal,
            fs=fs
        ),
        estimated_snr=calculate_estimated_snr(
            signal=signal,
            fs=fs
        ),
        rr_plausibility=calculate_rr_plausibility(
            rpeaks=rpeaks,
            fs=fs
        )
    )