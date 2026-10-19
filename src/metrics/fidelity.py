from dataclasses import dataclass

import numpy as np


@dataclass
class FidelityMetrics:

    rmse: float

    prd: float

    snr_after_db: float
    snr_improvement_db: float

    full_segment_corr: float


def calculate_rmse(
        ref_signal,
        test_signal):
    """
    Root Mean Square Error.
    """

    ref_signal = np.asarray(ref_signal)
    test_signal = np.asarray(test_signal)

    return float(
        np.sqrt(
            np.mean(
                (ref_signal - test_signal) ** 2
            )
        )
    )


def calculate_prd(
        ref_signal,
        test_signal):
    """
    Percentage Root-mean-square Difference.
    """

    ref_signal = np.asarray(ref_signal)
    test_signal = np.asarray(test_signal)

    numerator = np.sum(
        (ref_signal - test_signal) ** 2
    )

    denominator = np.sum(
        ref_signal ** 2
    )

    if denominator == 0:
        return np.nan

    return float(
        np.sqrt(
            numerator / denominator
        ) * 100
    )


def calculate_snr(
        ref_signal,
        test_signal):
    """
    Referenzbasierter SNR.

    ref_signal = sauberes Signal
    test_signal = verrauschtes oder
                  gefiltertes Signal
    """

    ref_signal = np.asarray(ref_signal)
    test_signal = np.asarray(test_signal)

    noise = (
        ref_signal -
        test_signal
    )

    signal_power = np.mean(
        ref_signal ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    if noise_power == 0:
        return np.nan

    return float(
        10 * np.log10(
            signal_power /
            noise_power
        )
    )


def calculate_snr_improvement(
        snr_before_db,
        snr_after_db):
    """
    Verbesserter SNR durch Denoising.
    """

    return float(
        snr_after_db -
        snr_before_db
    )


def calculate_full_segment_correlation(
        ref_signal,
        test_signal):
    """
    Korrelation über das gesamte Segment.
    """

    ref_signal = np.asarray(ref_signal)
    test_signal = np.asarray(test_signal)

    if len(ref_signal) < 2:
        return np.nan

    return float(
        np.corrcoef(
            ref_signal,
            test_signal
        )[0, 1]
    )


def create_fidelity_metrics(
        clean_signal,
        noisy_signal,
        denoised_signal):
    """
    Erstellt ein komplettes FidelityMetrics-Objekt.
    """

    rmse = calculate_rmse(
        clean_signal,
        denoised_signal
    )

    prd = calculate_prd(
        clean_signal,
        denoised_signal
    )

    snr_before = calculate_snr(
        clean_signal,
        noisy_signal
    )

    snr_after = calculate_snr(
        clean_signal,
        denoised_signal
    )

    snr_improvement = calculate_snr_improvement(
        snr_before,
        snr_after
    )

    full_corr = calculate_full_segment_correlation(
        clean_signal,
        denoised_signal
    )

    return FidelityMetrics(
        rmse=rmse,
        prd=prd,
        snr_after_db=snr_after,
        snr_improvement_db=snr_improvement,
        full_segment_corr=full_corr
    )