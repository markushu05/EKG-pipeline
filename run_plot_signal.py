from pathlib import Path

from src.data.data_loader import ECGDataLoader
from src.denoising.bandpass import BandpassFilter
from src.plotting.signal_plotter import SignalPlotter


def main():

    fs = 360

    record = "118e_6"
    segment_idx = 35

    PROJECT_ROOT = Path(__file__).resolve().parent

    RAW_DATA_PATH = (
        PROJECT_ROOT
        / "data"
        / "raw"
    )

    loader = ECGDataLoader(
        data_path=RAW_DATA_PATH
    )

    reference = loader.load_record(
        "118/118"
    )

    noisy = loader.load_record(
        "118/118e_6"
    )

    reference_signal = reference.p_signal[:, 0]
    raw_signal = noisy.p_signal[:, 0]

    start = segment_idx * 10 * fs
    end = start + 10 * fs

    reference_segment = reference_signal[start:end]
    raw_segment = raw_signal[start:end]

    bandpass = BandpassFilter()

    # ANPASSEN AN DEINE KLASSE
    bandpass_segment = bandpass.filter(
        raw_segment,
        fs
    )

    SignalPlotter.save_stacked_signals(
        signals=[
            reference_segment,
            raw_segment,
            bandpass_segment
        ],
        labels=[
            "Reference",
            "RAW",
            "Always Bandpass"
        ],
        output_file="plots/filter_example.png",
        fs=fs
    )


if __name__ == "__main__":
    main()