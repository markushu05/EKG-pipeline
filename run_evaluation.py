from pathlib import Path

from src.data.data_loader import (
    ECGDataLoader
)

from src.denoising.bandpass import (
    BandpassFilter
)

from src.denoising.wavelet import (
    WaveletDenoiser
)

from src.denoising.swt import (
    SWTDenoiser
)

from src.peak_detection.peak_detector import (
    ECGWaveDetector
)

from src.evaluation import (
    DatasetEvaluationRunner
)


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
)

SEGMENT_LENGTH_S = 10
CHANNEL = 0
SAMPLING_RATE = 360

RUN_SEGMENTED_EVALUATION = True
RUN_FULL_SIGNAL_EVALUATION = True


def create_evaluation_runner():
    data_loader = ECGDataLoader(
        data_path=RAW_DATA_PATH
    )

    denoisers = {
        "Bandpass": BandpassFilter(),
        "Wavelet": WaveletDenoiser(),
        "SWT": SWTDenoiser(),
    }

    peak_detector = ECGWaveDetector(
        sampling_rate=SAMPLING_RATE
    )

    return DatasetEvaluationRunner(
        data_loader=data_loader,
        denoisers=denoisers,
        peak_detector=peak_detector,
        raw_data_path=RAW_DATA_PATH,
        output_path=OUTPUT_PATH,
        segment_length_s=SEGMENT_LENGTH_S,
        channel=CHANNEL,
        run_segmented=RUN_SEGMENTED_EVALUATION,
        run_full_signal=RUN_FULL_SIGNAL_EVALUATION,
    )


def run_evaluation():
    runner = create_evaluation_runner()

    return runner.run()


def main():
    run_evaluation()


if __name__ == "__main__":
    main()