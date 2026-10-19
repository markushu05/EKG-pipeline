from pathlib import Path

from src.data.data_loader import ECGDataLoader
from src.denoising.bandpass import BandpassFilter
from src.peak_detection.peak_detector import ECGWaveDetector

from src.pipeline.adaptive_pipeline import (
    AdaptivePipeline,
    EmpiricalQualityScorer,
)
from src.pipeline.decision_policy import DecisionPolicy
from src.pipeline_evaluation.pipeline_evaluator import PipelineEvaluator
from src.pipeline_evaluation.dataset_evaluator import DatasetEvaluator


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
GLOBAL_ANALYSIS_FILE = (
    PROJECT_ROOT
    / "output"
    / "analysis"
    / "all_records_bandpass_analysis.xlsx"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "pipeline_evaluation"
    / "pipeline_strategy_comparison.xlsx"
)

SEGMENT_LENGTH_S = 10
CHANNEL = 0
SAMPLING_RATE = 360
KEEP_THRESHOLD = 0.35
REJECT_THRESHOLD = 0.75


def create_dataset_evaluator():
    data_loader = ECGDataLoader(data_path=RAW_DATA_PATH)
    bandpass_filter = BandpassFilter()
    peak_detector = ECGWaveDetector(sampling_rate=SAMPLING_RATE)

    quality_scorer = EmpiricalQualityScorer.from_analysis_file(
        GLOBAL_ANALYSIS_FILE,
        snr_weight=0.70,
        correlation_weight=0.30,
    )

    decision_policy = DecisionPolicy(
        keep_threshold=KEEP_THRESHOLD,
        reject_threshold=REJECT_THRESHOLD,
    )

    adaptive_pipeline = AdaptivePipeline(
        peak_detector=peak_detector,
        denoiser=bandpass_filter,
        quality_scorer=quality_scorer,
        decision_policy=decision_policy,
        segment_length_s=SEGMENT_LENGTH_S,
    )

    pipeline_evaluator = PipelineEvaluator(
        adaptive_pipeline=adaptive_pipeline,
        peak_detector=peak_detector,
        bandpass_filter=bandpass_filter,
        segment_length_s=SEGMENT_LENGTH_S,
    )

    return DatasetEvaluator(
        data_loader=data_loader,
        pipeline_evaluator=pipeline_evaluator,
        raw_data_path=RAW_DATA_PATH,
        output_file=OUTPUT_FILE,
        channel=CHANNEL,
    )


def main():
    evaluator = create_dataset_evaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
