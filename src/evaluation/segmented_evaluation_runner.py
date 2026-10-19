from pathlib import Path
import re

from src.metrics.embedded import (
    start_measurement,
    stop_measurement,
)

from src.metrics.medical_metrics import (
    create_medical_metrics,
)

from src.metrics.signal_quality_metrics import (
    create_signal_quality_metrics,
)

from src.metrics.morphology_metrics import (
    create_morphology_metrics,
)

from src.metrics.fidelity import (
    create_fidelity_metrics,
)

from src.metrics.segment_result import (
    SegmentResult,
)

from src.export.excel_exporter import (
    ExcelExporter,
)

from src.segmentation.segmenter import (
    Segmenter,
)


class SegmentedEvaluationRunner:
    """
    Führt eine referenzbasierte, segmentweise Evaluation durch.

    Für jedes Segment werden erzeugt:

        REF
        RAW
        Ergebnisse aller übergebenen Denoising-Verfahren

    Die Klasse dient der Offline-Evaluation für die
    Bachelorarbeit. Sie ist nicht die adaptive Laufzeitpipeline.
    """

    def __init__(
        self,
        data_loader,
        denoisers,
        peak_detector,
        output_file,
        segment_length_s=10,
        channel=0,
    ):
        self.data_loader = data_loader
        self.denoisers = denoisers
        self.peak_detector = peak_detector
        self.output_file = Path(output_file)
        self.segment_length_s = segment_length_s
        self.channel = channel

    def run(
        self,
        noise_folder,
        noisy_record_name,
        reference_record_name=None,
    ):
        results = []

        if reference_record_name is None:
            reference_record_name = (
                self._infer_reference_record_name(
                    noisy_record_name
                )
            )

        reference_path = (
            Path(noise_folder)
            / reference_record_name
        )

        noisy_path = (
            Path(noise_folder)
            / noisy_record_name
        )

        reference_record = self.data_loader.load_record(
            reference_path
        )

        noisy_record = self.data_loader.load_record(
            noisy_path
        )

        reference_signal = self.data_loader.get_signal(
            reference_record,
            channel=self.channel,
        )

        noisy_signal = self.data_loader.get_signal(
            noisy_record,
            channel=self.channel,
        )

        sampling_rate = (
            self.data_loader.get_sampling_rate(
                reference_record
            )
        )

        reference_segments = Segmenter.split(
            signal=reference_signal,
            sampling_rate=sampling_rate,
            segment_length_s=self.segment_length_s,
        )

        noisy_segments = Segmenter.split(
            signal=noisy_signal,
            sampling_rate=sampling_rate,
            segment_length_s=self.segment_length_s,
        )

        number_of_segments = min(
            len(reference_segments),
            len(noisy_segments),
        )

        if len(reference_segments) != len(noisy_segments):
            print(
                "WARNUNG: Referenz und verrauschtes Signal "
                "haben unterschiedlich viele Segmente."
            )

            print(
                "Es werden nur die gemeinsam vorhandenen "
                f"{number_of_segments} Segmente verarbeitet."
            )

        for reference_segment, noisy_segment_object in zip(
            reference_segments,
            noisy_segments,
        ):
            clean_segment = reference_segment.samples
            noisy_segment = noisy_segment_object.samples

            reference_waves = (
                self.peak_detector.detect_waves(
                    clean_segment
                )
            )

            reference_result = self._create_result(
                record_name=reference_record_name,
                segment_id=reference_segment.segment_id,
                method_name="REF",
                clean_segment=clean_segment,
                noisy_segment=clean_segment,
                processed_segment=clean_segment,
                reference_waves=reference_waves,
                sampling_rate=sampling_rate,
                embedded_metrics=(
                    self._empty_embedded_metrics()
                ),
            )

            results.append(reference_result)

            raw_result = self._create_result(
                record_name=noisy_record_name,
                segment_id=noisy_segment_object.segment_id,
                method_name="RAW",
                clean_segment=clean_segment,
                noisy_segment=noisy_segment,
                processed_segment=noisy_segment,
                reference_waves=reference_waves,
                sampling_rate=sampling_rate,
                embedded_metrics=(
                    self._empty_embedded_metrics()
                ),
            )

            results.append(raw_result)

            for method_name, denoiser in self.denoisers.items():
                start_time = start_measurement()

                processed_segment = self._apply_denoiser(
                    denoiser=denoiser,
                    signal=noisy_segment,
                    sampling_rate=sampling_rate,
                )

                embedded_metrics = stop_measurement(
                    start_time
                )

                result = self._create_result(
                    record_name=noisy_record_name,
                    segment_id=(
                        noisy_segment_object.segment_id
                    ),
                    method_name=method_name,
                    clean_segment=clean_segment,
                    noisy_segment=noisy_segment,
                    processed_segment=processed_segment,
                    reference_waves=reference_waves,
                    sampling_rate=sampling_rate,
                    embedded_metrics=embedded_metrics,
                )

                results.append(result)

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ExcelExporter.export_results(
            results,
            str(self.output_file),
        )

        return results

    def _create_result(
        self,
        record_name,
        segment_id,
        method_name,
        clean_segment,
        noisy_segment,
        processed_segment,
        reference_waves,
        sampling_rate,
        embedded_metrics,
    ):
        test_waves = self.peak_detector.detect_waves(
            processed_segment
        )

        medical_metrics = create_medical_metrics(
            reference_waves.r_peaks,
            test_waves.r_peaks,
            sampling_rate,
        )

        signal_quality_metrics = (
            create_signal_quality_metrics(
                processed_segment,
                test_waves.r_peaks,
                sampling_rate,
            )
        )

        morphology_metrics = create_morphology_metrics(
            clean_segment,
            processed_segment,
            reference_waves,
            test_waves,
            sampling_rate,
        )

        fidelity_metrics = create_fidelity_metrics(
            clean_signal=clean_segment,
            noisy_signal=noisy_segment,
            denoised_signal=processed_segment,
        )

        return SegmentResult(
            record=record_name,
            segment_id=segment_id,
            method=method_name,
            channel=self.channel,
            num_samples=len(processed_segment),
            medical=medical_metrics,
            signal_quality=signal_quality_metrics,
            morphology=morphology_metrics,
            fidelity=fidelity_metrics,
            embedded=embedded_metrics,
        )

    @staticmethod
    def _apply_denoiser(
        denoiser,
        signal,
        sampling_rate,
    ):
        if hasattr(denoiser, "denoise"):
            try:
                return denoiser.denoise(
                    signal,
                    sampling_rate,
                )
            except TypeError:
                return denoiser.denoise(signal)

        if hasattr(denoiser, "filter"):
            return denoiser.filter(
                signal,
                sampling_rate,
            )

        raise ValueError(
            "Unbekannter Denoiser-Typ: "
            f"{type(denoiser)}"
        )

    @staticmethod
    def _infer_reference_record_name(
        noisy_record_name,
    ):
        match = re.match(
            r"(\d+)",
            noisy_record_name,
        )

        if match is None:
            raise ValueError(
                "Referenzrecord konnte nicht aus "
                "folgendem Recordnamen bestimmt werden: "
                f"{noisy_record_name}"
            )

        return match.group(1)

    @staticmethod
    def _empty_embedded_metrics():
        start_time = start_measurement()

        return stop_measurement(start_time)