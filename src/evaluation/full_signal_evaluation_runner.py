print("1: START FULL SIGNAL FILE")

from pathlib import Path

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

print("2: VOR KLASSE")

class FullSignalEvaluationRunner:
    """
    Führt eine referenzbasierte Evaluation des vollständigen
    Signals durch.

    Diese Auswertung dient als ergänzende Gesamtbetrachtung.
    Die spätere adaptive Pipeline arbeitet dagegen segmentweise.
    """

    def __init__(
        self,
        data_loader,
        denoisers,
        peak_detector,
        output_file,
        channel=0,
    ):
        self.data_loader = data_loader
        self.denoisers = denoisers
        self.peak_detector = peak_detector
        self.output_file = Path(output_file)
        self.channel = channel

    def run(
        self,
        noise_folder,
        noisy_record_name,
        reference_record_name,
    ):
        results = []

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

        clean_signal = self.data_loader.get_signal(
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

        reference_waves = (
            self.peak_detector.detect_waves(
                clean_signal
            )
        )

        reference_result = self._create_result(
            record_name=reference_record_name,
            method_name="REF",
            clean_signal=clean_signal,
            noisy_signal=clean_signal,
            processed_signal=clean_signal,
            reference_waves=reference_waves,
            sampling_rate=sampling_rate,
            embedded_metrics=self._empty_embedded_metrics(),
        )

        results.append(reference_result)

        raw_result = self._create_result(
            record_name=noisy_record_name,
            method_name="RAW",
            clean_signal=clean_signal,
            noisy_signal=noisy_signal,
            processed_signal=noisy_signal,
            reference_waves=reference_waves,
            sampling_rate=sampling_rate,
            embedded_metrics=self._empty_embedded_metrics(),
        )

        results.append(raw_result)

        for method_name, denoiser in self.denoisers.items():
            start_time = start_measurement()

            processed_signal = self._apply_denoiser(
                denoiser=denoiser,
                signal=noisy_signal,
                sampling_rate=sampling_rate,
            )

            embedded_metrics = stop_measurement(
                start_time
            )

            result = self._create_result(
                record_name=noisy_record_name,
                method_name=method_name,
                clean_signal=clean_signal,
                noisy_signal=noisy_signal,
                processed_signal=processed_signal,
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
        method_name,
        clean_signal,
        noisy_signal,
        processed_signal,
        reference_waves,
        sampling_rate,
        embedded_metrics,
    ):
        test_waves = self.peak_detector.detect_waves(
            processed_signal
        )

        medical_metrics = create_medical_metrics(
            reference_waves.r_peaks,
            test_waves.r_peaks,
            sampling_rate,
        )

        signal_quality_metrics = (
            create_signal_quality_metrics(
                processed_signal,
                test_waves.r_peaks,
                sampling_rate,
            )
        )

        morphology_metrics = create_morphology_metrics(
            clean_signal,
            processed_signal,
            reference_waves,
            test_waves,
            sampling_rate,
        )

        fidelity_metrics = create_fidelity_metrics(
            clean_signal=clean_signal,
            noisy_signal=noisy_signal,
            denoised_signal=processed_signal,
        )

        return SegmentResult(
            record=record_name,
            segment_id=0,
            method=method_name,
            channel=self.channel,
            num_samples=len(processed_signal),
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
    def _empty_embedded_metrics():
        start_time = start_measurement()

        return stop_measurement(start_time)