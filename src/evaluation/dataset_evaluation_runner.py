from pathlib import Path

from src.evaluation.segmented_evaluation_runner import (
    SegmentedEvaluationRunner,
)

from src.evaluation.full_signal_evaluation_runner import (
    FullSignalEvaluationRunner,
)


class DatasetEvaluationRunner:

    def __init__(
        self,
        data_loader,
        denoisers,
        peak_detector,
        raw_data_path,
        output_path,
        segment_length_s=10,
        channel=0,
        run_segmented=True,
        run_full_signal=True,
    ):
        self.data_loader = data_loader
        self.denoisers = denoisers
        self.peak_detector = peak_detector

        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)

        self.segment_length_s = segment_length_s
        self.channel = channel

        self.run_segmented = run_segmented
        self.run_full_signal = run_full_signal

    def run(self):

        processed_records = []

        folders = sorted(
            folder
            for folder in self.raw_data_path.iterdir()
            if folder.is_dir()
        )

        for reference_folder in folders:

            folder_name = reference_folder.name

            #
            # 118e -> 118
            #
            reference_record_name = (
                folder_name.rstrip("e")
            )

            record_names = self._get_record_names(
                reference_folder
            )

            segmented_output_folder = (
                self.output_path
                / reference_record_name
                / "segmented"
            )

            full_output_folder = (
                self.output_path
                / reference_record_name
                / "full_signal"
            )

            segmented_output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            full_output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            for record_name in record_names:

                if record_name == reference_record_name:
                    continue

                print()
                print(
                    f"Starte Verarbeitung: {record_name}"
                )

                if self.run_full_signal:

                    full_output_file = (
                        full_output_folder
                        / f"results_{record_name}.xlsx"
                    )

                    runner = FullSignalEvaluationRunner(
                        data_loader=self.data_loader,
                        denoisers=self.denoisers,
                        peak_detector=self.peak_detector,
                        output_file=full_output_file,
                        channel=self.channel,
                    )

                    runner.run(
                        noise_folder=folder_name,
                        noisy_record_name=record_name,
                        reference_record_name=(
                            reference_record_name
                        ),
                    )

                if self.run_segmented:

                    segmented_output_file = (
                        segmented_output_folder
                        / f"results_{record_name}.xlsx"
                    )

                    runner = SegmentedEvaluationRunner(
                        data_loader=self.data_loader,
                        denoisers=self.denoisers,
                        peak_detector=self.peak_detector,
                        output_file=segmented_output_file,
                        segment_length_s=self.segment_length_s,
                        channel=self.channel,
                    )

                    runner.run(
                        noise_folder=folder_name,
                        noisy_record_name=record_name,
                        reference_record_name=(
                            reference_record_name
                        ),
                    )

                processed_records.append(
                    record_name
                )

                print(
                    f"Fertig: {record_name}"
                )

        return processed_records

    @staticmethod
    def _get_record_names(
        folder_path,
    ):
        record_names = []

        for header_file in folder_path.glob(
            "*.hea"
        ):
            record_names.append(
                header_file.stem
            )

        return sorted(record_names)