from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.analysis import (
    GlobalBandpassAnalysis,
    RecordAnalysis,
)


INPUT_ROOT = Path("output")

INPUT_PATTERN = "*/segmented/results_*.xlsx"


def run_record_analyses():
    """
    Erstellt für jede results_*.xlsx eine separate Analyse.

    Beispiel:
        output/118e24/analysis/
        output/118e18/analysis/
        output/119e24/analysis/
    """

    analysis = RecordAnalysis(
        input_root=INPUT_ROOT,
        input_pattern=INPUT_PATTERN,
    )

    return analysis.run()


def run_global_bandpass_analysis():
    """
    Erstellt eine gemeinsame globale Analyse aller Records.

    Es werden alle Records gemeinsam normalisiert und nur
    die Bandpass-Ergebnisse ausgewertet.
    """

    output_file = (
        INPUT_ROOT
        / "analysis"
        / "all_records_bandpass_analysis.xlsx"
    )

    analysis = GlobalBandpassAnalysis(
        input_root=INPUT_ROOT,
        input_pattern=INPUT_PATTERN,
        output_file=output_file,
    )

    return analysis.run()


def run_all_analyses():
    """
    Führt zuerst die einzelnen Record-Analysen und danach
    die globale Bandpass-Analyse aus.
    """

    print()
    print("=" * 70)
    print("STARTE ANALYSEPIPELINE")
    print("=" * 70)

    record_output_files = run_record_analyses()

    print()
    print("=" * 70)

    global_output_file = (
        run_global_bandpass_analysis()
    )

    print()
    print("=" * 70)
    print("ANALYSEPIPELINE ABGESCHLOSSEN")
    print("=" * 70)

    print()
    print(
        "Anzahl erzeugter Record-Analysen: "
        f"{len(record_output_files)}"
    )

    if global_output_file is not None:
        print(
            "Globale Analysedatei: "
            f"{global_output_file}"
        )

    return {
        "record_output_files": record_output_files,
        "global_output_file": global_output_file,
    }


def main():
    run_all_analyses()


if __name__ == "__main__":
    main()