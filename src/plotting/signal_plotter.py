from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class SignalPlotter:

    @staticmethod
    def plot_signal(
            signal,
            title="Signal",
            fs=None,
            figsize=(12, 4)):
        """
        Einfaches Signal.
        """

        plt.figure(figsize=figsize)

        if fs is not None:
            x = np.arange(len(signal)) / fs
            plt.xlabel("Time [s]")
        else:
            x = np.arange(len(signal))
            plt.xlabel("Samples")

        plt.plot(x, signal)

        plt.title(title)
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_signals(
            signals,
            labels,
            title="Signal Comparison",
            fs=None,
            figsize=(12, 4)):
        """
        Mehrere Signale vergleichen.
        """

        plt.figure(figsize=figsize)

        for signal, label in zip(
                signals,
                labels):

            if fs is not None:
                x = np.arange(len(signal)) / fs
            else:
                x = np.arange(len(signal))

            plt.plot(
                x,
                signal,
                label=label
            )

        plt.title(title)

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def plot_with_rpeaks(
            signal,
            rpeaks,
            fs=None,
            title="Signal with R-Peaks",
            figsize=(12, 4)):
        """
        Signal mit Peak-Markierungen.
        """

        plt.figure(figsize=figsize)

        if fs is not None:
            x = np.arange(len(signal)) / fs
            peak_x = np.asarray(rpeaks) / fs
        else:
            x = np.arange(len(signal))
            peak_x = rpeaks

        plt.plot(
            x,
            signal,
            label="Signal"
        )

        plt.scatter(
            peak_x,
            signal[rpeaks],
            color="red",
            label="R-Peaks"
        )

        plt.title(title)

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def save_comparison(
            signals,
            labels,
            output_file,
            title="Signal Comparison",
            fs=None,
            figsize=(12, 4)):
        """
        Plot als Bild speichern.
        """

        output_file = Path(output_file)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.figure(figsize=figsize)

        for signal, label in zip(
                signals,
                labels):

            if fs is not None:
                x = np.arange(len(signal)) / fs
            else:
                x = np.arange(len(signal))

            plt.plot(
                x,
                signal,
                label=label
            )

        plt.title(title)

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


    @staticmethod
    def save_stacked_signals(
            signals,
            labels,
            output_file,
            fs=None,
            figsize=(12, 8)
    ):

        output_file = Path(output_file)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fig, axes = plt.subplots(
            len(signals),
            1,
            figsize=figsize,
            sharex=True
        )

    

        for ax, signal, label in zip(
                axes,
                signals,
                labels):

            if fs:
                x = np.arange(len(signal)) / fs
                ax.set_ylabel("mV")
            else:
                x = np.arange(len(signal))

            ax.plot(
                x,
                signal,
                linewidth=1
            )

            ax.set_title(label)

            ax.grid(True)

        axes[-1].set_xlabel("Time [s]")

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()