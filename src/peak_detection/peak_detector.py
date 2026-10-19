from dataclasses import dataclass

import numpy as np
import neurokit2 as nk


@dataclass
class ECGWaves:

    p_peaks: np.ndarray

    q_peaks: np.ndarray
    r_peaks: np.ndarray
    s_peaks: np.ndarray

    t_peaks: np.ndarray


class ECGWaveDetector:

    def __init__(
            self,
            sampling_rate: int):

        self.sampling_rate = sampling_rate

    @staticmethod
    def _clean_peaks(
            peaks):

        if peaks is None:
            return np.array(
                [],
                dtype=int
            )

        cleaned = []

        for peak in peaks:

            if peak is None:
                continue

            if np.isnan(peak):
                continue

            cleaned.append(
                int(peak)
            )

        return np.asarray(
            cleaned,
            dtype=int
        )

    def detect_waves(
            self,
            ecg_signal):

        try:

            _, info = nk.ecg_peaks(
                ecg_signal,
                sampling_rate=self.sampling_rate
            )

            _, waves = nk.ecg_delineate(
                ecg_signal,
                rpeaks=info["ECG_R_Peaks"],
                sampling_rate=self.sampling_rate,
                method="dwt"
            )

            return ECGWaves(

                p_peaks=self._clean_peaks(
                    waves.get(
                        "ECG_P_Peaks"
                    )
                ),

                q_peaks=self._clean_peaks(
                    waves.get(
                        "ECG_Q_Peaks"
                    )
                ),

                r_peaks=self._clean_peaks(
                    info.get(
                        "ECG_R_Peaks"
                    )
                ),

                s_peaks=self._clean_peaks(
                    waves.get(
                        "ECG_S_Peaks"
                    )
                ),

                t_peaks=self._clean_peaks(
                    waves.get(
                        "ECG_T_Peaks"
                    )
                )
            )

        except Exception as e:

            print(
                "WaveDetector Error:",
                e
            )

            return ECGWaves(
                p_peaks=np.array([], dtype=int),
                q_peaks=np.array([], dtype=int),
                r_peaks=np.array([], dtype=int),
                s_peaks=np.array([], dtype=int),
                t_peaks=np.array([], dtype=int)
            )