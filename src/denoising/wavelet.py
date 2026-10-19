import pywt
import numpy as np

class WaveletDenoiser:

    def __init__(
        self,
        wavelet="db8",
        level=7
    ):
        self.wavelet = wavelet
        self.level = level

    def denoise(self, signal):

        coeffs = pywt.wavedec(
            signal,
            self.wavelet,
            level=self.level
        )

        # Rauschschätzung über höchste Frequenzkomponente
        sigma = np.median(
            np.abs(coeffs[-1])
        ) / 0.6745

        threshold_factor = 2.0

        threshold = (
            sigma
            * np.sqrt(2 * np.log(len(signal)))
            * threshold_factor
        )

        coeffs_denoised = [coeffs[0]]

        for detail in coeffs[1:]:

            coeffs_denoised.append(
                pywt.threshold(
                    detail,
                    threshold,
                    mode="soft"
                )
            )

        reconstructed = pywt.waverec(
            coeffs_denoised,
            self.wavelet
        )

        return reconstructed[:len(signal)]