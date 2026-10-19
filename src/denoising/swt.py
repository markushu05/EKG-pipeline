import pywt
import numpy as np

class SWTDenoiser:

    def __init__(
        self,
        wavelet="rbio3.9",
        level=4,
        threshold_scale=0.5
    ):
        self.wavelet = wavelet
        self.level = level
        self.threshold_scale = threshold_scale

    def denoise(self, signal):

        coeffs = pywt.swt(
            signal,
            self.wavelet,
            level=self.level
        )

        coeffs_denoised = []

        for cA, cD in coeffs:

            sigma = (
                np.median(np.abs(cD))
                / 0.6745
            )

            threshold = (
                sigma
                * np.sqrt(2 * np.log(len(signal)))
                * self.threshold_scale
            )

            cD_thresh = pywt.threshold(
                cD,
                threshold,
                mode="soft"
            )

            coeffs_denoised.append(
                (cA, cD_thresh)
            )

        reconstructed = pywt.iswt(
            coeffs_denoised,
            self.wavelet
        )

        return reconstructed[:len(signal)]