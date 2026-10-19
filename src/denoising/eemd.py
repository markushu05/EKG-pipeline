from PyEMD import EEMD
import numpy as np

class EEMDDenoiser:

    def __init__(
        self,
        trials=50,
        noise_width=0.05
    ):
        self.trials = trials
        self.noise_width = noise_width

    def denoise(self, signal):

        eemd = EEMD(
            trials=self.trials,
            noise_width=self.noise_width
        )

        imfs = eemd.eemd(signal)

        # erste IMF = meist hochfrequentes Rauschen
        # letzte IMF = oft Baseline-Wander

        if len(imfs) < 3:
            return signal

        reconstructed = np.sum(
            imfs[1:],
            axis=0
        )

        return reconstructed