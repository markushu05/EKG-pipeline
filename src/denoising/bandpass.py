from scipy.signal import butter, filtfilt

class BandpassFilter:

    def __init__(
        self,
        lowcut=0.5,
        highcut=40,
        order=4
    ):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def filter(
        self,
        signal,
        sampling_rate
    ):

        nyquist = 0.5 * sampling_rate

        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        b, a = butter(
            self.order,
            [low, high],
            btype="band"
        )

        filtered_signal = filtfilt(
            b,
            a,
            signal
        )

        return filtered_signal