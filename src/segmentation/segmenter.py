from dataclasses import dataclass

import numpy as np


@dataclass
class SignalSegment:

    segment_id: int

    start_idx: int
    end_idx: int

    start_time_s: float
    end_time_s: float

    samples: np.ndarray


class Segmenter:

    @staticmethod
    def split(
            signal,
            sampling_rate,
            segment_length_s=10):
        """
        Teilt ein Signal in gleich lange Segmente.
        """

        segment_size = int(
            sampling_rate *
            segment_length_s
        )

        segments = []

        segment_id = 0

        for start_idx in range(
                0,
                len(signal),
                segment_size):

            end_idx = (
                start_idx +
                segment_size
            )

            if end_idx > len(signal):
                break

            segments.append(
                SignalSegment(
                    segment_id=segment_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time_s=(
                        start_idx /
                        sampling_rate
                    ),
                    end_time_s=(
                        end_idx /
                        sampling_rate
                    ),
                    samples=signal[
                        start_idx:end_idx
                    ]
                )
            )

            segment_id += 1

        return segments