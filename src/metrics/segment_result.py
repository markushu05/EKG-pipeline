from dataclasses import dataclass, asdict

from src.metrics.medical_metrics import MedicalMetrics
from src.metrics.signal_quality_metrics import SignalQualityMetrics
from src.metrics.morphology_metrics import MorphologyMetrics
from src.metrics.fidelity import FidelityMetrics
from src.metrics.embedded import EmbeddedMetrics


@dataclass
class SegmentResult:

    record: str
    segment_id: int

    method: str
    channel: int

    num_samples: int

    medical: MedicalMetrics
    signal_quality: SignalQualityMetrics
    morphology: MorphologyMetrics
    fidelity: FidelityMetrics
    embedded: EmbeddedMetrics

    def to_dict(self):

        row = {
            "record": self.record,
            "segment_id": self.segment_id,
            "method": self.method,
            "channel": self.channel,
            "num_samples": self.num_samples,
        }

        row.update(
            {
                f"medical_{k}": v
                for k, v in asdict(self.medical).items()
            }
        )

        row.update(
            {
                f"quality_{k}": v
                for k, v in asdict(self.signal_quality).items()
            }
        )

        row.update(
            {
                f"morphology_{k}": v
                for k, v in asdict(self.morphology).items()
            }
        )

        row.update(
            {
                f"fidelity_{k}": v
                for k, v in asdict(self.fidelity).items()
            }
        )

        row.update(
            {
                f"embedded_{k}": v
                for k, v in asdict(self.embedded).items()
            }
        )

        return row