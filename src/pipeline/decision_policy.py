from enum import Enum


class SegmentDecision(str, Enum):
    KEEP = "KEEP"
    FILTER = "FILTER"
    REJECT = "REJECT"


class DecisionPolicy:
    def __init__(self, keep_threshold=0.35, reject_threshold=0.75):
        if not 0.0 <= keep_threshold < reject_threshold <= 1.0:
            raise ValueError(
                "Es muss gelten: 0 <= keep_threshold < "
                "reject_threshold <= 1."
            )

        self.keep_threshold = float(keep_threshold)
        self.reject_threshold = float(reject_threshold)

    def decide(self, quality_score):
        if quality_score is None:
            return SegmentDecision.REJECT

        score = float(quality_score)

        if score != score:
            return SegmentDecision.REJECT

        if score < self.keep_threshold:
            return SegmentDecision.KEEP

        if score < self.reject_threshold:
            return SegmentDecision.FILTER

        return SegmentDecision.REJECT
