"""The bystander gate.

Only consenting people appear in the source video.  The gate stops the run
from touching any frame that holds a second person.

The gate has three outcomes:

- PASS. The frame holds exactly one face.  That face is the target.
- SKIP. The frame holds no face.  The frame passes through untouched.
  This is a normal outcome, not an error.
- REFUSE. The frame holds two or more faces.  The frame is skipped
  untouched, because the gate cannot tell the consenting person from a
  bystander.

The gate never chooses between faces.  It does not pick the largest face, the
first face, or the face with the highest score.  Any such choice could select a
bystander, so there is no choice to make and no parameter that changes this.
There is no bypass.

`RefusalTracker` holds the run level rule.  It stops the run when more than
`LIMIT_PERCENT` of the frames it has seen were refused.  A video of a person
walking past a crowd must fail the run, not return a video of the bystander.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .detect import FaceBox


class Outcome(StrEnum):
    """What the gate decided for one frame."""

    PASS = "PASS"
    SKIP = "SKIP"
    REFUSE = "REFUSE"


@dataclass(frozen=True)
class GateDecision:
    """The gate result for one frame.

    `face` holds the single detected face on PASS.  It is None on SKIP and on
    REFUSE.
    """

    outcome: Outcome
    face: FaceBox | None


def evaluate(faces: list[FaceBox]) -> GateDecision:
    """Return the gate decision for a frame that holds `faces`."""
    count = len(faces)
    if count == 1:
        return GateDecision(Outcome.PASS, faces[0])
    if count == 0:
        return GateDecision(Outcome.SKIP, None)
    return GateDecision(Outcome.REFUSE, None)


class RefusalLimitExceeded(RuntimeError):
    """Too many frames in the run held more than one face."""


class RefusalTracker:
    """Count the refused frames of one run.

    The frame loop calls `record` for every frame it processes.  The method
    raises `RefusalLimitExceeded` once the refused share is above
    `LIMIT_PERCENT`, so the loop stops.  This limit is fixed.  No parameter or
    setting can raise or lower it.
    """

    LIMIT_PERCENT = 10

    def __init__(self) -> None:
        self.processed = 0
        self.refused = 0

    def record(self, decision: GateDecision) -> None:
        """Count one processed frame.  Raise when the run is over the limit."""
        self.processed += 1
        if decision.outcome is Outcome.REFUSE:
            self.refused += 1
        # Integer compare.  It does not round 10.0 percent up to a refusal.
        if self.refused * 100 > self.processed * self.LIMIT_PERCENT:
            raise RefusalLimitExceeded(
                f"{self.refused} of {self.processed} frames held more than one "
                f"face, above {self.LIMIT_PERCENT} percent. Stopping the run."
            )
