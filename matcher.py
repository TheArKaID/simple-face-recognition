"""Decision logic: threshold band plus the 1:N impostor cross-check.

A 1:1 "is this Budi?" test is the hard version of the question.  Here the
impostor population is known - it is the other employees - so the probe is
scored against everyone and the claim only stands if it also wins by a margin.
A colleague who merely looks similar still resembles their own template more,
and is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import config
import engine


@dataclass
class Decision:
    decision: str                       # accept | review | reject
    distance: Optional[float] = None    # to the claimed employee; lower is closer
    runner_up_distance: Optional[float] = None
    runner_up_id: Optional[str] = None  # internal only - never returned to callers
    margin: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    candidates_checked: int = 0

    @property
    def match(self) -> bool:
        """True when attendance may be recorded, review included."""
        return self.decision in ("accept", "review")

    @property
    def requires_review(self) -> bool:
        return self.decision == "review"

    def public_data(self) -> dict:
        """Response body for the HRIS.

        The runner-up's identity is deliberately withheld - the caller needs
        the score to reason about confidence, not the name of the employee
        someone was mistaken for.
        """
        return {
            "decision": self.decision,
            "match": self.match,
            "requires_review": self.requires_review,
            "distance": None if self.distance is None else round(self.distance, 4),
            "runner_up_distance": (
                None if self.runner_up_distance is None else round(self.runner_up_distance, 4)
            ),
            "margin": None if self.margin is None else round(self.margin, 4),
            "candidates_checked": self.candidates_checked,
            "reasons": self.reasons,
        }


def _band(distance: float) -> tuple:
    if distance <= config.ACCEPT_MAX_DISTANCE:
        return "accept", []
    if distance <= config.REVIEW_MAX_DISTANCE:
        return "review", ["borderline_distance"]
    return "reject", ["below_threshold"]


def verify(index, employee_id: str, probe: np.ndarray, others=()) -> Decision:
    """Score `probe` against the claimed employee and every other employee.

    `others` holds embeddings of any bystanders who wandered into the frame.
    They are not candidates for verification - the subject was already chosen as
    the largest face - but they are evidence: a bystander who matches the
    claimed employee BETTER than the subject does means the claimed person is in
    frame without being the one presenting, which is a deliberate attempt rather
    than an honest mismatch, and is worth rejecting under its own reason code.
    """
    if index.size == 0:
        return Decision(decision="reject", reasons=["not_enrolled"])

    claimed_mask = index.employee_ids == employee_id
    if not claimed_mask.any():
        return Decision(
            decision="reject",
            reasons=["not_enrolled"],
            candidates_checked=index.size,
        )

    all_distances = engine.distances(index.matrix, probe)

    # Best of the employee's own templates.  Enrolling more photos lowers this
    # distance - but it also lowers the runner-up distance for everyone probing
    # against this employee, so the net effect on the margin is a question for
    # tests/calibrate.py, not an assumption.
    distance = float(all_distances[claimed_mask].min())

    other_distances = all_distances[~claimed_mask]
    runner_up_distance = None
    runner_up_id = None
    margin = None
    if other_distances.size:
        best_other = int(other_distances.argmin())
        runner_up_distance = float(other_distances[best_other])
        runner_up_id = str(index.employee_ids[~claimed_mask][best_other])
        margin = runner_up_distance - distance

    decision, reasons = _band(distance)

    if others:
        reasons.append("extra_faces_present")
        claimed_templates = index.matrix[claimed_mask]
        for bystander in others:
            if float(engine.distances(claimed_templates, bystander).min()) < distance:
                decision = "reject"
                reasons.append("claimed_face_is_secondary")
                break

    if config.CROSS_CHECK_ENABLED and margin is not None and decision != "reject":
        if margin <= 0:
            # Someone else's template is a closer match than the claimed one.
            decision = "reject"
            reasons.append("identity_mismatch")
        elif margin < config.MIN_IMPOSTOR_MARGIN:
            # Close enough to another employee that the claim is not safe alone.
            decision = "review"
            reasons.append("low_margin")

    return Decision(
        decision=decision,
        distance=distance,
        runner_up_distance=runner_up_distance,
        runner_up_id=runner_up_id,
        margin=margin,
        reasons=reasons,
        candidates_checked=index.size,
    )
