"""Should this accept's probe be folded back into the employee's template?

Two independent gates, both required.  `eligible()` asks whether the DECISION
was clean enough to trust; `due()` asks whether enough TIME has passed to make
another update worth the drift it costs.  Kept separate because the caller
needs only `due()` to hit the database, and only when `eligible()` already
said yes - checking a timestamp for a decision that will be thrown out anyway
is one avoidable read per rejected attempt.

Nothing here writes anything.  The caller does that, through the same
store.enroll() that POST /enroll already uses - this module only says when.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import config


def eligible(decision) -> bool:
    """Is this Decision clean enough to let its probe influence future ones?

    Three conditions, all deliberately stricter than what "accept" alone
    requires - because this probe will not just decide today's attendance, it
    will become evidence future probes are judged against.

      - decision == "accept", never "review".  review exists precisely because
        the match was not confident enough to stand alone; letting it feed the
        template would let the least trustworthy accepts drive drift fastest.
      - reasons is empty.  A clean accept carries none; "extra_faces_present"
        means a bystander was in frame, a more complex scene than a routine
        clock-in, so it is excluded even though it does not change the
        decision.
      - margin comfortably clears MIN_IMPOSTOR_MARGIN, not merely enough to
        avoid "low_margin".  A margin that just barely avoids the review band
        is exactly the case template drift concentrates in.
    """
    if not config.AUTO_UPDATE_ENABLED:
        return False
    if decision.decision != "accept" or decision.reasons:
        return False
    if decision.margin is None or decision.margin < config.AUTO_UPDATE_MIN_MARGIN:
        return False
    return True


def due(last_update_iso: Optional[str]) -> bool:
    """Has it been long enough since this employee's newest template?

    `last_update_iso` is MAX(created_at) over the employee's current templates.
    It should never be None here in practice - eligible() only returns True
    after matcher.verify() found a claimed-employee match, which means at least
    one template already exists - but a missing or unparsable timestamp fails
    towards NOT updating rather than updating on every call, since the cost of
    one missed refresh is far smaller than the cost of an ungated one.
    """
    if not last_update_iso:
        return False
    try:
        last = datetime.fromisoformat(last_update_iso)
    except ValueError:
        return False
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - last).total_seconds() / 86400.0
    return age_days >= config.AUTO_UPDATE_INTERVAL_DAYS
