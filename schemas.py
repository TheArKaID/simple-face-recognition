"""Request bodies.

Constraints are checked in the handlers rather than through Field(...) so the
module behaves the same under pydantic v1 and v2.
"""
from typing import List, Optional

from pydantic import BaseModel


class FaceComparisonRequest(BaseModel):
    """Legacy two-image payload used by /compare-fr.

    The DeepFace-only knobs (model_name, detector_backend, distance_metric) are
    gone.  An HRIS still sending them is unaffected: pydantic ignores unknown
    fields by default.
    """

    reference_image: str  # Base64 encoded image
    target_image: str     # Base64 encoded image
    threshold: Optional[float] = None


class EnrollRequest(BaseModel):
    """Called by the HRIS when an employee's profile photo is set or changed."""

    employee_id: str
    images: List[str]
    tenant_id: str = "default"
    replace: bool = True


class VerifyRequest(BaseModel):
    """Called by the HRIS at attendance time.

    `reference_image` is optional and exists only to ease migration: when the
    employee has no stored template yet, the profile photo sent alongside is
    enrolled on the spot.  Once every employee is enrolled the HRIS can stop
    sending it.
    """

    employee_id: str
    image: str
    tenant_id: str = "default"
    reference_image: Optional[str] = None
