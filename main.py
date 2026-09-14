from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import auto_update
import config
import engine
import liveness
import matcher
from schemas import EnrollRequest, FaceComparisonRequest, VerifyRequest
from store import TemplateStore

app = FastAPI()

store = TemplateStore()
print(f"Template store ready at {store.path}: {store.stats()}")

# Add custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = []
    for error in exc.errors():
        error_msg = error.get("msg", "")
        error_loc = " -> ".join(str(loc) for loc in error.get("loc", []))
        errors.append(f"{error_loc}: {error_msg}")

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": errors
        }
    )

def _error(status_code, message, reason, detail=None):
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": message,
            "reason": reason,
            "errors": detail or message,
        },
    )


@app.get("/health")
def health():
    return {
        "status": "success",
        "message": "Service healthy",
        "data": {
            "store": store.stats(),
            "liveness": {
                "mode": config.LIVENESS_MODE,
                "available": liveness.available(),
                "min_score": config.LIVENESS_MIN_SCORE,
                "required": config.LIVENESS_REQUIRED,
            },
            "max_templates_per_employee": config.MAX_TEMPLATES_PER_EMPLOYEE,
            "auto_update": {
                "enabled": config.AUTO_UPDATE_ENABLED,
                "interval_days": config.AUTO_UPDATE_INTERVAL_DAYS,
                "min_margin": config.AUTO_UPDATE_MIN_MARGIN,
            },
            "thresholds": {
                "accept_max_distance": config.ACCEPT_MAX_DISTANCE,
                "review_max_distance": config.REVIEW_MAX_DISTANCE,
                "min_impostor_margin": config.MIN_IMPOSTOR_MARGIN,
                "cross_check_enabled": config.CROSS_CHECK_ENABLED,
                "legacy_tolerance": config.LEGACY_TOLERANCE,
            },
        },
    }


@app.post("/enroll")
def enroll(request: EnrollRequest):
    """Store face templates for one employee.

    Called when a profile photo is uploaded or changed.  Several photos taken
    under different angles and lighting widen the gap between genuine and
    impostor scores, so the HRIS should send more than one where it has them.
    """
    if not request.images:
        return _error(422, "No images supplied", "no_images")

    accepted = []
    rejected = []
    for position, image_b64 in enumerate(request.images):
        try:
            result = engine.embed_base64(image_b64)
            accepted.append((result.embedding, result.quality()))
        except engine.FaceError as exc:
            rejected.append({"index": position, "reason": exc.reason, "detail": exc.detail})

    if not accepted:
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "message": "No usable face found in any supplied image",
                "reason": rejected[0]["reason"] if rejected else "no_face",
                "errors": rejected,
            },
        )

    stored = store.enroll(
        request.tenant_id, request.employee_id, accepted, replace=request.replace
    )
    return {
        "status": "success",
        "message": "Enrollment successful",
        "data": {
            "tenant_id": request.tenant_id,
            "employee_id": request.employee_id,
            "templates_stored": stored,
            "rejected": rejected,
            "engine_id": engine.ENGINE_ID,
        },
    }


@app.get("/enroll/{tenant_id}/{employee_id}")
def enrollment_status(tenant_id: str, employee_id: str):
    count = store.template_count(tenant_id, employee_id)
    return {
        "status": "success",
        "message": "Enrollment status",
        "data": {
            "tenant_id": tenant_id,
            "employee_id": employee_id,
            "enrolled": count > 0,
            "templates_stored": count,
            "engine_id": engine.ENGINE_ID,
        },
    }


@app.delete("/enroll/{tenant_id}/{employee_id}")
def delete_enrollment(tenant_id: str, employee_id: str):
    """Remove an employee's templates.

    Biometric data counts as sensitive personal data under UU PDP, so the HRIS
    must call this when an employee leaves or is deleted.
    """
    removed = store.delete_employee(tenant_id, employee_id)
    return {
        "status": "success",
        "message": "Enrollment deleted",
        "data": {
            "tenant_id": tenant_id,
            "employee_id": employee_id,
            "templates_removed": removed,
        },
    }


@app.post("/verify")
def verify(request: VerifyRequest):
    """Verify an attendance photo against the claimed employee.

    The probe is scored against every enrolled employee in the tenant, not just
    the claimed one, so a lookalike colleague fails even when the 1:1 distance
    alone would have passed.
    """
    try:
        probe = engine.embed_base64(request.image)
    except engine.FaceError as exc:
        store.log_verification(
            tenant_id=request.tenant_id,
            employee_id=request.employee_id,
            source="verify",
            decision="reject",
            reasons=[exc.reason],
        )
        return _error(422, "Attendance image unusable", exc.reason, exc.detail)

    # Liveness before identity: a photo of the right person is still a photo.
    live = liveness.check(probe.image, probe.bbox)
    if config.LIVENESS_MODE == "model":
        if live is None:
            if config.LIVENESS_REQUIRED:
                store.log_verification(
                    tenant_id=request.tenant_id, employee_id=request.employee_id,
                    source="verify", decision="reject", reasons=["liveness_unavailable"],
                )
                return _error(
                    503, "Liveness check is enabled but unavailable",
                    "liveness_unavailable",
                    "No usable weights in FACE_LIVENESS_MODEL_DIR",
                )
        elif not live.is_live:
            store.log_verification(
                tenant_id=request.tenant_id, employee_id=request.employee_id,
                source="verify", decision="reject", reasons=["spoof_suspected"],
                quality={**probe.quality(), "liveness": live.public_data()},
            )
            return _error(
                422, "Liveness check failed", "spoof_suspected",
                f"Liveness score {live.score:.3f} below {config.LIVENESS_MIN_SCORE}",
            )

    # Migration convenience: enrol from the profile photo on first sight.
    if request.reference_image and store.template_count(
        request.tenant_id, request.employee_id
    ) == 0:
        try:
            reference = engine.embed_base64(request.reference_image)
            store.enroll(
                request.tenant_id,
                request.employee_id,
                [(reference.embedding, reference.quality())],
            )
        except engine.FaceError as exc:
            return _error(422, "Reference image unusable", f"reference_{exc.reason}", exc.detail)

    index = store.index(request.tenant_id)
    decision = matcher.verify(index, request.employee_id, probe.embedding, probe.others)

    store.log_verification(
        tenant_id=request.tenant_id,
        employee_id=request.employee_id,
        source="verify",
        decision=decision.decision,
        distance=decision.distance,
        runner_up_id=decision.runner_up_id,
        runner_up_distance=decision.runner_up_distance,
        margin=decision.margin,
        reasons=decision.reasons,
        quality={**probe.quality(),
                 **({"liveness": live.public_data()} if live else {})},
    )

    if "not_enrolled" in decision.reasons:
        return _error(
            409,
            "Employee has no enrolled face template",
            "not_enrolled",
            f"Call POST /enroll for {request.employee_id} first",
        )

    # Auto re-enrolment: fold this probe into the template set if the accept
    # was clean and enough time has passed.  Never affects the response - a
    # skipped or failed refresh is not a verification failure; see
    # auto_update.py and store.auto_update_template() for why each gate
    # exists.
    if auto_update.eligible(decision):
        last = store.last_template_update(request.tenant_id, request.employee_id)
        if auto_update.due(last):
            store.auto_update_template(
                request.tenant_id, request.employee_id,
                probe.embedding, probe.quality(),
            )

    data = decision.public_data()
    data["quality"] = probe.quality()
    data["liveness"] = live.public_data() if live else None
    return {
        "status": "success",
        "message": "Face verification successful",
        "data": data,
    }


@app.post("/compare-fr")
def compare_faces(
        request: FaceComparisonRequest
    ):
    """Legacy 1:1 comparison kept alive while the HRIS migrates to /verify.

    Response shape is unchanged; `data` only gains fields.  Behaviour differs
    from before in two ways: the tolerance is configurable and stricter than
    the 0.6 default the endpoint shipped with, and an image containing more than one face
    is refused instead of having the first detection picked arbitrarily.
    """
    try:
        gates = config.LEGACY_QUALITY_GATES
        try:
            profile = engine.embed_base64(request.reference_image, quality_gates=gates)
        except engine.FaceError as exc:
            store.log_verification(
                source="legacy", decision="error", reasons=[f"reference_{exc.reason}"]
            )
            return {
                "status": "error",
                "message": "Invalid reference image"
                if exc.reason == "invalid_image"
                else "No usable face in reference image",
                "reason": exc.reason,
                "errors": exc.detail,
            }

        try:
            current = engine.embed_base64(request.target_image, quality_gates=gates)
        except engine.FaceError as exc:
            store.log_verification(
                source="legacy", decision="error", reasons=[f"target_{exc.reason}"]
            )
            return {
                "status": "error",
                "message": "Invalid target image"
                if exc.reason == "invalid_image"
                else "No usable face in target image",
                "reason": exc.reason,
                "errors": exc.detail,
            }

        face_distance = engine.distance(profile.embedding, current.embedding)
        tolerance = (
            request.threshold if request.threshold is not None else config.LEGACY_TOLERANCE
        )
        matched = bool(face_distance <= tolerance)

        # Bystanders are tolerated in the frame, but the subject still has to be
        # the reference person.  With no roster to consult, the reference image
        # itself is the yardstick: a bystander closer to it than the subject is
        # means the reference person is in frame without being the one
        # presenting - someone holding up their photo, most likely.
        reason = None
        for bystander in current.others:
            if engine.distance(profile.embedding, bystander) < face_distance:
                matched = False
                reason = "claimed_face_is_secondary"
                break
        if current.others and reason is None:
            reason = "extra_faces_present"

        # Logged so the calibration set starts filling up before the HRIS moves
        # to /verify; these rows are what the thresholds get retuned against.
        store.log_verification(
            source="legacy",
            decision="match" if matched else "no_match",
            distance=face_distance,
            reasons=[reason] if reason else [],
            quality=current.quality(),
        )

        # Return uniform JSON response
        return {
            "status": "success",
            "message": "Face recognition successful",
            "data": {
                "match": matched,
                "distance": float(face_distance),
                "tolerance": float(tolerance),
                "reason": reason,
                "quality": current.quality(),
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Face recognition failed",
            "errors": str(e)
        }
