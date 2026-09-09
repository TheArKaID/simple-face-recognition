from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
import os
from deepface import DeepFace
import tensorflow as tf

import config
import engine
import matcher
from schemas import EnrollRequest, FaceComparisonRequest, VerifyRequest
from store import TemplateStore

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Configure memory growth to avoid TensorFlow taking all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
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

# Helper function to decode base64 to image
def decode_base64_to_image(base64_string):
    return engine.decode_base64_image(base64_string)

# Function to pre-load DeepFace models using local test images
def preload_deepface_model():
    print("Pre-loading DeepFace models...")
    try:
        # Build the model explicitly
        _ = DeepFace.build_model("VGG-Face")

        # Optionally, perform a test verification with dummy images
        # Get the current directory where main.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dummy1_path = os.path.join(current_dir, "dummy1.jpg")
        dummy2_path = os.path.join(current_dir, "dummy2.jpg")

        # Check if the dummy files exist
        if os.path.exists(dummy1_path) and os.path.exists(dummy2_path):
            print(f"Using test images: {dummy1_path} and {dummy2_path}")

            # Perform test verification to ensure everything is loaded
            result = DeepFace.verify(
                img1_path=dummy1_path,
                img2_path=dummy2_path,
                model_name="VGG-Face",
                detector_backend="dlib",
                distance_metric="cosine",
                enforce_detection=True
            )
            print("Model pre-loading complete with test verification")
        else:
            print("Dummy image files not found, model built without verification test")
    except Exception as e:
        print(f"Error pre-loading model: {e}")

# Call the preload function at startup
preload_deepface_model()

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
    decision = matcher.verify(index, request.employee_id, probe.embedding)

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
        quality=probe.quality(),
    )

    if "not_enrolled" in decision.reasons:
        return _error(
            409,
            "Employee has no enrolled face template",
            "not_enrolled",
            f"Call POST /enroll for {request.employee_id} first",
        )

    data = decision.public_data()
    data["quality"] = probe.quality()
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
    face_recognition's 0.6 default, and an image containing more than one face
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

        # Logged so the calibration set starts filling up before the HRIS moves
        # to /verify; these rows are what the thresholds get retuned against.
        store.log_verification(
            source="legacy",
            decision="match" if matched else "no_match",
            distance=face_distance,
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
                "quality": current.quality(),
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Face recognition failed",
            "errors": str(e)
        }

@app.post("/compare-df")
async def verify_faces(
    request: FaceComparisonRequest
):
    try:
        # Decode the reference image
        try:
            profile_image = decode_base64_to_image(request.reference_image)
        except Exception as e:
            return {
                "status": "error",
                "message": "Invalid reference image",
                "errors": str(e)
            }

        # Decode the target image
        try:
            current_image = decode_base64_to_image(request.target_image)
        except Exception as e:
            return {
                "status": "error",
                "message": "Invalid target image",
                "errors": str(e)
            }
        
        # Convert images to numpy arrays
        profile_np = np.array(profile_image)
        current_np = np.array(current_image)

        # Prepare parameters including the pre-built model to bypass repeated building
        verify_params = {
            "img1_path": profile_np,
            "img2_path": current_np,
            "model_name": request.model_name,
            "detector_backend": request.detector_backend,
            "distance_metric": request.distance_metric,
            "enforce_detection": True
        }

        if request.threshold is not None:
            verify_params["threshold"] = request.threshold

        result = DeepFace.verify(**verify_params)
        # Return uniform JSON response
        return {
            "status": "success",
            "message": "Face verification successful",
            "data": {
                "match": bool(result.get("verified")),
                "distance": float(result.get("distance"))
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Face verification failed",
            "errors": str(e)
        }
