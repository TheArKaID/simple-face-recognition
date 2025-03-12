import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
from PIL import Image
import io
from deepface import DeepFace
import tensorflow as tf
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

# Pre-build and store the model globally to avoid rebuilding in each request
_ = DeepFace.build_model("VGG-Face")

@app.post("/compare-fr")
async def compare_faces(
    reference_file: UploadFile = File(...),
    target_file: UploadFile = File(...)
):
    try:
        # Load the reference image from the request
        ref_data = await reference_file.read()
        try:
            profile_image = Image.open(io.BytesIO(ref_data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid reference image file")
        profile_array = np.array(profile_image)
        profile_encodings = face_recognition.face_encodings(profile_array)
        if not profile_encodings:
            raise HTTPException(status_code=400, detail="No face found in reference image")
        profile_encoding = profile_encodings[0]

        # Load the target image from the request
        curr_data = await target_file.read()
        try:
            current_image = Image.open(io.BytesIO(curr_data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid target image file")
        current_array = np.array(current_image)
        current_encodings = face_recognition.face_encodings(current_array)
        if not current_encodings:
            raise HTTPException(status_code=400, detail="No face found in target image")
        current_encoding = current_encodings[0]

        # Compare the face from the reference to the target face
        match_results = face_recognition.compare_faces([profile_encoding], current_encoding)
        face_distance = face_recognition.face_distance([profile_encoding], current_encoding)[0]

        # Return uniform JSON response
        return {
            "status": "success",
            "message": "Face recognition successful",
            "data": {
                "match": bool(match_results[0]),
                "distance": float(face_distance)
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
    reference_file: UploadFile = File(...),
    target_file: UploadFile = File(...),
    model_name: str = Form("VGG-Face"),         # Options include VGG-Face, Facenet, OpenFace, DeepFace, etc.
    detector_backend: str = Form("dlib"),       # Options include opencv, mtcnn, etc.
    distance_metric: str = Form("cosine"),        # Options include cosine, euclidean, euclidean_l2
    threshold: float = Form(None)                 # Optional: custom threshold value, e.g., 0.4 or 10
):
    try:
        # Load and convert the reference image
        try:
            ref_bytes = await reference_file.read()
            profile_image = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid reference image file")
        
        # Load and convert the target image
        try:
            curr_bytes = await target_file.read()
            current_image = Image.open(io.BytesIO(curr_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid target image file")
        
        # Convert images to numpy arrays
        profile_np = np.array(profile_image)
        current_np = np.array(current_image)
        
        # Prepare parameters including the pre-built model to bypass repeated building
        verify_params = {
            "img1_path": profile_np,
            "img2_path": current_np,
            "model_name": model_name,
            "detector_backend": detector_backend,
            "distance_metric": distance_metric,
            "enforce_detection": True
        }
        
        if threshold is not None:
            verify_params["threshold"] = threshold

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

_ = verify_faces("dummy.jpg", "dummy.jpg")