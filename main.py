
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import face_recognition
import numpy as np
from PIL import Image
import io
from deepface import DeepFace

app = FastAPI()

@app.post("/compare-fr")
async def compare_faces(
    profile: UploadFile = File(...),
    current: UploadFile = File(...)
):
    # Load the profile image from the request
    profile_data = await profile.read()
    try:
        profile_image = Image.open(io.BytesIO(profile_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid profile image file")
    profile_array = np.array(profile_image)
    profile_encodings = face_recognition.face_encodings(profile_array)
    if not profile_encodings:
        raise HTTPException(status_code=400, detail="No face found in profile image")
    profile_encoding = profile_encodings[0]

    # Load the current image from the request
    current_data = await current.read()
    try:
        current_image = Image.open(io.BytesIO(current_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid current image file")
    current_array = np.array(current_image)
    current_encodings = face_recognition.face_encodings(current_array)
    if not current_encodings:
        raise HTTPException(status_code=400, detail="No face found in current image")
    current_encoding = current_encodings[0]

    # Compare the face from the profile to the current face
    # match_results = face_recognition.compare_faces([profile_encoding], current_encoding)
    # face_distance = face_recognition.face_distance([profile_encoding], current_encoding)[0]

    # return {"match": match_results[0], "distance": face_distance}
    match_results = face_recognition.compare_faces([profile_encoding], current_encoding)
    face_distance = face_recognition.face_distance([profile_encoding], current_encoding)[0]

    # Convert to native Python types
    return {
        "match": bool(match_results[0]),
        "distance": float(face_distance)
    }

@app.post("/compare-df")
async def verify_faces(
    profile: UploadFile = File(...),
    current: UploadFile = File(...),
    model_name: str = Form("VGG-Face"),         # Options include VGG-Face, Facenet, OpenFace, DeepFace, etc.
    detector_backend: str = Form("opencv"),       # Options include opencv, mtcnn, etc.
    distance_metric: str = Form("cosine"),        # Options include cosine, euclidean, euclidean_l2
    threshold: float = Form(None)                 # Optional: custom threshold value, e.g., 0.4 or 10
):
    # Load and convert the profile image
    try:
        profile_bytes = await profile.read()
        profile_image = Image.open(io.BytesIO(profile_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid profile image file")
    
    # Load and convert the current image
    try:
        current_bytes = await current.read()
        current_image = Image.open(io.BytesIO(current_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid current image file")
    
    # Convert images to numpy arrays (DeepFace accepts both file paths and numpy arrays)
    profile_np = np.array(profile_image)
    current_np = np.array(current_image)
    
    # Prepare parameters for DeepFace.verify
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

    try:
        result = DeepFace.verify(**verify_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return result
