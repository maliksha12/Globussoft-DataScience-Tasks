from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import torch
import io


# 1. Create FastAPI app

app = FastAPI(
    title="Face Verification API (FaceNet)",
    description="Verify whether two face images belong to the same person using FaceNet embeddings.",
    version="1.0.0",
)


# 2. Load models (device, MTCNN, FaceNet)

device = torch.device("cpu")  # change to "cuda" if you have GPU + CUDA

# MTCNN: face detection + alignment
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)

# FaceNet model (InceptionResnetV1) for embeddings
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)



# 3. Utility: read uploaded image as PIL RGB

def read_image_from_upload(file: UploadFile) -> Image.Image:      # Read an uploaded file into a PIL RGB image.
    
    image_bytes = file.file.read()
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")



# 4. Utility: detect faces and get embeddings

def get_face_boxes_and_embeddings(img: Image.Image):    
    
    boxes, probs = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        return [], None, []

    # Extract aligned face tensors for each detected box 
    face_tensors = mtcnn.extract(img, boxes, None)  # (N, 3, 160, 160)

    with torch.no_grad():
        face_tensors = face_tensors.to(device)
        embeddings = resnet(face_tensors)  # (N, 512)

    return boxes.tolist(), embeddings, probs.tolist()



# 5. Utility: get the "main" face embedding (highest prob)

def get_main_face_embedding(img: Image.Image):
   
    boxes, embeddings, probs = get_face_boxes_and_embeddings(img)

    if embeddings is None or len(boxes) == 0:
        return None, [], []

    probs_np = np.array(probs)
    best_idx = int(probs_np.argmax())

    best_emb = embeddings[best_idx]
    best_emb_norm = best_emb / best_emb.norm(p=2)  # L2-normalize

    return best_emb_norm, boxes, probs



# 6. Utility: cosine similarity between two embeddings

def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
   
    return float(torch.dot(emb1, emb2).item())



# 7. API endpoint: /verify-faces

@app.post("/verify-faces")
async def verify_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    
    # Basic file type check
    valid_types = {"image/jpeg", "image/png", "image/jpg"}
    if image1.content_type not in valid_types:
        raise HTTPException(status_code=400, detail="image1 must be JPEG or PNG")
    if image2.content_type not in valid_types:
        raise HTTPException(status_code=400, detail="image2 must be JPEG or PNG")

    # Read both images
    img1 = read_image_from_upload(image1)
    img2 = read_image_from_upload(image2)

    # Get main face embedding + all boxes
    emb1, boxes1, probs1 = get_main_face_embedding(img1)
    emb2, boxes2, probs2 = get_main_face_embedding(img2)

    if emb1 is None:
        raise HTTPException(status_code=400, detail="No face detected in image1")
    if emb2 is None:
        raise HTTPException(status_code=400, detail="No face detected in image2")

    # Similarity and decision
    similarity = compute_cosine_similarity(emb1, emb2)

    # Threshold for deciding "same person"
    THRESHOLD = 0.7  # you can tune this value

    if similarity >= THRESHOLD:
        result = "Same person"
    else:
        result = "Different person"

    # Build response
    response_data = {
        "verification_result": result,
        "similarity_score": similarity,
        "image1": {
            "num_faces": len(boxes1),
            "bounding_boxes": boxes1,
            "detection_probs": probs1,
        },
        "image2": {
            "num_faces": len(boxes2),
            "bounding_boxes": boxes2,
            "detection_probs": probs2,
        },
        "threshold_used": THRESHOLD,
        "model": "FaceNet (InceptionResnetV1, vggface2)",
    }

    return JSONResponse(content=response_data)



# (Optional) Simple root endpoint

@app.get("/")
def root():
    return {
        "message": "Face Verification API is running.",
        "usage": "Go to /docs to test the /verify-faces endpoint.",
    }
