import numpy as np
import json
import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

FACE_DB_PATH = "face_db.json"

def load_face_db():
    if not os.path.exists(FACE_DB_PATH):
        return []
    with open(FACE_DB_PATH, "r") as f:
        return json.load(f)

# Initialize once
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, thresholds=[0.6, 0.7, 0.7])
model = InceptionResnetV1(pretrained='vggface2').eval()

def extract_embedding(frame):
    """Extract FaceNet embedding from a frame"""
    try:
        img = Image.fromarray(frame[:, :, ::-1])  # BGR to RGB
        face = mtcnn(img)
        if face is None:
            print("[FaceRecog] No face detected")
            return None
        if face.ndim == 3:
            face = face.unsqueeze(0)
        with torch.no_grad():
            embedding = model(face).squeeze().numpy()
        return embedding
    except Exception as e:
        print(f"[FaceRecog] Error extracting embedding: {e}")
        return None

def recognize_user(frame):
    """Recognize user from frame"""
    try:
        embedding = extract_embedding(frame)
        if embedding is None:
            print("[FaceRecog] No embedding extracted, skip recognition")
            return None

        face_db = load_face_db()

        for user in face_db:
            for known_face in user.get("face_embeddings", []):
                known_embedding = np.array(known_face["embedding"])
                distance = np.linalg.norm(known_embedding - embedding)
                print(f"[DEBUG] Distance to {user['username']}: {distance}")
                if distance < 0.8:  # Typical threshold for facenet
                    return user["username"]

        return None
    except Exception as e:
        print(f"[FaceRecog] Error during recognition: {e}")
        return None
