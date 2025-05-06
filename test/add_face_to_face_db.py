import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# --- Config ---
camera_url = "rtsp://admin:L2CF2ED0@192.168.1.115:554/cam/realmonitor?channel=1&subtype=1"  # 0 = default webcam, or you can put RTSP url
username = "hphun9"  # your username to add
face_db_file = "face_db.json"

# --- Initialize models ---
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, thresholds=[0.6, 0.7, 0.7])
model = InceptionResnetV1(pretrained="vggface2").eval()

# --- Load existing face_db.json ---
face_db = []
if os.path.exists(face_db_file):
    if os.path.getsize(face_db_file) > 0:
        try:
            with open(face_db_file, "r") as f:
                face_db = json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load face_db.json, resetting: {e}")
            face_db = []
    else:
        print("[INFO] face_db.json is empty, starting fresh...")
else:
    print("[INFO] No existing face_db.json found, creating new one...")

# --- Helper functions ---
def get_embedding(face_img):
    if face_img is None:
        return None
    if face_img.ndim == 3:
        face_img = face_img.unsqueeze(0)
    with torch.no_grad():
        embedding = model(face_img).squeeze().numpy()
    return embedding

# --- Main ---
cap = cv2.VideoCapture(camera_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit(1)

print("[INFO] Camera opened. Press SPACE to capture face, or ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to capture frame, skipping...")
        continue

    show_frame = frame.copy()
    cv2.putText(show_frame, "Press SPACE to capture face", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Add Face", show_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key
        break

    if key == 32:  # SPACE key
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)

            face = mtcnn(img)
            if face is None:
                print("[WARNING] No face detected, try again!")
                continue

            embedding = get_embedding(face)
            if embedding is None:
                print("[ERROR] Failed to generate embedding")
                continue

            # Find user or create new
            user_entry = None
            for user in face_db:
                if user["username"] == username:
                    user_entry = user
                    break

            if user_entry is None:
                user_entry = {"username": username, "face_embeddings": []}
                face_db.append(user_entry)

            user_entry["face_embeddings"].append({
                "embedding": embedding.tolist()
            })

            # Save updated face_db.json
            with open(face_db_file, "w") as f:
                json.dump(face_db, f, indent=2)

            print(f"[INFO] Successfully added new face for user '{username}'")

        except Exception as e:
            print(f"[ERROR] {e}")

cap.release()
cv2.destroyAllWindows()
