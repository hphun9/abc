import cv2
import json
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# --- Config ---
camera_url = "rtsp://admin:L2CF2ED0@192.168.1.115:554/cam/realmonitor?channel=1&subtype=1&transport=tcp"
threshold = 0.6  # recognition threshold (lower = stricter)

# --- Load face database ---
try:
    with open("face_db.json", "r") as f:
        face_db = json.load(f)
except Exception as e:
    print(f"[ERROR] Failed to load face_db.json: {e}")
    exit(1)

# --- Initialize models ---
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, thresholds=[0.6, 0.7, 0.7])
model = InceptionResnetV1(pretrained="vggface2").eval()

# --- Helper functions ---
def get_embedding(face_img):
    if face_img is None:
        return None
    if face_img.ndim == 3:
        face_img = face_img.unsqueeze(0)
    with torch.no_grad():
        embedding = model(face_img).squeeze().numpy()
    return embedding

def recognize_user(embedding):
    if embedding is None:
        return None, float('inf')

    best_match = None
    best_distance = float('inf')

    for user in face_db:
        for known_face in user.get("face_embeddings", []):
            known_embedding = np.array(known_face["embedding"])
            distance = np.linalg.norm(known_embedding - embedding)

            if distance < best_distance:
                best_match = user["username"]
                best_distance = distance

    return best_match, best_distance

# --- Main ---
cap = cv2.VideoCapture(camera_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency

if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit(1)

print("[INFO] Camera opened, starting real-time recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Frame capture failed, skipping...")
        continue  # skip to next frame safely

    # Optional: resize for speed
    # frame = cv2.resize(frame, (640, 480))

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        boxes, _ = mtcnn.detect(img)

        if boxes is not None:
            faces = mtcnn(img)

            if faces is not None:
                for box, face in zip(boxes, faces):
                    embedding = get_embedding(face)
                    username, distance = recognize_user(embedding)

                    if username and distance < threshold:
                        label = f"{username} ({distance:.2f})"
                    else:
                        label = "Unknown"

                    # Draw box
                    box = [int(b) for b in box]
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                    # Draw label
                    cv2.putText(frame, label, (box[0], box[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except Exception as e:
        print(f"[ERROR] Recognition error: {e}")
        continue

    # Show the frame
    cv2.imshow("Face Recognition Test", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
