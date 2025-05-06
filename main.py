import logging
import time
import json
import cv2
from datetime import datetime, timezone
from yolov8_inference import count_people
from utils import capture_image_from_ip
from api_client import send_detection, send_attendance
from face_sync import sync_face_data
from face_recognition_utils import recognize_user

AGENT_ID = "edge-agent"

# Setup logging
logging.basicConfig(
    filename="logs/edge_agent.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def main():
    # Đồng bộ face DB định kỳ nếu đến hạn
    sync_face_data(AGENT_ID)

    # Đếm người ở các camera được khai báo (loại trừ camera nhận diện nếu có flag riêng)
    try:
        with open("cameras.json") as f:
            cameras = json.load(f)
    except Exception as e:
        logging.error(f"Could not load cameras.json: {e}")
        return

    for cam in cameras:
        logging.info(f"Processing camera: {cam['name']}")
        try:
            frame = capture_image_from_ip(cam["ip"])

            if cam.get("type") == "recognition":
                logging.info(f"Starting real-time face recognition for {cam['name']}")
                cap = cv2.VideoCapture(cam["ip"])

                if not cap.isOpened():
                    logging.error(f"Failed to open camera {cam['name']}")
                    continue

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logging.warning(f"Failed to read frame from camera {cam['name']}")
                        break

                    username = recognize_user(frame)
                    print(username)

                    if username:
                        send_attendance(
                            username=username,
                            camera_id=cam["camera_id"],
                            timestamp=datetime.now(timezone.utc)
                        )
                        logging.info(f"Recognized {username} on camera {cam['name']}")
                        break  # stop after match

                    time.sleep(0.5)  # 500ms delay between frames

                cap.release()

            else:
                people_count = count_people(frame)
                logging.info(f"People detect in camera {cam['name']}: {people_count}")

                # Gửi dữ liệu đếm về backend
                send_detection(
                    room_id=cam["room_id"],
                    camera_id=cam["camera_id"],
                    people_count=people_count,
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            logging.error(f"Failed processing {cam['name']}: {e}")

if __name__ == "__main__":
    while True:
        logging.info("Job started")
        main()
        logging.info("Job done, sleeping 15 minutes")
        time.sleep(15 * 60)