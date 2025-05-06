import requests
import time

API_ENDPOINT = "http://localhost:9002"
API_DETECTION_ENPOINT = API_ENDPOINT + "/office/camera/v1/detections/report"
API_REGCONITION_ENDPOINT = API_ENDPOINT+ "/office/camera/v1/attendance/report"
API_KEY = "ThisIsSecretKeyForIOT"

MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds

def send_detection(room_id, camera_id, people_count, timestamp):
    payload = {
        "room_id": room_id,
        "camera_id": camera_id,
        "timestamp": int(timestamp.timestamp() * 1000),
        "people_count": people_count
    }

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_DETECTION_ENPOINT, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            print(f"[OK] Sent data for camera {camera_id}: {people_count} people")
            return
        except Exception as e:
            print(f"[RETRY {attempt + 1}] Failed sending data: {e}")
            time.sleep(RETRY_DELAY)

    print(f"[FAILED] All retries failed for camera {camera_id}")

def send_attendance(username, camera_id, timestamp):
    url = API_REGCONITION_ENDPOINT
    payload = {
        "username": username,
        "camera_id": camera_id,
        "timestamp": int(timestamp.timestamp() * 1000)
    }

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            print(f"[OK] Sent attendance: {username} at camera {camera_id}")
            return
        except Exception as e:
            print(f"[RETRY {attempt + 1}] Failed to send attendance: {e}")
            time.sleep(RETRY_DELAY)

    print(f"[FAILED] All retries failed for attendance {username} - {camera_id}")
