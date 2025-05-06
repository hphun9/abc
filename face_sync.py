import requests
import json
import os
from datetime import datetime, timedelta

CONFIG_PATH = "config.json"
FACE_DB_PATH = "face_db.json"
SYNC_API = "http://localhost:9002/office/camera/v1/attendance/facesync"
API_KEY = "ThisIsSecretKeyForIOT"
HEADERS = {"x-api-key": API_KEY}

def should_sync():
    if not os.path.exists(CONFIG_PATH):
        return True

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    last_sync = datetime.fromtimestamp(config.get("last_sync", 0))
    interval_days = config.get("sync_interval_days", 30)
    return datetime.now() - last_sync >= timedelta(days=interval_days)

def update_sync_time():
    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "last_sync": int(datetime.now().timestamp()),
            "sync_interval_days": 30
        }, f)

def sync_face_data(agent_id):
    if not should_sync():
        return False

    try:
        response = requests.post(SYNC_API, json={"agent_id": agent_id}, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        with open(FACE_DB_PATH, "w") as f:
            json.dump(data, f)

        update_sync_time()
        print("[SYNC] Face DB synced successfully.")
        return True
    except Exception as e:
        print(f"[SYNC] Error syncing face DB: {e}")
        return False
