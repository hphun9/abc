import cv2

def capture_image_from_ip(ip_url):
    cap = cv2.VideoCapture(ip_url)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        raise Exception("Failed to capture frame")
