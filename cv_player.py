from ultralytics import YOLO
import cv2
import cvzone
import math
from deep_sort_realtime.deepsort_tracker import DeepSort

# Install DeepSORT: pip3 install deep_sort_realtime
# Install PyTorch (with or without CUDA): https://pytorch.org/get-started/locally/

def start_cv_player(config):
    unique_ids = set()

    cv2.namedWindow("main", cv2.WINDOW_NORMAL)

    if config["video_source"] == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(config["video_source"])

    frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    tracker = DeepSort(max_age=frames_per_second)
    cv2.resizeWindow('main', frame_width, frame_height)

    model = YOLO(config["yolo_model"])
    selected_device = 0 if config["run_on_gpu"] else "cpu"

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True, device=selected_device)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if cls == 0 and conf >= 0.65:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue

            unique_ids.add(track.track_id)
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(img, (x1, y1, w, h))

            cvzone.putTextRect(img, f'ID: {track_id}', (x1, max(30, y1 - 10)), scale=1, thickness=1)

        cvzone.putTextRect(img, f"Detected {len(unique_ids)} unique people", (15, 25), scale=0.75, thickness=1)

        cv2.imshow("main", img)
        if cv2.waitKey(frames_per_second) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()