import cv2
import numpy as np
import pickle
from facenet_pytorch import MTCNN
from ultralytics import YOLO
from facenet import load_model, run_embeddings  # Ensure these are implemented properly

# RTSP Stream URL
rtsp_url = "rtsp://smart_attendance:123456789@192.168.29.24:554/stream2"

# Optimize RTSP Capture with GStreamer (Low Latency)
gstreamer_pipeline = f"rtspsrc location={rtsp_url} latency=0 ! decodebin ! videoconvert ! appsink"
cap = cv2.VideoCapture(0)#gstreamer_pipeline, cv2.CAP_GSTREAMER))

# If GStreamer is not available, fallback to FFMPEG with reduced buffer
if not cap.isOpened():
    print("Error: GStreamer failed. Trying FFMPEG...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size

if not cap.isOpened():
    print("Error: Cannot open video source.")
    exit()

# Load YOLOv8 for object detection
yolo_model = YOLO("yolov8m.pt")  # COCO-trained model

# Initialize MTCNN for face detection
face_detector = MTCNN(image_size=160, margin=20, post_process=False)

# Load FaceNet model for face recognition
face_recognition_model = load_model(r"model\20180402-114759.pb")

# Load trained classifier for face recognition
with open(r"classifier.pkl", "rb") as file:
    classifier, label_encoder = pickle.load(file)

while cap.isOpened():
    # Drop old frames to keep the latest one (reduces lag)
    for _ in range(5):  # Adjust this based on latency
        ret, frame = cap.read()

    if not ret:
        break

    # Resize frame for faster YOLO processing
    small_frame = cv2.resize(frame, (640, 360))
    results = yolo_model(small_frame, agnostic_nms=True)  # Faster inference

    # Extract mobile phone detections (Class ID 67)
    phone_boxes = []
    for det in results[0].boxes:
        class_id = int(det.cls)

        if class_id == 67:  # Mobile Phone (ID 67 in COCO dataset)
            # Ensure tensor is on CPU and convert it properly
            x1, y1, x2, y2 = (det.xyxy[0].cpu().numpy() * np.array([
                frame.shape[1] / 640, frame.shape[0] / 360,
                frame.shape[1] / 640, frame.shape[0] / 360
            ])).astype(int)

            phone_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Mobile Phone", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Perform face detection using MTCNN
    boxes, _ = face_detector.detect(frame)

    spoof_detected = False  # Flag for spoofing detection

    if boxes is not None:
        for box in boxes:
            x, y, x2, y2 = map(int, box)
            face = frame[y:y2, x:x2]

            try:
                # Preprocess face
                face_resized = cv2.resize(face, (160, 160))
                face_normalized = face_resized.astype("float32") / 255.0

                # Generate embeddings using FaceNet
                embedding = run_embeddings(face_normalized, face_recognition_model)

                if embedding is not None and embedding.shape[0] == 512:
                    probabilities = classifier.predict_proba([embedding])
                    max_prob = max(probabilities[0])
                    predicted_label = np.argmax(probabilities, axis=1)[0]

                    # Recognize or classify as unknown
                    if max_prob > 0.8:
                        name = label_encoder.inverse_transform([predicted_label])[0]
                    else:
                        name = "Unknown"

                    # Draw face bounding box and label
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    label = f"{name} ({max_prob * 100:.2f}%)"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Check for spoofing: If any mobile phone is near the detected face
                    for px1, py1, px2, py2 in phone_boxes:
                        if (
                            px1 < x2 and px2 > x and  # Horizontal overlap
                            py1 < y2 and py2 > y  # Vertical overlap
                        ):
                            spoof_detected = True

            except Exception as e:
                print(f"Error processing face: {e}")

    # Display "Spoofing Attempt!" if a phone is detected near a face
    if spoof_detected:
        cv2.putText(frame, "Spoofing Attempt!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Display frame with both detections
    cv2.imshow("YOLOv8 + Face Recognition + Spoof Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
