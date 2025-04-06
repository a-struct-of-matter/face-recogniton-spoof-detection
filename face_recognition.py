from facenet_pytorch import MTCNN
import cv2
import numpy as np
from facenet import load_model, run_embeddings
import pickle

def recognize_faces(video_source, model_path, classifier_path):
    # Initialize MTCNN from facenet-pytorch
    face_detector = MTCNN(image_size=160, margin=20, post_process=False)

    # Load FaceNet model
    sess = load_model(model_path)

    # Load the trained classifier
    with open(classifier_path, 'rb') as file:
        classifier, label_encoder = pickle.load(file)

    # Open the video stream
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Detect faces using facenet-pytorch MTCNN
        boxes, _ = face_detector.detect(frame)

        if boxes is not None:
            for box in boxes:
                x, y, x2, y2 = map(int, box)
                face = frame[y:y2, x:x2]

                try:
                    # Preprocess the face
                    face_resized = cv2.resize(face, (160, 160))
                    face_normalized = face_resized.astype("float32") / 255.0

                    # Generate embeddings
                    embedding = run_embeddings(face_normalized, sess)

                    if embedding is not None and embedding.shape[0] == 512:
                        probabilities = classifier.predict_proba([embedding])
                        max_prob = max(probabilities[0])
                        predicted_label = np.argmax(probabilities, axis=1)[0]

                        # Recognize or classify as unknown
                        if max_prob > 0.8:
                            name = label_encoder.inverse_transform([predicted_label])[0]
                        else:
                            name = "Unknown"

                        # Draw bounding box and label
                        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                        label = f"{name} ({max_prob * 100:.2f}%)"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error processing face: {e}")

        # Display the frame with detections
        cv2.imshow("Face Recognition", frame)

        # Exit the video stream when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces(
        video_source='rtsp://admin:L2B81D42@192.168.137.110:554/cam/realmonitor?channel=1&subtype=1',
        model_path= r'model\20180402-114759.pb',
        classifier_path= r'classifier.pkl'
    )
#rtsp://admin:L2B81D42@192.168.29.177:554/cam/realmonitor?channel=1&subtype=1
#rtsp://smart_attendance:Attendance@321@192.168.29.23:554/stream000