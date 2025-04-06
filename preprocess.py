import cv2
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import Normalizer

# Load the model
model_path = r"D:\face_recon\model\20180402-114759.pb"
classifier_path = r"D:\face_recon\classifier.pkl"

# Load FaceNet model
def load_model(model_path):
    graph = tf.Graph()
    with graph.as_default():
        with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    return graph

# Load classifier
with open(classifier_path, 'rb') as file:
    model, class_names = pickle.load(file)

# Preprocess image
def preprocess(face):
    resized = cv2.resize(face, (160, 160))
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)

# Extract embeddings
# Extract embeddings with phase_train
def get_embedding(face, session, images_placeholder, embeddings):
    preprocessed_face = preprocess(face)

    # Include phase_train in the feed dictionary
    phase_train_placeholder = session.graph.get_tensor_by_name('phase_train:0')

    return session.run(embeddings, feed_dict={
        images_placeholder: preprocessed_face,
        phase_train_placeholder: False  # Set to False for inference mode
    })[0]


# Initialize model
graph = load_model(model_path)
session = tf.compat.v1.Session(graph=graph)
images_placeholder = graph.get_tensor_by_name('input:0')
embeddings = graph.get_tensor_by_name('embeddings:0')

# Face detection setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        emb = get_embedding(face_img, session, images_placeholder, embeddings)

        # Normalize the embedding
        emb = Normalizer(norm='l2').transform([emb])
        prediction = model.predict(emb)
        name = class_names.inverse_transform(prediction)[0]


        # Display results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
