import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from facenet import load_model, run_embeddings

def get_embeddings_from_images(image_dir, model_path):
    sess = load_model(model_path)
    embeddings = []
    labels = []

    for subdir, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith((".jpg", ".png")):
                image_path = os.path.join(subdir, file)
                print(f"Processing image: {image_path}")
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                try:
                    image_resized = cv2.resize(image, (160, 160))
                    image_preprocessed = image_resized.astype("float32") / 255.0

                    embedding = run_embeddings(image_preprocessed, sess)
                    if embedding is not None:
                        embeddings.append(embedding)
                        label = os.path.basename(subdir)
                        labels.append(label)
                    else:
                        print(f"Failed to generate embedding for {image_path}")
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

    return np.array(embeddings), np.array(labels)

def train_classifier(image_dir, model_path, classifier_output_path):
    embeddings, labels = get_embeddings_from_images(image_dir, model_path)
    if len(embeddings) == 0:
        print("No embeddings found. Exiting...")
        return

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(embeddings, labels_encoded)

    with open(classifier_output_path, 'wb') as file:
        pickle.dump((clf, label_encoder), file)
    print("Classifier saved successfully.")

if __name__ == "__main__":
    train_classifier('./aligned_images', './model/20180402-114759.pb', './classifier.pkl')
