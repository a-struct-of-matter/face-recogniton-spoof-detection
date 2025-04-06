import pickle
from sklearn.svm import SVC
import numpy as np


def train_classifier(embeddings, labels, output_path):
    """
    Train a classifier using face embeddings and labels.

    Args:
        embeddings (numpy.ndarray): Array of face embeddings.
        labels (list): List of labels corresponding to the embeddings.
        output_path (str): File path to save the trained classifier.
    """
    # Train an SVM classifier with a linear kernel
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels)

    # Save the trained model and label list
    with open(output_path, 'wb') as f:
        pickle.dump((model, labels), f)

    print(f"Classifier trained and saved at: {output_path}")


# Example usage
if __name__ == "__main__":
    # Replace with actual embeddings and labels
    embeddings = np.random.rand(10, 128)  # Example: 10 embeddings of size 128
    labels = ["PersonA", "PersonB", "PersonC", "PersonD", "PersonE"] * 2

    train_classifier(embeddings, labels, "classifier.pkl")
