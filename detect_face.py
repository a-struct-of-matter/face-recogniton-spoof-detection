from mtcnn import MTCNN

def detect_faces(image):
    detector = MTCNN()
    return detector.detect_faces(image)
