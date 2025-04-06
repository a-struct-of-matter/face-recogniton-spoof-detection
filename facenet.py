import tensorflow as tf
import os

def load_model(model_path):
    """
    Load the pre-trained FaceNet model.
    """
    print("Loading model from:", model_path)
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    model_exp = os.path.expanduser(model_path)
    with tf.compat.v1.gfile.GFile(model_exp, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    print("Model loaded successfully.")
    return sess

def run_embeddings(face_image, sess):
    """
    Generate embeddings for a given face image.
    """
    try:
        images_placeholder = sess.graph.get_tensor_by_name("input:0")
        phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
        embeddings_tensor = sess.graph.get_tensor_by_name("embeddings:0")

        feed_dict = {images_placeholder: [face_image], phase_train_placeholder: False}
        embeddings = sess.run(embeddings_tensor, feed_dict=feed_dict)
        return embeddings[0]
    except Exception as e:
        print(f"Error in generating embeddings: {e}")
        return None
