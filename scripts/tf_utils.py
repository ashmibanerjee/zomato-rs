import tensorflow as tf
import numpy as np


def compute_cosine_sim(a, b):
    '''
    Note that it is a number between -1 and 1. When it is a negative number
    between -1 and 0, 0 indicates orthogonality and values closer to -1
    indicate greater similarity. The values closer to 1 indicate greater
    dissimilarity.
    Hence, we multiply the result with *-1 to make it at par with the Torch results.
    '''
    s = tf.keras.losses.cosine_similarity(a, b)
    return s*-1


def convert_to_tensor(query_vals, remaining_vals):
    if type(query_vals) is list:
        query_embeddings = tf.convert_to_tensor(query_vals, dtype=tf.float32)
    else:
        query_embeddings = tf.convert_to_tensor(query_vals, dtype=tf.float32)
    remaining_embeddings = np.vstack(remaining_vals).astype(float)
    remaining_embeddings = tf.convert_to_tensor(remaining_embeddings, dtype=tf.float32)
    return query_embeddings, remaining_embeddings
