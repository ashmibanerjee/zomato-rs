import torch
from sentence_transformers import util
import numpy as np


def compute_cosine_sim(query_vector, remaining_vector):
    results = util.cos_sim(query_vector, remaining_vector)
    return results.flatten().tolist()


def convert_to_tensor(query_vals, remaining_vals):

    if type(query_vals) is list:
        query_embeddings = torch.FloatTensor(query_vals).float()
    else:
        query_embeddings = torch.from_numpy(query_vals).float()
    remaining_embeddings = np.vstack(remaining_vals).astype(float)
    remaining_embeddings = torch.from_numpy(remaining_embeddings).float()

    return query_embeddings, remaining_embeddings
