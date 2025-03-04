
import torch
from torch import Tensor
import torch.nn as nn

from sentence_transformers.util import cos_sim, dot_score, euclidean_sim

cross_entropy = nn.CrossEntropyLoss()

def contrastive_loss(q_embeddings, ctx_embeddings, scale = 1):

    scores = dot_score(q_embeddings, ctx_embeddings) * scale

    range_labels = torch.arange(0, scores.size(0), device=scores.device)
    
    loss = cross_entropy(scores, range_labels)

    return loss
