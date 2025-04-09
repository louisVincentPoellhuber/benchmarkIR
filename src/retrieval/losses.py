
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers.util import cos_sim, dot_score, euclidean_sim

cross_entropy = nn.CrossEntropyLoss()

def contrastive_loss(q_embeddings, ctx_embeddings, scale = 1):

    scores = dot_score(q_embeddings, ctx_embeddings) * scale

    range_labels = torch.arange(0, scores.size(0), device=scores.device)
    
    loss = cross_entropy(scores, range_labels)

    return loss


class InBatchNegativeLoss(nn.Module):
    def __init__(self):
        super(InBatchNegativeLoss, self).__init__()
    
    def forward(self, query_vectors, doc_vectors):
        """
        Args:
            query_vectors: Tensor of shape [batch_size, vector_dim]
            doc_vectors: Tensor of shape [batch_size, vector_dim]
        """
        batch_size = doc_vectors.size(0)
        
        # Calculate similarity matrix
        # TODO: output similarity
        similarity_matrix = torch.matmul(query_vectors, doc_vectors.transpose(0, 1))
        
        # Labels are just the diagonal indices
        labels = torch.arange(batch_size, device=query_vectors.device)
        
        # Cross entropy loss
        # TODO: ensure it's normalized somewhere
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss