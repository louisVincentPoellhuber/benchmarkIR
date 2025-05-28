
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
from enum import Enum
from sentence_transformers.util import cos_sim, dot_score, euclidean_sim

cross_entropy = nn.CrossEntropyLoss()

class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)

class InBatchNegativeLoss(nn.Module):
    def __init__(self):
        super(InBatchNegativeLoss, self).__init__()
    
    def forward(self, query_vectors, doc_vectors):
        """
        Args:
            query_vectors: Tensor of shape [batch_size, vector_dim]
            doc_vectors: Tensor of shape [batch_size, vector_dim]
        """
        batch_size = query_vectors.size(0)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(query_vectors, doc_vectors.transpose(0, 1))
        
        # Labels are just the diagonal indices
        labels = torch.arange(batch_size, device=query_vectors.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    

class TripletLoss(TripletMarginLoss):
    def forward(self, query_vectors: Tensor, doc_vectors: Tensor) -> Tensor:
        batch_size = doc_vectors.size(0)
        assert batch_size / 2 == int(batch_size / 2), "Batch size must be even for triplet loss."
        
        positive = doc_vectors[:batch_size // 2]
        negative = doc_vectors[batch_size // 2:]

        return super().forward(query_vectors, positive, negative)
    
