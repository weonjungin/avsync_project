import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCESyncLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, v_emb, a_emb):
        """
        v_emb: (B, D)
        a_emb: (B, D)
        """
        v = F.normalize(v_emb, dim=1)
        a = F.normalize(a_emb, dim=1)

        logits = v @ a.t() / self.temperature   # (B, B)
        labels = torch.arange(v.size(0), device=v.device)

        loss_v2a = F.cross_entropy(logits, labels)
        loss_a2v = F.cross_entropy(logits.t(), labels)

        return 0.5 * (loss_v2a + loss_a2v)
