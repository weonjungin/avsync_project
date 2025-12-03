import torch
import torch.nn as nn

class ContrastiveSyncLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)
        self.margin = margin

    def forward(self, v_pos, a_pos, v_neg, a_neg):
        pos_sim = self.cos(v_pos, a_pos)
        pos_loss = 1 - pos_sim.mean()

        neg_sim = self.cos(v_neg, a_neg)
        neg_loss = torch.clamp(neg_sim - self.margin, min=0).mean()

        return pos_loss + neg_loss
