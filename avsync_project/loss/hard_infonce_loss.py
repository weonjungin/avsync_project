import torch
import torch.nn as nn
import torch.nn.functional as F

class HardInfoNCESyncLoss(nn.Module):
    """
    Hard negative InfoNCE:
    For each sample i,
      pos = sim(i,i)
      hard_negs = topk(sim(i,j), j!=i)
    logits = [pos, hard_negs...], label=0
    Also symmetric direction (a->v) averaged.
    """
    def __init__(self, temperature: float = 0.07, hard_k: int = 5):
        super().__init__()
        assert hard_k >= 1
        self.temperature = float(temperature)
        self.hard_k = int(hard_k)

    def _one_direction(self, sim: torch.Tensor) -> torch.Tensor:
        # sim: (B,B) already divided by temperature
        B = sim.size(0)
        if B < 2:
            # not enough negatives
            return sim.new_tensor(0.0)

        # positives: diagonal
        pos = sim.diag()  # (B,)

        # negatives: off-diagonal
        mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
        neg = sim[mask].view(B, B - 1)  # (B, B-1)

        k = min(self.hard_k, B - 1)
        hard_neg, _ = neg.topk(k=k, dim=1, largest=True, sorted=False)  # (B,k)

        logits = torch.cat([pos.unsqueeze(1), hard_neg], dim=1)  # (B, 1+k)
        labels = torch.zeros(B, dtype=torch.long, device=sim.device)  # pos index=0
        return F.cross_entropy(logits, labels)

    def forward(self, v_emb: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        v = F.normalize(v_emb, dim=1)
        a = F.normalize(a_emb, dim=1)

        sim = (v @ a.t()) / self.temperature  # (B,B)

        loss_v2a = self._one_direction(sim)
        loss_a2v = self._one_direction(sim.t())
        return 0.5 * (loss_v2a + loss_a2v)
