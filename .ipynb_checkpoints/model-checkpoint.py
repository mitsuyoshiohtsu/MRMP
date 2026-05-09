import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, temperature):
        super(AttentionFusion, self).__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Identity()
        self.temperature = temperature

    def forward(self, x_query, x_kv):
        query = self.query_proj(x_query)
        key = self.key_proj(x_kv)
        value = self.value_proj(x_kv)
        weight = torch.sigmoid((-1 * query * key / self.temperature).sum(-1))  # (B, 1)
        weight = weight.unsqueeze(-1)  # (B, 1, 1)
        prob = weight * value + x_query  # residual connection
        prob /= prob.sum(2, keepdim=True)
        return prob  # (B, 1, D)


class NoiseDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, temperature):
        super(NoiseDetector, self).__init__()
        self.cross_attn = AttentionFusion(input_dim=input_dim, hidden_dim=hidden_dim, temperature=temperature)

    def forward(self, y_ema, y_prev):
        # Inputs: (B, D)
        y_prev = y_prev.float().unsqueeze(1)  # (B, 1, D) - query
        y_ema = y_ema.float().unsqueeze(1)  # (B, 1, D)
        refined = self.cross_attn(y_prev, y_ema)  # (B, 1, D)
        return refined.squeeze(1)  # (B, D)