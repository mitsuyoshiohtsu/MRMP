import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """
    Soft attention gate that fuses a query distribution with a key-value distribution.

    Given:
        x_query : (B, 1, D)  — noisy label distribution (y_prev)
        x_kv    : (B, 1, D)  — EMA pseudo-label distribution (y_ema)

    The gate weight is a per-sample scalar in (0, 1) computed from the
    dot-product similarity between projected query and key vectors.
    A high weight means the EMA pseudo-label is trusted more; a low weight
    falls back to the noisy label via the residual connection.

    Output: (B, 1, D)  — purified soft label, normalised to sum to 1.
    """

    def __init__(self, input_dim: int, hidden_dim: int, temperature: float):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj   = nn.Linear(input_dim, hidden_dim)
        self.temperature = temperature

    def forward(self, x_query: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_query : (B, 1, D)
            x_kv    : (B, 1, D)
        Returns:
            prob    : (B, 1, D)  normalised purified label distribution
        """
        query = self.query_proj(x_query)   # (B, 1, H)
        key   = self.key_proj(x_kv)        # (B, 1, H)
        value = x_kv                       # (B, 1, D)  — FIX #16: direct use

        gate = torch.sigmoid(
            (-1.0 * query * key / self.temperature).sum(dim=-1, keepdim=True)
        )  # (B, 1, 1)

        prob = gate * value + (1.0 - gate) * x_query  # (B, 1, D)

        prob = prob.clamp(min=0.0)
        prob = prob / prob.sum(dim=2, keepdim=True).clamp(min=1e-8)

        return prob  # (B, 1, D)


class NoiseDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, temperature: float):
        super().__init__()
        self.cross_attn = AttentionFusion(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            temperature=temperature,
        )

    def forward(self, y_ema: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        # Expand to (B, 1, D) for AttentionFusion
        y_prev = y_prev.float().unsqueeze(1)  # (B, 1, D) — query
        y_ema  = y_ema.float().unsqueeze(1)   # (B, 1, D) — key / value

        refined = self.cross_attn(y_prev, y_ema)  # (B, 1, D)
        return refined.squeeze(1)                  # (B, D)
