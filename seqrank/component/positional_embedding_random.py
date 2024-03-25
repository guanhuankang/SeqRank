import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    @property
    def device(self):
        return self.positional_encoding_gaussian_matrix.device

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, feat) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = feat.shape[2::]
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        pe = pe.permute(2, 0, 1)  # C x H x W
        pe = pe.unsqueeze(0).expand(len(feat), -1, -1, -1)  ## B, C, H, W
        return pe

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone().to(torch.float)
        coords[:, :, 0] = coords[:, :, 0] / float(image_size[1])
        coords[:, :, 1] = coords[:, :, 1] / float(image_size[0])
        return self._pe_encoding(coords.to(self.device))  # B x N x C