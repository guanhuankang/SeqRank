import torch
import torch.nn as nn
from detectron2.config import configurable

from ..component import Attention, MLPBlock, init_weights_
from .registry import GAZE_SHIFT_HEAD

class CenterShiftBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.self_attn1 = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn2 = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout2 = nn.Dropout(p=dropout_attn)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.cross_attn1 = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout3 = nn.Dropout(p=dropout_attn)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.cross_attn2 = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout4 = nn.Dropout(p=dropout_attn)
        self.norm4 = nn.LayerNorm(embed_dim)
        
        self.ffn1 = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout5 = nn.Dropout(p=dropout_attn)
        self.norm5 = nn.LayerNorm(embed_dim)
        self.ffn2 = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout6 = nn.Dropout(p=dropout_attn)
        self.norm6 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim+embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout_mlp = nn.Dropout(p=dropout_ffn)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        
        self.softmax_head = nn.Linear(embed_dim, 1)
        self.before = nn.Linear(embed_dim, embed_dim)
        self.after = nn.Linear(embed_dim, embed_dim, bias=True)

        init_weights_(self)
    
    def forward(self, q, qpe, z, zpe, gaze, gaze_emb, prev, prev_emb):
        """
        q, qpe: B, nq, C
        gaze: B, nq, 1 [0/1 indicate which one is gazed at present]
        z, zpe: B, hw, C
        gaze_emb: 1, 1, C
        """
        q0 = self.before(q)  ## B, nq, C

        q = q + prev * prev_emb + gaze * gaze_emb  ## add emb

        q = self.norm1(q + self.dropout1(self.self_attn1(q=q+qpe, k=q+qpe, v=q)))
        q = self.norm2(q + self.dropout2(self.ffn1(q)))
        z = self.norm3(z + self.dropout3(self.cross_attn1(q=z+zpe, k=q+qpe, v=q)))

        q_cat = torch.sum(gaze * q, dim=1, keepdim=True) + gaze_emb  ## B, 1, C
        q_cat = q_cat.repeat_interleave(z.shape[1], dim=1)  ## B, hw, C
        q_cat = torch.cat([q_cat, z], dim=-1)  ## B, hw, 2C
        z = self.norm_mlp(z + self.dropout_mlp(self.mlp(q_cat)))  ## B, hw, C

        q = self.norm4(q + self.dropout4(self.cross_attn2(q=q+qpe, k=z+zpe, v=z)))
        q = self.norm5(q + self.dropout5(self.self_attn2(q=q+qpe, k=q+qpe, v=q)))
        q = self.norm6(q + self.dropout6(self.ffn2(q)))

        q = q * torch.softmax(self.softmax_head(q), dim=-1)  ## B, nq, C
        q = self.after(q0 + q)  ## B, nq, C|merge q0 information to help filtering

        return q

@GAZE_SHIFT_HEAD.register()
class SequentialRankingModule(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2):
        super().__init__()
        self.prev_emb = nn.Parameter(torch.randn(embed_dim))
        self.gaze_emb = nn.Parameter(torch.randn(embed_dim))

        self.blocks = nn.ModuleList([CenterShiftBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout_attn=dropout_attn,
            dropout_ffn=dropout_ffn
        ) for _ in range(num_blocks)])

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim":    cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads":    cfg.MODEL.COMMON.NUM_HEADS,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "hidden_dim":   cfg.MODEL.COMMON.HIDDEN_DIM,
            "dropout_ffn":  cfg.MODEL.COMMON.DROPOUT_FFN,
            "num_blocks":   cfg.MODEL.GAZE_SHIFT_HEAD.NUM_BLOCKS
        }

    def forward(self, q, z, qpe, zpe, q_vis, bbox, size):
        """

        Args:
            q: B, nq, C
            z: B, hw, C
            qpe: B, nq, C
            zpe: B, hw, C
            q_vis: B, nq, 1 (int: 0-n)
            bbox: B, nq, 4 [xyhw] in [0,1]
            size: Tuple(h, w)

        Returns:
            saliency: B, nq, 1 (logit)
        """
        prev_emb = self.prev_emb[None, None, :]  ## 1, 1, C
        gaze_emb = self.gaze_emb[None, None, :]  ## 1, 1, C

        gaze_rank, _ = torch.max(q_vis[:, :, 0], dim=1)  ## B
        prev = (q_vis.gt(0.5) * q_vis.le(gaze_rank[:, None, None])).float()  ## B, nq, 1
        gaze = (q_vis.gt(0.5) * q_vis.eq(gaze_rank[:, None, None])).float()  ## B, nq, 1

        # q = q + prev * prev_emb + gaze * gaze_emb  ## add emb

        predictions = []
        for layer in self.blocks:
            q = layer(q=q, qpe=qpe, z=z, zpe=zpe, gaze=gaze, gaze_emb=gaze_emb, prev=prev, prev_emb=prev_emb)
            predictions.append(self.head(q))  ## B, nq, 1
        
        out = predictions[-1]
        auxs = predictions[0:-1]
        
        if self.training:
            return out, []
        else:
            return out
