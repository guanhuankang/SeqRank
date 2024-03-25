import torch.nn as nn

class LayerNorm2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        '''

        Args:
            x: B, C, H, W

        Returns:
            y: B, C, H, W
        '''
        x = self.norm(x.permute(0,2,3,1))
        return x.permute(0,3,1,2)

def init_weights_(m):
    for w in m.modules():
        if isinstance(w, nn.Linear) or isinstance(w, nn.Conv2d):
            nn.init.xavier_uniform_(w.weight)
            nn.init.zeros_(w.bias)
        elif isinstance(w, nn.LayerNorm):
            nn.init.constant_(w.weight, 1.0)
            nn.init.zeros_(w.bias)

def reset_parameters_(m):
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
