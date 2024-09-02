from torch import nn

class ViTWrapper(nn.Module):
    def __init__(self, vit):
        super(ViTWrapper, self).__init__()
        self._vit = vit
        self.softmax = nn.Softmax(dim=1)
        return
    
    def forward(self, x):
        x = self._vit(x)
        out = self.softmax(x)
        return out