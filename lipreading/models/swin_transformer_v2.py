# Path: model/components/swin_transformer_v3.py
import torch.nn as nn
from torchinfo import summary
from torchvision.models.swin_transformer import swin_v2_t


class SwinTransformerV2_T(nn.Module):
    def __init__(self, weights=False):
        super(SwinTransformerV2_T, self).__init__()
        self.model = swin_v2_t(weights=weights)
        # remove last flatten and linear layer
        swin_feature = list(self.model.children())[:-2]
        self.model = nn.Sequential(*swin_feature)

    def forward(self, x):
        x = self.model(x)
        # reshape from (batch, channel, height, width) to (batch, channel*height*width)
        x = x.reshape(x.shape[0], -1)
        return x


if __name__ == "__main__":
    model = SwinTransformerV2_T(weights=False)
    summary(model, input_size=(1, 3, 88, 88), verbose=1, depth=4, device="cpu")
    model.eval()
