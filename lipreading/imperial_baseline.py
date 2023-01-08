import torch
import torch.nn as nn
import math
import numpy as np
from lipreading.models.resnet import ResNet, BasicBlock
from lipreading.models.resnet1D import ResNet1D, BasicBlock1D
from lipreading.models.shufflenetv2 import ShuffleNetV2
from lipreading.models.tcn import MultibranchTemporalConvNet, TemporalConvNet
from lipreading.models.densetcn import DenseTemporalConvNet
from lipreading.models.swish import Swish


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    return torch.stack(
        [torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0
    )


class MultiscaleMultibranchTCN(nn.Module):
    def __init__(
        self,
        input_size,
        num_channels,
        num_classes,
        tcn_options,
        dropout,
        relu_type,
        dwpw=False,
    ):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options["kernel_size"]
        self.num_kernels = len(self.kernel_sizes)

        self.mb_ms_tcn = MultibranchTemporalConvNet(
            input_size,
            num_channels,
            tcn_options,
            dropout=dropout,
            relu_type=relu_type,
            dwpw=dwpw,
        )
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        out = self.consensus_func(out, lengths, B)
        return self.tcn_output(out)


class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(
        self,
        input_size,
        num_channels,
        num_classes,
        tcn_options,
        dropout,
        relu_type,
        dwpw=False,
    ):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(
            input_size,
            num_channels,
            dropout=dropout,
            tcn_options=tcn_options,
            relu_type=relu_type,
            dwpw=dwpw,
        )
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

        self.has_aux_losses = False

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func(x, lengths, B)
        return self.tcn_output(x)


class DenseTCN(nn.Module):
    def __init__(
        self,
        block_config,
        growth_rate_set,
        input_size,
        reduced_size,
        num_classes,
        kernel_size_set,
        dilation_size_set,
        dropout,
        relu_type,
        squeeze_excitation=False,
    ):
        super(DenseTCN, self).__init__()

        num_features = reduced_size + block_config[-1] * growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet(
            block_config,
            growth_rate_set,
            input_size,
            reduced_size,
            kernel_size_set,
            dilation_size_set,
            dropout=dropout,
            relu_type=relu_type,
            squeeze_excitation=squeeze_excitation,
        )
        self.tcn_output = nn.Linear(num_features, num_classes)
        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func(x, lengths, B)
        return self.tcn_output(x)


class Lipreading(nn.Module):
    def __init__(
        self,
        modality="video",
        hidden_dim=256,
        backbone_type="resnet",
        num_classes=500,
        relu_type="prelu",
        tcn_options={},
        densetcn_options={},
        width_mult=1.0,
        use_boundary=False,
        extract_feats=False,
        backbone_squeeze_excitation=False,
        n_batch=1,
    ):
        super(Lipreading, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.modality = modality
        self.use_boundary = use_boundary

        if self.modality == "audio":
            self.frontend_nout = 1
            self.backend_out = 512
            self.trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type=relu_type)
        elif self.modality == "video":
            # ==> define backbone feature extractor
            if self.backbone_type == "resnet":
                # self.frontend_out = int(64 * width_mult)  # scale the number of channels
                # self.backend_out = int(512 * width_mult)  # scale the number of channels
                divisor = 1 if width_mult == 1.0 else 2
                self.frontend_nout = 64 // divisor
                self.backend_out = 512 // divisor
                # add width_mult and SE on ResNet
                self.trunk = ResNet(
                    BasicBlock,
                    [2, 2, 2, 2],
                    relu_type=relu_type,
                    squeeze_excitation=backbone_squeeze_excitation,
                    width_mult=width_mult,
                )
            elif self.backbone_type == "resnet-tsm": # HACK: Batch size is must be equal for all subset
                divisor = 1 if width_mult == 1.0 else 2
                self.frontend_nout = 64 // divisor
                self.backend_out = 512 // divisor

                from lipreading.models.temporal_shift import make_temporal_shift

                resnet = ResNet(
                    BasicBlock,
                    [2, 2, 2, 2],
                    relu_type=relu_type,
                    squeeze_excitation=backbone_squeeze_excitation,
                    width_mult=width_mult,
                )
                # FIXME: n_segment should by dynamic based on input becuase we are using variable length augmentation
                make_temporal_shift(
                    resnet, n_segment=29, n_div=8, place="blockres", n_batch=n_batch # default: n_segment = 29 (seq max length)
                )  # employ TSM module

                self.trunk = resnet
            elif self.backbone_type == "shufflenet":
                assert width_mult in [
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                ], "width_mult for shufflenet_v2 should be in [0.5, 1.0, 1.5, 2.0]"
                self.frontend_nout = 24
                self.backend_out = 1024 if width_mult != 2.0 else 2048
                svn = ShuffleNetV2(
                    input_size=96, width_mult=width_mult
                )  # TODO: input size should be 88x88 after preprocessing
                self.trunk = nn.Sequential(
                    svn.features,
                    svn.conv_last,
                    svn.globalpool,
                )
                self.stage_out_channels = svn.stage_out_channels[-1]
            elif self.backbone_type == "mobilenet":  # mobilenetv3 small
                from lipreading.models.mobilenetv3 import mobilenetv3_small

                self.frontend_nout = 16
                self.backend_out = 48
                self.trunk = (
                    mobilenetv3_small()
                )  # mobilenetv3 already use SE layer in some blocks
            elif self.backbone_type == "mobilenet-large":  # mobilenetv3 large
                from lipreading.models.mobilenetv3 import mobilenetv3_large

                self.frontend_nout = 16
                self.backend_out = 112
                self.trunk = (
                    mobilenetv3_large()
                )  # mobilenetv3 already use SE layer in some blocks
            elif self.backbone_type == "swin-transformer-v2":
                from lipreading.models.swin_transformer_v2 import SwinTransformerV2_T

                self.frontend_nout = 3
                self.backend_out = 768
                self.trunk = SwinTransformerV2_T()
            else:
                raise ValueError(
                    f"backbone type {self.backbone_type} is not supported."
                )

            # -- frontend3D
            if relu_type == "relu":
                frontend_relu = nn.ReLU(True)
            elif relu_type == "prelu":
                frontend_relu = nn.PReLU(self.frontend_nout)
            elif relu_type == "swish":
                frontend_relu = Swish()

            self.frontend3D = nn.Sequential(
                nn.Conv3d(
                    1,
                    self.frontend_nout,
                    kernel_size=(5, 7, 7),
                    stride=(1, 2, 2),
                    padding=(2, 3, 3),
                    bias=False,
                ),
                nn.BatchNorm3d(self.frontend_nout),
                frontend_relu,
                nn.MaxPool3d(
                    kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
                ),
            )
        else:
            raise NotImplementedError

        if tcn_options:
            tcn_class = (
                TCN
                if len(tcn_options["kernel_size"]) == 1
                else MultiscaleMultibranchTCN
            )
            self.tcn = tcn_class(
                input_size=self.backend_out,
                num_channels=[
                    hidden_dim
                    * len(tcn_options["kernel_size"])
                    * tcn_options["width_mult"]
                ]
                * tcn_options["num_layers"],
                num_classes=num_classes,
                tcn_options=tcn_options,
                dropout=tcn_options["dropout"],
                relu_type=relu_type,
                dwpw=tcn_options["dwpw"],
            )
        elif densetcn_options:
            self.tcn = DenseTCN(
                block_config=densetcn_options["block_config"],
                growth_rate_set=densetcn_options["growth_rate_set"],
                input_size=self.backend_out
                if not self.use_boundary
                else self.backend_out + 1,
                reduced_size=densetcn_options["reduced_size"],
                num_classes=num_classes,
                kernel_size_set=densetcn_options["kernel_size_set"],
                dilation_size_set=densetcn_options["dilation_size_set"],
                dropout=densetcn_options["dropout"],
                relu_type=relu_type,
                squeeze_excitation=densetcn_options["squeeze_excitation"],
            )
        else:
            raise NotImplementedError

        # -- initialize
        self._initialize_weights_randomly()

    def forward(self, x, lengths=None, boundaries=None):
        if lengths is None:
            lengths = [x.shape[2]] * x.shape[0]

        if self.modality == "video":
            B, C, T, H, W = x.size()
            x = self.frontend3D(x)
            # print("@ImperialBaselineModel.forward after frontend3d x.shape : ", x.shape)
            Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
            x = threeD_to_2D_tensor(x)
            # print("@ImperialBaselineModel.forward after to2d tensor x.shape : ", x.shape)
            # TODO: Better idea: use Temporal information to determine n_segment, pass temporal information in forward method maybe using Tnew!!!
            x = self.trunk(x)

            if self.backbone_type == "shufflenet":
                x = x.view(-1, self.stage_out_channels)
            x = x.view(B, Tnew, x.size(1))
        elif self.modality == "audio":
            B, C, T = x.size()
            x = self.trunk(x)
            x = x.transpose(1, 2)
            lengths = [_ // 640 for _ in lengths]

        # -- duration
        if self.use_boundary:
            x = torch.cat([x, boundaries], dim=-1)

        return x if self.extract_feats else self.tcn(x, lengths, B)

    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:

            def f(n):
                return math.sqrt(2.0 / float(n))

        else:

            def f(n):
                return 2.0 / float(n)

        for m in self.modules():
            if (
                isinstance(m, nn.Conv3d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                n = np.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif (
                isinstance(m, nn.BatchNorm3d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
            ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))
