from timm.models.registry import register_model
import torch
from torch import  nn
import torch.nn.functional as F
from einops import *
from .mamba_vision import MambaVision, MambaVisionMixer
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import math
from .registry import create_model, register_pip_model, list_models

class ToSequenceForm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.ndim == 3: return x # already sequence
        return rearrange(x, "b c h w -> b (h w) c")

class ToImageForm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        '''
        assume image has equal width and height, and sequence length is a perfect square
        '''
        if x.ndim == 4: return x # already image

        B, L, D = x.shape
        H = W = int(L ** 0.5)
        assert H * W == L, "L must be a perfect square"
        return rearrange(x, "b (h w) d -> b d h w", h=H, w=W)

def test_conversion_round_trip():
    to_seq = ToSequenceForm()
    to_img = ToImageForm()
    N, C, W, H = 2, 8, 32, 32
    x = torch.rand(N, C, W, H)
    assert torch.equal(x, to_img(to_seq(x))), "conversion round trip img -> seq -> img fail"

    x = torch.rand(N, W*H, C)
    assert torch.equal(x, to_seq(to_img(x))), "conversion round trip seq -> img -> seq fail" 
    print(f"Conversion round trip assert pass")    


class GlobalExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 expand=2, 
                 d_conv=4, 
                 d_state=16,
                 dt_rank="auto", 
                 dt_scale=1.0, 
                 dt_init="random", 
                 dt_init_floor=1e-4,
                 dt_min=0.001,
                 dt_max=0.1,
                 device="cuda"):

        super().__init__()
        self.d_state = d_state
        self.d_inner = int(in_channels*expand)
        self.dt_rank = math.ceil(in_channels/ 16) if dt_rank == "auto" else dt_rank
        self.mamba_mixer_path = MambaVisionMixer(in_channels, expand=expand, use_linear=False, d_conv=d_conv)
        self.global_proj = nn.Linear(in_channels, self.d_inner)
        self.global_conv = nn.Sequential(
            nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding="same"),
            nn.SiLU()
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        self.global_path_ln = nn.LayerNorm(self.d_inner)
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.out_proj = nn.Linear(self.d_inner, out_channels)
        self.to_sequence = ToSequenceForm()
        self.to_img = ToImageForm()

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
   
    def global_path(self, x):
        _, seqlen, D = x.shape
        x = self.global_proj(x)
        x = self.global_path_ln(x)
        x =  self.global_conv(rearrange(x, "b l d -> b d l"))

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        A = -torch.exp(self.A_log.float())
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        y = rearrange(y, "b d l -> b l d")
        return y

    def forward(self, f1, f2):
        f1 = self.to_sequence(f1)
        f2 = self.to_sequence(f2)

        x11 = self.mamba_mixer_path(f1)
        x12 = self.global_path(f2)

        x21 = self.mamba_mixer_path(f2)
        x22 = self.global_path(f1)

        return self.to_img(self.out_proj(x11 * x12)), self.to_img(self.out_proj(x21 * x22))
            
    

class LocalExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, expand=2):
        super().__init__()
        self.d_inner = int(in_channels*expand)
        self.mamba_mixer_path = MambaVisionMixer(in_channels, expand=expand, use_linear=False)
        self.local_ext_path = nn.Sequential(
            ToImageForm(),
            nn.Conv2d(in_channels, self.d_inner, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=3, padding=1),
            ToSequenceForm(),
            nn.SiLU()
        ) 
        self.out_proj = nn.Linear(self.d_inner, out_channels)
        self.to_img = ToImageForm()

    def forward(self, f1, f2):
        x11 = self.mamba_mixer_path(f1)
        x12 = self.local_ext_path(f2)

        x21 = self.mamba_mixer_path(f2)
        x22 = self.local_ext_path(f1)
        return self.to_img(self.out_proj(x11 * x12)), self.to_img(self.out_proj(x21 * x22))
        

class LocalGlobalFusion(nn.Module):
    def __init__(self, in_channels, reduce_to_dim=None, apply_layernorm=False, dim_reduce_mode="linear"):
        super().__init__()
        self.apply_layernorm = apply_layernorm
        self.in_channels = in_channels
        self.ln = nn.LayerNorm(in_channels)
        self.reduce_to_dim = in_channels if reduce_to_dim is None else reduce_to_dim
        if dim_reduce_mode == "linear":
            self.dim_reduce = nn.Linear(in_channels, self.reduce_to_dim)
        elif dim_reduce_mode == "conv":
            self.dim_reduce = nn.Conv2d(in_channels, self.reduce_to_dim, kernel_size=3, padding=1)
        else: 
            print(f"invalid reduce mode: {dim_reduce_mode}, must be either linear or conv")
        self.global_extractor = GlobalExtractor(self.reduce_to_dim, self.reduce_to_dim)
        self.local_extractor = LocalExtractor(self.reduce_to_dim, self.reduce_to_dim)
        self.sum_weight_proj = nn.Linear(self.reduce_to_dim*2, 2)
        self.to_seq = ToSequenceForm()

    def compute_gate_score(self, f_g, f_l): # each of shape B L D
        f_g = self.to_seq(f_g)
        f_l = self.to_seq(f_l)
        f_g_mean = torch.mean(f_g, dim=1) # B D
        f_l_mean = torch.mean(f_l, dim=1) # B D
        f_gl_mean = torch.cat([f_g_mean, f_l_mean], dim=-1) # B 2D
        gate_score = F.softmax(self.sum_weight_proj(f_gl_mean), dim=-1) # B 2
        # gate score weights importance of each dimension
        return gate_score
 
    def forward(self, x1, x2):
        x1 = self.to_seq(x1)
        x2 = self.to_seq(x2)
        if self.apply_layernorm:
            x1 = self.ln(x1)
            x2 = self.ln(x2)
        if self.reduce_to_dim != self.in_channels:
            x1 = self.dim_reduce(x1)
            x2 = self.dim_reduce(x2)
        B, L, D = x1.shape
        glb1, glb2 = self.global_extractor(x1, x2)
        lcl1, lcl2 = self.local_extractor(x1, x2)

        G1 = self.compute_gate_score(glb1, lcl1)
        G2 = self.compute_gate_score(glb2, lcl2)

        x1 = G1[:, 0:1].view(B, 1, 1, 1)*lcl1 + G1[:, 1:2].view(B, 1, 1, 1)*glb1
        x2 = G2[:, 0:1].view(B, 1, 1, 1)*lcl2 + G2[:, 1:2].view(B, 1, 1, 1)*glb2

        return torch.abs(x1 - x2)

class MambaVisionCDDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, fuse_features=True):
        super().__init__()
        self.upsample = upsample
        self.fuse_features = fuse_features
        self.mixer = MambaVisionMixer(in_channels)
        self.forward_features = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )
        self.to_seq = ToSequenceForm()
        self.to_img = ToImageForm()

    def forward(self, x, x_last=None):
        x = self.to_seq(x)
        x = self.mixer(x)
        x = self.to_img(x)
        if self.fuse_features and x_last is not None:
            x_last = self.to_img(x_last)
            x += x_last
        if not self.upsample:
            return x
        return self.forward_features(x)

class ConvUpsampleAndClassify(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dims=256, upsample=True):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = upsample

        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1)
        self.dense = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims, in_channels, kernel_size=3, padding=1)
        )
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1)
        self.conv_classify = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        N, C, W, H = x.shape
        if self.upsample:
            x = self.conv1(x)
            assert x.shape == (N, C, W*2, H*2), x.shape
        x1 = self.dense(x)
        if self.upsample:
            x1 = self.conv2(x + x1)
            assert x1.shape == (N, C, W*4, H*4), x.shape
        class_logits = self.conv_classify(x1)
        return class_logits

class MambaVisionCDDecoder(nn.Module):
    def __init__(self,
                 num_classes,
                 dims,
                 reduced_dims=None,
                 upsample=True):

        reduced_dims = dims if reduced_dims is None else reduced_dims
        super().__init__()
        self.dims = dims
        self.fusion = nn.ModuleList([
            LocalGlobalFusion(dims[i], reduce_to_dim=reduced_dims[i], dim_reduce_mode="linear", apply_layernorm=True) for i in range(len(dims)) 
        ])

        # for now, assume len(dims) = 4
        self.lowest_block = MambaVisionCDDecoderBlock(reduced_dims[3], reduced_dims[2], upsample=True, fuse_features=False)
        self.block1 = MambaVisionCDDecoderBlock(reduced_dims[2], reduced_dims[1], upsample=True, fuse_features=True)
        self.block2 = MambaVisionCDDecoderBlock(reduced_dims[1], reduced_dims[0], upsample=True, fuse_features=True)
        self.final_block = MambaVisionCDDecoderBlock(reduced_dims[0], reduced_dims[0], upsample=False, fuse_features=True)
        self.classifier = ConvUpsampleAndClassify(reduced_dims[0], num_classes, upsample=upsample)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x1s, x2s):
        x11, x12, x13, x14 = x1s
        x21, x22, x23, x24 = x2s
        x_4_fuse = self.lowest_block(self.fusion[3](x14, x24))
        x_3_fuse = self.block1(self.fusion[2](x13, x23), x_last=x_4_fuse)
        x_2_fuse = self.block2(self.fusion[1](x12, x22), x_last=x_3_fuse)
        x_1_fuse = self.final_block(self.fusion[0](x11, x21), x_last=x_2_fuse)
        return self.classifier(x_1_fuse)

class MambaVisionCD(nn.Module):
    def __init__(self,
                 in_chans,
                 encoder_model=None,
                 dims=[64, 128, 256, 512],
                 reduced_dims=None,
                 depths=[2, 2, 4, 2],
                 window_size=[4, 4, 6, 8],
                 mlp_ratio=4,
                 num_heads=[2, 4, 8, 16],
                 drop_path_rate=0.2,
                 num_classes=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        super().__init__()
        if encoder_model is not None:
            self.enc = create_model(encoder_model, in_chans=in_chans, pretrained=True, **kwargs)
        else:
            self.enc = MambaVision(
                     in_chans,
                     dims,
                     depths,
                     window_size,
                     mlp_ratio,
                     num_heads,
                     drop_path_rate=drop_path_rate,
                     num_classes=num_classes,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     drop_rate=drop_rate,
                     attn_drop_rate=attn_drop_rate,
                     layer_scale=layer_scale,
                     layer_scale_conv=layer_scale_conv
            )
        self.dec = MambaVisionCDDecoder(num_classes,
                                        dims=self.enc.dims,
                                        reduced_dims=None,
                                        upsample=True)

    def forward(self, x1, x2):
        x1s = self.enc(x1)
        x2s = self.enc(x2)
        return self.dec(x1s, x2s)


if __name__ == "__main__":
    print(list_models())
