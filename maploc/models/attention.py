import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math 
from einops import rearrange, reduce, repeat

def l2norm(inp, dim=0):
    return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        # from IPython import embed;embed()

        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class TCenterAttentionLayer(nn.Module):
    def __init__(self, global_channels, num_centers=64, local_channels=0, num_heads=1):
        super().__init__()
        assert global_channels % num_heads == 0
        self.num_heads = num_heads
        init_centers = torch.Tensor(global_channels, num_centers)
        init_centers.normal_(0, math.sqrt(2.0 / num_centers))
        init_centers = l2norm(init_centers)
        self.centers = torch.nn.Parameter(init_centers)
        if local_channels > 0:
            self.fuse_conv = nn.Conv1d(
                global_channels + local_channels, global_channels, 1)

        channels = global_channels
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, local_x, t_embed):
        """
            local_x: [bs, global_channels, npts]
            t_embed: [bs, global_channels]
        """
        bs, _, _ = local_x.shape

        # from IPython import embed;embed()
        if t_embed is not None:
            x = local_x + t_embed.unsqueeze(dim=-1).repeat(1, 1, local_x.shape[-1])
        else:
            x = local_x

        x_q = self.q_conv(x).permute(0, 2, 1)  # bs, npts, channels
        x_k = self.centers.unsqueeze(dim=0).repeat(
            bs, 1, 1)  # bs, channels, num_centers
        x_v = self.v_conv(self.centers.unsqueeze(dim=0)).repeat(
            bs, 1, 1)  # bs, channels, num_centers

        x_q = torch.cat(torch.chunk(x_q, self.num_heads, dim=2),
                        dim=0)  # num_heads * bs, npts, channels/num_heads
        x_k = torch.cat(torch.chunk(x_k, self.num_heads, dim=1),
                        dim=0)  # num_heads * bs, channels/num_heads, num_centers
        x_v = torch.cat(torch.chunk(x_v, self.num_heads, dim=1),
                        dim=0)  # num_heads * bs, channels/num_heads, num_centers

        energy = torch.bmm(x_q, x_k)  # num_heads * bs, npts, num_centers
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        # num_heads * bs, channels/num_heads, num_centers
        x_r = torch.bmm(x_v, attention.permute(0, 2, 1))
        x_r = torch.cat(torch.chunk(x_r, self.num_heads, dim=0), dim=1)
        # bs, channels, npts
        x_r = self.act(self.after_norm(self.trans_conv(x_r-x)))
        x = x + x_r
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        channels = cfg.global_channels
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.ReLU(inplace=True)

        self.attn_layers = nn.ModuleList()
        for i in range(cfg.num_attn_layers):
            self.attn_layers.append(
                TCenterAttentionLayer(cfg.global_channels,
                                     cfg.num_centers[i], cfg.local_channels, cfg.num_heads)
            )

        self.conv2 = nn.Conv1d(sum(cfg.num_centers), channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, global_x, local_x=None):

        global_x = self.act1(self.bn1(self.conv1(global_x)))

        xs = [global_x]

        for attn_layer in self.attn_layers:
            xs.append(attn_layer(xs[-1], local_x))

        out = self.act2(self.bn2(self.conv2(torch.cat(xs, dim=1))))

        return out

