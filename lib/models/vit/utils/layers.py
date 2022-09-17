import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (nn.Sequential(nn.Linear(inner_dim, dim),
                                     nn.Dropout(dropout))
                       if project_out else nn.Identity())

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j'),
        )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout))

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class LeFF(nn.Module):
    def __init__(self, dim=192, scale=4, depth_kernel=3):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(
            nn.Linear(dim, scale_dim),
            Rearrange('b n c -> b c n'),
            nn.BatchNorm1d(scale_dim),
            nn.GELU(),
            Rearrange('b c (h w) -> b c h w', h=2, w=2),
        )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(
                scale_dim,
                scale_dim,
                kernel_size=depth_kernel,
                padding=1,
                groups=scale_dim,
                bias=False,
            ),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=2, w=2),
        )

        self.down_proj = nn.Sequential(
            nn.Linear(scale_dim, dim),
            Rearrange('b n c -> b c n'),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            Rearrange('b c n -> b n c'),
        )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x


class LCAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (nn.Sequential(nn.Linear(inner_dim, dim),
                                     nn.Dropout(dropout))
                       if project_out else nn.Identity())

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q[:, :, -1, :].unsqueeze(2)  # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self,
                 feats: int,
                 mlp_hidden: int,
                 head: int = 8,
                 dropout: float = 0.0):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats: int, head: int = 8, dropout: float = 0.0):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head,
                           self.feats // self.head).transpose(1, 2)
        k = self.k(x).view(b, n, self.head,
                           self.feats // self.head).transpose(1, 2)
        v = self.v(x).view(b, n, self.head,
                           self.feats // self.head).transpose(1, 2)

        score = F.softmax(torch.einsum('bhif, bhjf->bhij', q, k) / self.sqrt_d,
                          dim=-1)  # (b,h,n,n)
        attn = torch.einsum('bhij, bhjf->bihf', score, v)  # (b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o


if __name__ == '__main__':
    b, n, f = 4, 16, 128
    x = torch.randn(b, n, f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n, f))
    # out = net(x)
    # print(out.shape)
