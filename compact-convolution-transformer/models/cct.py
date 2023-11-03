import torch
from torch import nn, einsum
from timm.models import register_model
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .modules import Attention, PreNorm, FeedForward



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# compare with nn.Conv2D, add more prior knowledge without extra computation
class RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        #nn.init.zeros_(self.convmap.weight)
        self.bias = None#nn.Parameter(torch.zeros(out_channels), requires_grad=True)     # must have a bias for identical initialization
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups, bias=self.bias)



class ConvEmbed(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=7, stride=2, padding=3, pool_kernel_size=3, pool_stride=2,
                 pool_padding=1):
        super(ConvEmbed, self).__init__()
        self.conv_layers = nn.Sequential(
            RepConv(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                    padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
            # flatten start_dim=2 --------> transpose(-2, -1)
            Rearrange('b d h w -> b (h w) d')
        )

        self.apply(self.init_weight)

    def forward(self, x):
        return self.conv_layers(x)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]


    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class CompactTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim=768, depth=12, heads=12, pool='cls', in_channels=3,
                 dim_head=64, dropout=0.1, emb_dropout=0.1, scale_dim=4, conv_embed=False, **kwargs):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        if conv_embed:
            self.to_patch_embedding = ConvEmbed(in_channels, dim)
            num_patches = self.to_patch_embedding.sequence_length()
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, dim),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.2)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.pool = nn.Linear(dim, 1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.apply(self.init_weight)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        g = self.pool(x)
        xl = F.softmax(g, dim=1)
        x = einsum('b n l, b n d -> b l d', xl, x)

        return self.mlp_head(x.squeeze(-2))


    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


@register_model
def CCT_Model(num_classes, **kwargs):
    image_size = 224
    patch_size = 16
    model = CompactTransformer(image_size, patch_size, num_classes)
    return model