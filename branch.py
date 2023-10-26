import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)


    def forward(self,  x,Q):
        B, N, C = Q.shape

        q = self.qkv1(x[:,0]).reshape(B, 1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = Q.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = Q.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], k[0],  v[0]

        attn = (q @ k.transpose(-2, -1).detach()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v.detach()).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.LeakyReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.LeakyReLU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

    def forward(self, x,q):
        p = self.attn(self.norm1(x),q)
        return p


class Branch(nn.Module):
    def __init__(self, num_classes=30,
                 embed_dim=768,num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None,  drop_ratio=0.,
                 attn_drop_ratio=0.,
                 act_layer=None):

        super(Branch, self).__init__()
        self.fc1=nn.Sequential(
            nn.Linear(768,768)
        )
        self.fc2=nn.Sequential(
            nn.Linear(768,num_classes)
        )
        self.block1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, act_layer=act_layer)

        self.block2 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, act_layer=act_layer)


        self.apply(_init_vit_weights)

    def forward(self, feat1,feautre_negative,feautre_positive):
        k1=F.normalize(feautre_positive, dim=-1)
        k2=F.normalize(feautre_negative, dim=-1)
        q_q=self.block1(feat1.unsqueeze(1),feautre_negative)
        f_q=self.block1(feat1.unsqueeze(1),feautre_positive)
        con_q=feat1+q_q.squeeze(1)+f_q.squeeze(1)
        l_pos = torch.einsum('ab,acb->ac', [F.normalize(feat1, dim=-1), k1.detach()])
        l_neg = torch.einsum('ab,acb->ac', [F.normalize(feat1, dim=-1), k2.detach()])
        multi_feature = torch.cat([l_pos, l_neg], dim=1)
        cla1=self.fc2(con_q)
        return multi_feature, cla1

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def branch(num_classes):
    model = Branch(num_classes=num_classes)
    return model
