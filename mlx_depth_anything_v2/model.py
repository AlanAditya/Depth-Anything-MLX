
import math
from typing import List, Tuple, Union, Optional, Callable
from functools import partial

import mlx.core as mx
import mlx.nn as nn

def resize(x, size, method='linear', align_corners=False):
    # x shape (B, H, W, C) or (H, W, C)? MLX conv expects (B, H, W, C).
    # nn.Upsample operates on the spatial dimensions.
    # Assuming x is (B, H, W, C).
    
    B, H, W, C = x.shape
    target_h, target_w = size
    
    scale_h = target_h / H
    scale_w = target_w / W
    
    # Check if we can use integer scaling to be precise?
    # nn.Upsample in MLX might support size directly? Check source?
    # Signature said scale_factor.
    
    upsample = nn.Upsample(scale_factor=(scale_h, scale_w), mode=method, align_corners=align_corners)
    return upsample(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x):
        return self.proj(x) # (B, H, W, C)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def __call__(self, x):
        x12 = self.w12(x)
        x1, x2 = mx.split(x12, 2, axis=-1)
        hidden = nn.silu(x1) * x2
        return self.w3(hidden)

class SwiGLUFFNFused(SwiGLUFFN):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=None, drop=0.0):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            drop=drop
        )

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = mx.full((dim,), init_values)

    def __call__(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, proj_bias=True, ffn_bias=True, 
                 drop=0.0, attn_drop=0.0, init_values=None, drop_path=0.0, act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, ffn_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dims=dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = nn.Identity() # DropPath is identity in inference

        self.norm2 = norm_layer(dims=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        ) # Note: ffn_bias is handled in Mlp/SwiGLU if implemented correctly, but stock Mlp uses bias by default. 
          # We need to ensure we pass bias setting if needed, but simple Mlp above has bias=True default.
        
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = nn.Identity()

    def __call__(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=768, depth=12, num_heads=12, 
                 mlp_ratio=4.0, qkv_bias=True, proj_bias=True, ffn_bias=True, init_values=None, 
                 drop_path_rate=0.0, ffn_layer="mlp", num_register_tokens=0):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_register_tokens = num_register_tokens

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, num_patches + self.num_tokens, embed_dim))
        self.mask_token = mx.zeros((1, embed_dim))
        self.register_tokens = mx.zeros((1, num_register_tokens, embed_dim)) if num_register_tokens else None

        if ffn_layer == "mlp":
            ffn_layer_class = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer_class = SwiGLUFFNFused
        else:
             ffn_layer_class = Mlp # Fallback

        dpr = [x.item() for x in mx.linspace(0, drop_path_rate, depth)]

        self.blocks = [
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias,
                ffn_bias=ffn_bias, drop_path=dpr[i], norm_layer=partial(nn.LayerNorm, eps=1e-6), ffn_layer=ffn_layer_class,
                init_values=init_values
            )
            for i in range(depth)
        ]
        self.norm = nn.LayerNorm(dims=embed_dim, eps=1e-6)
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        
        if npatch == N and w == h:
            return self.pos_embed
            
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        
        w0 = w // self.patch_size[0]
        h0 = h // self.patch_size[1]
        
        # We add a small number to avoid floating point error in the interpolation
        # See discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        
        sqrt_N = int(math.sqrt(N))
        
        # MLX resize operates on spatial dimensions (H, W)
        # patch_pos_embed is (1, N, dim) -> (1, sqrt_N, sqrt_N, dim)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_N, sqrt_N, dim)
        
        # Target size
        target_h = int(h // self.patch_size[1])
        target_w = int(w // self.patch_size[0])
        
        # Interpolate
        # Note: PyTorch uses bicubic. MLX 'bicubic' support might vary, using 'bicubic' string if valid, else 'linear'
        # Trying bicubic
        patch_pos_embed = resize(patch_pos_embed, (target_h, target_w), method='cubic')
        
        patch_pos_embed = patch_pos_embed.reshape(1, -1, dim)
        return mx.concatenate((class_pos_embed[:, None, :], patch_pos_embed), axis=1)

    def forward_features(self, x):
        B, H, W, C = x.shape
        x = self.patch_embed(x) # (B, H_patches, W_patches, embed_dim)
        
        # Flatten patches
        B, patch_H, patch_W, dim = x.shape
        x = x.reshape(B, -1, dim)
        
        # Add cls token
        cls_tokens = mx.repeat(self.cls_token, repeats=B, axis=0)
        x = mx.concatenate([cls_tokens, x], axis=1)
        
        # Add pos embed - interpolate
        x = x + self.interpolate_pos_encoding(x, W, H)
        
        if self.register_tokens is not None:
            reg_tokens = mx.repeat(self.register_tokens, repeats=B, axis=0)
            x = mx.concatenate([x[:, :1], reg_tokens, x[:, 1:]], axis=1)

        for blk in self.blocks:
            x = blk(x)
            
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1 + self.num_register_tokens:],
            "x_prenorm": x,
        }

    def get_intermediate_layers(self, x, n=1, return_class_token=False, norm=True):
        outputs = []
        
        # Manual forward pass copy
        B, H, W, C = x.shape
        x_emb = self.patch_embed(x)
        B, patch_H, patch_W, dim = x_emb.shape
        x_emb = x_emb.reshape(B, -1, dim)
        
        cls_tokens = mx.repeat(self.cls_token, repeats=B, axis=0)
        x_emb = mx.concatenate([cls_tokens, x_emb], axis=1)
        x_emb = x_emb + self.interpolate_pos_encoding(x_emb, W, H)
        
        # Registers
        if self.register_tokens is not None:
             reg_tokens = mx.repeat(self.register_tokens, repeats=B, axis=0)
             x_emb = mx.concatenate([x_emb[:, :1], reg_tokens, x_emb[:, 1:]], axis=1)
             
        # Run blocks
        total_block_len = len(self.blocks)
        
        # Adjust blocks_to_take indices to be absolute
        final_blocks_to_take = []
        if isinstance(n, int):
             final_blocks_to_take = list(range(total_block_len - n, total_block_len))
        else:
             final_blocks_to_take = sorted(n)
             
        curr_x = x_emb
        for i, blk in enumerate(self.blocks):
            curr_x = blk(curr_x)
            if i in final_blocks_to_take:
                outputs.append(curr_x)
                
        if norm:
            outputs = [self.norm(out) for out in outputs]
            
        # Split class tokens
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        
        # Reshape back to spacial
        outputs = [
            out.reshape(B, patch_H, patch_W, -1) for out in outputs
        ]
        
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        if self.bn:
            self.bn1 = nn.BatchNorm(features)
            self.bn2 = nn.BatchNorm(features)
        self.activation = activation

    def __call__(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn: out = self.bn1(out)
        
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn: out = self.bn2(out)
        
        return out + x # Skip connection


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.size = size

    def __call__(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = output + res
            
        output = self.resConfUnit2(output)
        
        # Interpolate
        # Use mx.image.resize or explicit bilinear upsample?
        # xs[0] is usually the spatial map we upscale from?
        # Actually in DPT, we upscale the output of this block usually, or we upscale before fusion?
        # In DPT `FeatureFusionBlock`, it interpolates `output` (which is previous stage) to the size of current stage or specific size.
        
        target_size = size if size is not None else self.size
        if target_size is None:
             # Default scale factor 2
             H, W = output.shape[1], output.shape[2]
             target_size = (H * 2, W * 2)

        # MLX: upscale
        # We need to ensure we match the shape. 
        # output is (B, H, W, C)
        # resize expects (H, W)
        output = resize(output, (target_size[0], target_size[1]), method='linear')
        
        output = self.out_conv(output)
        return output


class Scratch(nn.Module):
    def __init__(self, in_shape, out_shape, groups=1, expand=False):
        super().__init__()
        out_shape1 = out_shape
        out_shape2 = out_shape
        out_shape3 = out_shape
        out_shape4 = out_shape
        
        if expand:
            out_shape1 = out_shape
            out_shape2 = out_shape * 2
            out_shape3 = out_shape * 4
            out_shape4 = out_shape * 8
            
        self.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, padding=1, bias=False)

class DPTHead(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super().__init__()
        self.use_clstoken = use_clstoken
        
        self.projects = [
            nn.Conv2d(in_channels, out_channel, kernel_size=1) 
            for out_channel in out_channels
        ]
        
        self.resize_layers = [
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ]
        
        if use_clstoken:
            self.readout_projects = [
                nn.Sequential(
                    nn.Linear(2 * in_channels, in_channels),
                    nn.GELU()
                ) for _ in range(len(self.projects))
            ]
            
        self.scratch = Scratch(out_channels, features, groups=1, expand=False)
        
        self.scratch.refinenet1 = FeatureFusionBlock(features, nn.ReLU(), deconv=False, bn=use_bn, expand=False, align_corners=True)
        self.scratch.refinenet2 = FeatureFusionBlock(features, nn.ReLU(), deconv=False, bn=use_bn, expand=False, align_corners=True)
        self.scratch.refinenet3 = FeatureFusionBlock(features, nn.ReLU(), deconv=False, bn=use_bn, expand=False, align_corners=True)
        self.scratch.refinenet4 = FeatureFusionBlock(features, nn.ReLU(), deconv=False, bn=use_bn, expand=False, align_corners=True)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(head_features_2, 1, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Identity(),
        )

    def __call__(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                # x is tuple (patch_tokens, cls_token)
                x, cls_token = x[0], x[1]
                # x shape (B, H, W, C), cls (B, C)
                # Expand cls
                # readout = cls_token.unsqueeze(1).expand_as(x) -> needs careful dim matching
                # in original: cls_token was (B, C) -> unsqueeze(1) (B, 1, C) -> expand to (B, N, C)
                # But here x is already reshaped to (B, H, W, C)
                
                # MLX:
                readout = cls_token[:, None, None, :]
                readout = mx.repeat(readout, x.shape[1], axis=1)
                readout = mx.repeat(readout, x.shape[2], axis=2)
                x = self.readout_projects[i](mx.concatenate((x, readout), axis=-1))
            else:
                x = x[0] # assuming return_class_token=True was used so it's a tuple, we take patches
                
            # x is (B, H, W, C)
            # Projects expects (B, H, W, C) -> (B, H, W, out_c)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
            
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[1:3])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[1:3])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[1:3])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = resize(out, (int(patch_h * 14), int(patch_w * 14)), method='linear')
        out = self.scratch.output_conv2(out)
        
        return out


class DepthAnythingV2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False):
        super().__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 11], # Wait - original code said [4, 11, 17, 23]? Checking dinov2.py... 
            # Original dpt.py says 'vitl': [4, 11, 17, 23]
            'vitg': [9, 19, 29, 39]
        }
        
        # Override for implementation convenience if needed, but sticking to logic
        if encoder == 'vitl':
             self.intermediate_layer_idx[encoder] = [4, 11, 17, 23]

        self.encoder = encoder
        
        # Model config map
        model_configs = {
            'vits': {'embed_dim': 384, 'depth': 12, 'num_heads': 6, 'init_values': 1.0},
            'vitb': {'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'init_values': 1.0},
            'vitl': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'init_values': 1.0},
            'vitg': {'embed_dim': 1536, 'depth': 40, 'num_heads': 24, 'init_values': 1.0, 'ffn_layer': 'swiglufused'}
        }
        
        cfg = model_configs[encoder]
        self.pretrained = DinoVisionTransformer(
            embed_dim=cfg['embed_dim'], 
            depth=cfg['depth'], 
            num_heads=cfg['num_heads'],
            init_values=cfg.get('init_values', 1.0),
            ffn_layer=cfg.get('ffn_layer', 'mlp')
        )
        
        self.depth_head = DPTHead(cfg['embed_dim'], features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def __call__(self, x):
        # x shape (B, H, W, C)
        patch_h = x.shape[1] // 14
        patch_w = x.shape[2] // 14
        
        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder], return_class_token=True
        )
        
        depth = self.depth_head(features, patch_h, patch_w)
        depth = nn.relu(depth)
        
        return depth.squeeze(axis=-1) # (B, H, W)
