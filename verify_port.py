
import argparse
import cv2
import numpy as np
import torch
import mlx.core as mx
import mlx.nn as nn
from mlx_depth_anything_v2.model import DepthAnythingV2 as DepthAnythingV2_MLX
from depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2_PT
import os

def get_args():
    parser = argparse.ArgumentParser(description='Verify Depth Anything V2 Port')
    parser.add_argument('--img-path', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pt-weights', type=str, required=True, help='Path to PyTorch .pth weights')
    parser.add_argument('--mlx-weights', type=str, required=True, help='Path to MLX .npz weights')
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Load Image and Preprocess (Shared Logic)
    # We want to feed EXACTLY the same tensor to both to rule out preprocessing differences
    raw_image = cv2.imread(args.img_path)
    input_size = 518
    
    # Preprocess manually to get numpy array
    h, w = raw_image.shape[:2]
    scale = input_size / max(h, w)
    new_h = int(np.round(h * scale / 14) * 14)
    new_w = int(np.round(w * scale / 14) * 14)
    
    image = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Input for MLX: (1, H, W, C)
    input_mlx = mx.array(image)[None]
    
    # Input for PyTorch: (1, C, H, W)
    input_pt = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    
    # 2. Load Models
    print("Loading PyTorch model...")
    model_configs_pt = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model_pt = DepthAnythingV2_PT(**model_configs_pt[args.encoder])
    model_pt.load_state_dict(torch.load(args.pt_weights, map_location='cpu'))
    model_pt.eval()

    print("Loading MLX model...")
    model_configs_mlx = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model_mlx = DepthAnythingV2_MLX(**model_configs_mlx[args.encoder])
    weights = mx.load(args.mlx_weights)
    model_mlx.load_weights(list(weights.items()), strict=True)
    model_mlx.eval()

    # 2.5 Verification of Components
    print("\n--- Verifying Components ---")
    
    # Check PatchEmbed
    print("Checking PatchEmbed...")
    with torch.no_grad():
        # Need to capture intermediate usually, but here we run part manually
        pe_pt = model_pt.pretrained.patch_embed.proj(input_pt)
    
    # Manually run patch embed logic to match
    # PyTorch
    # pe_pt is (1, 384, 37, 37)
    pe_pt_np = pe_pt.permute(0, 2, 3, 1).detach().cpu().numpy() # (1, 37, 37, 384)
    
    # MLX
    pe_mlx = model_mlx.pretrained.patch_embed.proj(input_mlx) # (1, 37, 37, 384)
    pe_mlx_np = np.array(pe_mlx)
    
    print(f"PatchEmbed Proj Diff: {np.abs(pe_pt_np - pe_mlx_np).max():.6f}")
    
    # Check PosEmbed
    print("Checking PosEmbed...")
    pos_pt = model_pt.pretrained.pos_embed.detach().cpu().numpy()
    pos_mlx = np.array(model_mlx.pretrained.pos_embed)
    print(f"PosEmbed Diff: {np.abs(pos_pt - pos_mlx).max():.6f}")
    
    # Check Interpolated PosEmbed
    print("Checking Interpolated PosEmbed...")
    h_pixels, w_pixels = 518, 518
    
    # PT: x must be (B, N, C) for dim inference and patch count check
    # pe_pt is (1, 384, 37, 37)
    # We need (1, 1370, 384) (1369 patches + 1 cls)
    x_pt_dummy = pe_pt.flatten(2).transpose(1, 2)
    x_pt_dummy = torch.cat([torch.zeros(1, 1, 384), x_pt_dummy], dim=1)
    
    # MLX: x must be (B, N, C)
    # pe_mlx is (1, 37, 37, 384)
    B, H, W, C = pe_mlx.shape
    x_mlx_dummy = pe_mlx.reshape(B, -1, C)
    x_mlx_dummy = mx.concatenate([mx.zeros((1, 1, 384)), x_mlx_dummy], axis=1)
    
    with torch.no_grad():
        interp_pos_pt = model_pt.pretrained.interpolate_pos_encoding(x_pt_dummy, w_pixels, h_pixels)
    
    interp_pos_mlx = model_mlx.pretrained.interpolate_pos_encoding(x_mlx_dummy, w_pixels, h_pixels)
    
    interp_pos_pt_np = interp_pos_pt.detach().cpu().numpy()
    interp_pos_mlx_np = np.array(interp_pos_mlx)
    
    print(f"Interpolated PosEmbed Diff: {np.abs(interp_pos_pt_np - interp_pos_mlx_np).max():.6f}")
    
    # 3. Compare Intermediate Features (Backbone)
    print("\n--- Comparing Backbone Features (All Blocks) ---")
    
    all_indices = list(range(len(model_pt.pretrained.blocks))) # [0, 1, ..., 11]
    
    with torch.no_grad():
        pt_features = model_pt.pretrained.get_intermediate_layers(
            input_pt, all_indices, return_class_token=True, norm=True
        )
    
    mlx_features = model_mlx.pretrained.get_intermediate_layers(
        input_mlx, all_indices, return_class_token=True, norm=True
    )
    
    for i, (pt_feat, mlx_feat) in enumerate(zip(pt_features, mlx_features)):
        pt_patch, pt_cls = pt_feat
        mlx_patch, mlx_cls = mlx_feat
        
        pt_patch_np = pt_patch.cpu().numpy()
        pt_cls_np = pt_cls.cpu().numpy()
        mlx_patch_np = np.array(mlx_patch)
        
        if pt_patch_np.ndim == 3 and mlx_patch_np.ndim == 4:
             B, H, W, C = mlx_patch_np.shape
             mlx_patch_np = mlx_patch_np.reshape(B, -1, C)
        
        diff_patch = np.abs(pt_patch_np - mlx_patch_np).max()
        print(f"Block {i}: Patch Diff: {diff_patch:.6f}")
        
    # 4. Compare Final Output
    print("\n--- Running Final Head ---")
    with torch.no_grad():
        out_pt = model_pt(input_pt) # (B, H, W)
        
    out_pt_np = out_pt.numpy()
    
    out_mlx = model_mlx(input_mlx)
    mx.eval(out_mlx)
    out_mlx_np = np.array(out_mlx)
    
    print(f"PyTorch output shape: {out_pt_np.shape}")
    print(f"MLX output shape: {out_mlx_np.shape}")
    
    print(f"PyTorchStats: Min={out_pt_np.min():.4f}, Max={out_pt_np.max():.4f}, Mean={out_pt_np.mean():.4f}")
    print(f"MLX Stats:    Min={out_mlx_np.min():.4f}, Max={out_mlx_np.max():.4f}, Mean={out_mlx_np.mean():.4f}")
    
    diff = np.abs(out_pt_np - out_mlx_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"Max Absolute Difference: {max_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print("SUCCESS: Outputs are identical (within tolerance).")
    elif max_diff < 1e-1:
        print("WARNING: Small differences found. Could be interpolation/precision issues.")
    else:
        print("FAILURE: Significant differences found.")

if __name__ == '__main__':
    main()
