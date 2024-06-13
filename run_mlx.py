
import argparse
import cv2
import glob
import numpy as np
import os
import mlx.core as mx
import mlx.nn as nn
from mlx_depth_anything_v2.model import DepthAnythingV2

def get_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2 MLX')
    parser.add_argument('--img-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth_mlx')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--model-weights', type=str, required=True, help='Path to converted .npz weights')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    return parser.parse_args()

def load_image(filepath):
    return cv2.imread(filepath)

def preprocess(image, input_size=518):
    # Resize logic to ensure multiple of 14
    h, w = image.shape[:2]
    scale = input_size / max(h, w)
    new_h = int(np.round(h * scale / 14) * 14)
    new_w = int(np.round(w * scale / 14) * 14)
    
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    return image, (h, w)

def main():
    args = get_args()
    
    # Load Model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    print(f"Initializing model {args.encoder}...")
    model = DepthAnythingV2(**model_configs[args.encoder])
    
    print(f"Loading weights from {args.model_weights}...")
    weights = mx.load(args.model_weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    
    # Process Images
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        if os.path.isdir(filename): continue
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = load_image(filename)
        if raw_image is None:
            print(f"Could not load {filename}")
            continue
            
        image, (orig_h, orig_w) = preprocess(raw_image, args.input_size)
        
        # To MLX Tensor (B, H, W, C)
        x = mx.array(image)[None]
        
        # Inference
        depth = model(x)
        mx.eval(depth)
        
        # Post-process
        # depth is (1, H, W)
        depth = np.array(depth[0])
        
        # Resize back to original
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-255
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((orig_h, 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)

if __name__ == '__main__':
    main()
