import argparse
import cv2
import numpy as np
import time
import os
import mlx.core as mx
import mlx.nn as nn
from mlx_depth_anything_v2.model import DepthAnythingV2


# Configuration
CAMERA_ID = 0
INPUT_SIZE = 518
ENCODER = 'vits' # choices: 'vits', 'vitb', 'vitl', 'vitg'
MODEL_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'depth_anything_v2_vits.npz')
GRAYSCALE = False

def preprocess(image, input_size=518):
    # Resize logic to ensure multiple of 14
    h, w = image.shape[:2]
    scale = input_size / max(h, w)
    new_h = int(np.round(h * scale / 14) * 14)
    new_w = int(np.round(w * scale / 14) * 14)
    
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    image_norm = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_norm - mean) / std
    
    return image_norm, (h, w)

def main():
    # Load Model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    print(f"Initializing model {ENCODER}...")
    model = DepthAnythingV2(**model_configs[ENCODER])
    
    print(f"Loading weights from {MODEL_WEIGHTS}...")
    weights = mx.load(MODEL_WEIGHTS)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    
    # Open Camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_ID}")
        return
        
    print("Starting inference... Press 'q' to exit.")
    
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    fps_avg = 0
    alpha = 0.9
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        start_time = time.time()
        
        # Preprocess
        image, (orig_h, orig_w) = preprocess(frame, INPUT_SIZE)
        
        # To MLX Tensor
        x = mx.array(image)[None]
        
        # Inference
        depth = model(x)
        mx.eval(depth)
        
        # Post-process
        depth_np = np.array(depth[0])
        
        # Resize back to original
        depth_resized = cv2.resize(depth_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-255
        depth_norm = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
        depth_uint8 = depth_norm.astype(np.uint8)
        
        if GRAYSCALE:
            depth_vis = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (cmap(depth_uint8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
        # FPS Calculation
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        fps_avg = alpha * fps_avg + (1 - alpha) * fps
        
        # Combine
        combined_result = cv2.hconcat([frame, depth_vis])
        
        # Add FPS text
        cv2.putText(combined_result, f"FPS: {fps_avg:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Depth Anything V2 - MLX Live', combined_result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
