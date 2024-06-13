
import argparse
import numpy as np
import torch
import mlx.core as mx
import os

def convert_weights(model_path, output_path):
    print(f"Loading PyTorch weights from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    
    new_weights = {}
    
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.numpy()
            
        # Handle different layer types based on name and shape
        # Conv2d weights: (Out, In, H, W) -> (Out, H, W, In)
        # We identify Conv2d weights by checking specific layers or shape + name
        # Common Conv2d layers in DINOv2: patch_embed.proj.weight
        # Common Conv2d layers in DPTHead: projects, resize_layers, scratch (layerX_rn, refinenet, output_conv)
        
        # Linear weights: (Out, In) -> (In, Out)
        # Common Linear: attn.qkv, attn.proj, mlp.fc1, mlp.fc2, w12, w3, readout_projects
        
        # General heuristic:
        # If ndim=4 (Conv2d weight): permute (0, 2, 3, 1) to match MLX (Out, H, W, In) NO -> MLX Conv2d weights are (Out, H, W, In)
        # Wait, MLX Conv2d: 
        # "The weight shape is [offset_dim, kernel_size[0], ..., kernel_size[N], input_channels]" NO
        # Checking MLX docs: 
        # For Conv2d: "The weight parameter of shape (out_channels, *kernel_size, in_channels)"
        # PyTorch Conv2d: (out_channels, in_channels, *kernel_size)
        # So conversion is: permute(0, 2, 3, 1).
        
        # If ndim=2 (Linear weight): transpose(1, 0).
        # But be careful: 
        # pos_embed is (1, N, D) - do not touch.
        # cls_token is (1, 1, D) - do not touch.
        # gamma in LayerScale (dim,) - do not touch.
        # biases (dim,) - do not touch.
        
        # Specific names to verify:
        if "pos_embed" in k or "cls_token" in k or "register_tokens" in k or "mask_token" in k:
            new_weights[k] = v
        elif "gamma" in k: # LayerScale
            new_weights[k] = v
        elif "bias" in k: # Biases are usually 1D
             # Remap Sequential keys
             if "output_conv2.0" in k:
                 k = k.replace("output_conv2.0", "output_conv2.layers.0")
             elif "output_conv2.2" in k:
                 k = k.replace("output_conv2.2", "output_conv2.layers.2")
             
             # Also readout_projects.X.0 etc
             if "readout_projects" in k:
                 # Pattern: readout_projects.0.0.weight -> readout_projects.0.layers.0.weight
                 parts = k.split('.')
                 # parts: ['depth_head', 'readout_projects', '0', '0', 'weight']
                 # We need: ['depth_head', 'readout_projects', '0', 'layers', '0', 'weight']
                 # But MLX readout_projects is a list of Sequential?
                 # No, my implementation: self.readout_projects = [nn.Sequential(...)]
                 # So list item '0' is a Sequential.
                 # So key needs to be '...readout_projects.0.layers.0...' check this.
                 
                 # Let's inspect structure of sequential in list.
                 # List item name is just index integer in path?
                 # In MLX, if I have [Sequential(), ...], load_weights expects "0.layers.0..."?
                 # Yes.
                 # PyTorch: "readout_projects.0.0.weight" (List item 0, Sequential item 0)
                 # MLX: "readout_projects.0.layers.0.weight" (List item 0, Sequential layer 0)
                 k = k.replace(f".{parts[2]}.0.", f".{parts[2]}.layers.0.")
                 k = k.replace(f".{parts[2]}.1.", f".{parts[2]}.layers.1.")
                 
             new_weights[k] = v
        elif v.ndim == 4: # Conv2d weight
             print(f"Converting Conv2d: {k} {v.shape}")
             # Remap Sequential keys for Conv2d as well
             if "output_conv2.0" in k:
                 k = k.replace("output_conv2.0", "output_conv2.layers.0")
             elif "output_conv2.2" in k:
                 k = k.replace("output_conv2.2", "output_conv2.layers.2")
                 
             if "readout_projects" in k:
                 # But readout projects uses Linear! So Conv2d logic won't trigger there.
                 pass
             
             # PyTorch: (Out, In, H, W) -> MLX: (Out, H, W, In)
             # BUT ConvTranspose2d is (In, Out, H, W) -> MLX: (Out, H, W, In)
             if "resize_layers.0" in k or "resize_layers.1" in k:
                 print(f"Converting ConvTranspose2d: {k} {v.shape}")
                 # PyTorch (In, Out, H, W) -> MLX (Out, H, W, In) permutation: (1, 2, 3, 0)
                 new_weights[k] = v.transpose(1, 2, 3, 0)
             else:
                 new_weights[k] = v.transpose(0, 2, 3, 1)
        elif v.ndim == 2: # Linear weight
             print(f"Converting Linear: {k} {v.shape}")
             if "readout_projects" in k:
                 # Remap logic
                 parts = k.split('.')
                 # Assuming format depth_head.readout_projects.0.0.weight
                 try:
                     layer_idx = parts[3] # should be 0 or 1
                     k = k.replace(f".{parts[2]}.{layer_idx}.", f".{parts[2]}.layers.{layer_idx}.")
                 except:
                     pass
                     
             # PyTorch: (Out, In) -> MLX: (Out, In) (No transpose needed)
             new_weights[k] = v
        else:
             print(f"Keeping as is: {k} {v.shape}")
             new_weights[k] = v
             
    print(f"Saving converted weights to {output_path}")
    np.savez(output_path, **new_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to PyTorch .pth file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to output .npz file")
    args = parser.parse_args()
    
    convert_weights(args.model_path, args.output_path)
