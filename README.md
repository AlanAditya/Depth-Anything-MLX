# Depth-Anything-MLX

An [MLX](https://github.com/ml-explore/mlx) port of [Depth Anything V2](https://depth-anything-v2.github.io) for fast, native monocular depth estimation on Apple Silicon, plus helper scripts for point-cloud visualization and [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) video support.

The original PyTorch model (`depth_anything_v2/`) is ported layer-by-layer to MLX (`mlx_depth_anything_v2/`), with a weight-conversion script to translate official `.pth` checkpoints into MLX-compatible `.npz` files.

## Project layout

- `mlx_depth_anything_v2/model.py` — MLX implementation of the DINOv2 encoder + DPT depth head.
- `convert_weights.py` — converts a PyTorch `.pth` checkpoint to an MLX `.npz` checkpoint (handles Conv2d/Linear layout differences, `Sequential` key remapping, etc.).
- `verify_port.py` — sanity-checks the MLX port's outputs against the original PyTorch model.
- `run_mlx.py` — run depth estimation on an image or directory of images using the MLX model.
- `run_camera_mlx.py` — live depth estimation from a webcam using the MLX model.
- `run_pointcloud_mlx.py` / `run_video_pointcloud_mlx.py` — render an interactive 3D point cloud from an image or video using the MLX model (PyGame + OpenGL).
- `run.py` / `run_video.py` / `app.py` — original PyTorch (CPU/CUDA/MPS) image, video, and Gradio demo entry points from Depth Anything V2.
- `run_da3_video_mps.py` — runs [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) on a video via PyTorch/MPS and exports a `.glb` point cloud.
- `checkpoints/` — place PyTorch `.pth` weights here; converted `.npz` weights are also stored here.

## Setup

```bash
pip install -r requirements.txt
pip install mlx
```

Point-cloud visualization scripts additionally need:

```bash
pip install pygame PyOpenGL PyOpenGL_accelerate
```

Download a Depth Anything V2 checkpoint (e.g. `depth_anything_v2_vits.pth`) from the [official releases](https://github.com/DepthAnything/Depth-Anything-V2) into `checkpoints/`.

Running `run_da3_video_mps.py` requires the [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) source under `Depth-Anything-3/` (its own `requirements.txt`), and the original `depth_anything_v2/` PyTorch package for `run.py`, `run_video.py`, and `app.py`.

## Converting weights to MLX

```bash
python convert_weights.py \
    --model-path checkpoints/depth_anything_v2_vits.pth \
    --output-path checkpoints/depth_anything_v2_vits.npz
```

## Usage

**Image inference (MLX):**

```bash
python run_mlx.py \
    --img-path assets/examples \
    --encoder vits \
    --model-weights checkpoints/depth_anything_v2_vits.npz \
    --outdir vis_depth_mlx
```

**Webcam inference (MLX):**

```bash
python run_camera_mlx.py
```

**3D point cloud from an image (MLX):**

```bash
python run_pointcloud_mlx.py
```

**3D point cloud from a video (MLX):**

```bash
python run_video_pointcloud_mlx.py --video path/to/video.mp4
```

**Depth Anything 3 on a video (PyTorch/MPS):**

```bash
KMP_DUPLICATE_LIB_OK=TRUE python run_da3_video_mps.py --video assets/examples_video/201423-915387343_medium.mp4
```

Encoder sizes (`--encoder`) available for Depth Anything V2: `vits`, `vitb`, `vitl`, `vitg`.

## Verifying the port

To compare MLX outputs against the original PyTorch model for correctness:

```bash
python verify_port.py
```

## Credits

Built on top of [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3). See [DA-2K.md](DA-2K.md) for details on the DA-2K evaluation benchmark.
