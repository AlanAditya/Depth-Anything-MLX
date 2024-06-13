#!/usr/bin/env python3
"""
Run Depth Anything 3 on macOS with MPS (Metal Performance Shaders).

Extracts frames from a video, runs DA3 depth + pose estimation,
and exports a 3D point cloud as a .glb file.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python run_da3_video_mps.py \\
        --video assets/examples_video/201423-915387343_medium.mp4

    # Use a bigger model (needs 16+ GB RAM):
    KMP_DUPLICATE_LIB_OK=TRUE python run_da3_video_mps.py \\
        --video assets/examples_video/201423-915387343_medium.mp4 \\
        --model depth-anything/DA3-BASE
"""

import os
import sys
import argparse
import glob

# ---------------------------------------------------------------------------
# 0) Environment fixes for macOS
# ---------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Make sure the DA3 source is on the path
DA3_SRC = os.path.join(os.path.dirname(__file__), "Depth-Anything-3", "src")
if DA3_SRC not in sys.path:
    sys.path.insert(0, DA3_SRC)

# ---------------------------------------------------------------------------
# 1) Monkey-patch torch.cuda.is_bf16_supported for MPS
#    DA3's forward() calls this — it returns False on non-CUDA, which is fine.
# ---------------------------------------------------------------------------
import torch

if not hasattr(torch.cuda, "_original_is_bf16_supported"):
    _orig = torch.cuda.is_bf16_supported
    def _safe_bf16(*a, **kw):
        try:
            return _orig(*a, **kw)
        except Exception:
            return False
    torch.cuda.is_bf16_supported = _safe_bf16

# ---------------------------------------------------------------------------
# 2) Patch autocast for MPS — DA3 uses device_type from tensor, which may be
#    'mps'. torch.autocast supports 'mps' in recent PyTorch but not bf16.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3) Now import DA3
# ---------------------------------------------------------------------------
from depth_anything_3.api import DepthAnything3

import cv2
import numpy as np


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0, start_frame: int = 0, max_frames: int = 0) -> list[str]:
    """Extract frames from video at the given FPS."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if (max_frames != 0):
        total_frames = max_frames
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Calculate frame interval
    if fps <= 0:
        fps = video_fps  # Use all frames
    frame_interval = max(1, int(round(video_fps / fps)))

    print(f"Video: {video_fps:.1f} FPS, {duration:.1f}s, {total_frames} frames")
    print(f"Extracting at {fps:.1f} FPS (every {frame_interval} frames)")

    saved_paths = []
    frame_idx = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(path, frame)
            saved_paths.append(path)
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(saved_paths)} frames to {output_dir}")
    return sorted(saved_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 — Video → 3D Point Cloud (macOS MPS)",
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--model", type=str, default="depth-anything/DA3-SMALL",
        help="HuggingFace model ID (default: depth-anything/DA3-SMALL)",
    )
    parser.add_argument(
        "--export-dir", type=str, default="DA3_Output",
        help="Output directory (default: DA3_Output)",
    )
    parser.add_argument(
        "--fps", type=float, default=1.0,
        help="Frames per second to sample from video (default: 1.0)",
    )
    parser.add_argument(
        "--export-format", type=str, default="glb",
        help="Export format: glb, npz, depth_vis, mini_npz (default: glb)",
    )
    parser.add_argument(
        "--process-res", type=int, default=0,
        help="Processing resolution (0 = auto based on frame count)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=20,
        help="Max frames to process (DA3 processes all at once; default: 20)",
    )
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Start frame to process (default: 0)",
    )
    args = parser.parse_args()

    # Validate video exists
    if not os.path.isfile(args.video):
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Metal (MPS) GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  MPS not available, falling back to CPU")

    # -----------------------------------------------------------------------
    # Step 1: Extract frames
    # -----------------------------------------------------------------------
    frames_dir = os.path.join(args.export_dir, "input_images")
    image_paths = extract_frames(args.video, frames_dir, args.fps, args.start_frame, args.max_frames)

    if len(image_paths) == 0:
        print("❌ No frames extracted!")
        sys.exit(1)

    # Cap frames to prevent MPS OOM (DA3 processes all views in one forward pass)
    if len(image_paths) > args.max_frames:
        step = len(image_paths) / args.max_frames
        indices = [int(i * step) for i in range(args.max_frames)]
        image_paths = [image_paths[i] for i in indices]
        print(f"⚠️  Subsampled to {len(image_paths)} frames (--max-frames {args.max_frames})")

    # Auto-select resolution based on frame count to stay within MPS memory
    if args.process_res <= 0:
        if len(image_paths) <= 10:
            process_res = 504
        elif len(image_paths) <= 20:
            process_res = 378
        else:
            process_res = 280
        print(f"📐 Auto resolution: {process_res}px (for {len(image_paths)} frames)")
    else:
        process_res = args.process_res

    # -----------------------------------------------------------------------
    # Step 2: Load model
    # -----------------------------------------------------------------------
    print(f"\n🔄 Loading model: {args.model}")
    print("   (This downloads weights from HuggingFace on first run)")

    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded!\n")

    # -----------------------------------------------------------------------
    # Step 3: Run inference
    # -----------------------------------------------------------------------
    print(f"🧠 Running inference on {len(image_paths)} frames...")

    prediction = model.inference(
        image=image_paths,
        export_dir=args.export_dir,
        export_format=args.export_format,
        process_res=process_res,
        num_max_points=1_000_000,
        conf_thresh_percentile=5.0,
        show_cameras=True,
        # Override the hard-coded conf_thresh=1.05 which drops too many points
        export_kwargs={"glb": {"conf_thresh": 0.0}},
    )

    # -----------------------------------------------------------------------
    # Step 4: Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("✅ Done!")
    print(f"   Depth maps shape : {prediction.depth.shape}")
    print(f"   Extrinsics shape : {prediction.extrinsics.shape}")
    print(f"   Intrinsics shape : {prediction.intrinsics.shape}")
    print(f"   Export dir       : {args.export_dir}")

    glb_path = os.path.join(args.export_dir, "scene.glb")
    if os.path.isfile(glb_path):
        size_mb = os.path.getsize(glb_path) / (1024 * 1024)
        print(f"   GLB file         : {glb_path} ({size_mb:.1f} MB)")
        print(f"\n🎉 Open scene.glb in Quick Look, Blender, or https://gltf-viewer.donmccurdy.com/")
    print("=" * 60)


if __name__ == "__main__":
    main()
