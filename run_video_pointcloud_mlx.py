import os
import cv2
import numpy as np
import mlx.core as mx
import argparse
import sys
import warnings
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from mlx_depth_anything_v2.model import DepthAnythingV2

# Constants / Defaults
INPUT_SIZE = 518
ENCODER = 'vits'
MODEL_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'depth_anything_v2_vits.npz')
WINDOW_SIZE = (1280, 720)

def preprocess(image, input_size=518):
    # Resize to multiple of 14 for model
    h, w = image.shape[:2]
    scale = input_size / max(h, w)
    new_h = int(np.round(h * scale / 14) * 14)
    new_w = int(np.round(w * scale / 14) * 14)
    
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    image_norm = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_norm - mean) / std
    
    return image_norm, (h, w)

class VideoPointCloudApp:
    def __init__(self, video_path):
        self.video_path = video_path
        
        # Approximate camera matrix
        fx = 1000.0  # Guess focal length
        fy = 1000.0
        cx = WINDOW_SIZE[0] / 2
        cy = WINDOW_SIZE[1] / 2
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)
        
        # Camera Control State (virtual view camera)
        self.cam_dist = 5.0
        self.cam_rot_x = 0.0
        self.cam_rot_y = 0.0
        self.cam_pos = [0, 0, 0]
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        # Point Cloud Accumulation
        self.global_pts = []
        self.global_clrs = []
        self.max_points = 200000 # Limit to prevent out-of-memory/crashing and blob mess
        
        # Tracking State
        self.global_pose = np.eye(4, dtype=np.float32)
        self.prev_gray = None
        self.prev_pts_2d = None
        self.prev_pts_3d = None
        self.current_depth_map = None # Cached unprojected local depths
        self.frame_idx = 0
        
        self.init_pygame()
        self.init_model()
        self.init_video()

    def init_pygame(self):
        pygame.init()
        pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Video Point Cloud Accumulation")
        
        glEnable(GL_DEPTH_TEST)
        glPointSize(2.0)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (WINDOW_SIZE[0] / WINDOW_SIZE[1]), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def init_model(self):
        print("Initializing MLX Model...")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        self.model = DepthAnythingV2(**model_configs[ENCODER])
        weights = mx.load(MODEL_WEIGHTS)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        print("Model Loaded.")

    def init_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            pygame.quit()
            sys.exit(1)
            
        # Get properties to estimate camera matrix (K)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Approximate Focal Length
        f = max(w, h)
        self.K = np.array([
            [f, 0, w/2],
            [0, f, h/2],
            [0, 0, 1]
        ], dtype=np.float64)

    def process_input(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:
                    self.cam_dist -= 0.5
                elif event.button == 5:
                    self.cam_dist += 0.5
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    # Clear accumulated points
                    self.global_pts = []
                    self.global_clrs = []
                    self.global_pose = np.eye(4, dtype=np.float32)
                    self.prev_gray = None
            elif event.type == MOUSEMOTION:
                if self.mouse_dragging:
                    mx, my = pygame.mouse.get_pos()
                    dx = mx - self.last_mouse_pos[0]
                    dy = my - self.last_mouse_pos[1]
                    self.cam_rot_y += dx * 0.5
                    self.cam_rot_x += dy * 0.5
                    self.last_mouse_pos = (mx, my)
        
        keys = pygame.key.get_pressed()
        base_speed = 0.1
        if keys[K_LSHIFT]: base_speed = 0.5
        if keys[K_w]:  # Move forward relative to view
            self.cam_pos[2] += base_speed
        if keys[K_s]:
            self.cam_pos[2] -= base_speed
        if keys[K_a]:
            self.cam_pos[0] += base_speed
        if keys[K_d]:
            self.cam_pos[0] -= base_speed
        if keys[K_q]:
            return False
            
        return True

    def get_3d_points(self, pts_2d, depth_map):
        pts_3d = []
        valid_2d = []
        h, w = depth_map.shape
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        for pt in pts_2d:
            x, y = pt.ravel()
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                z = depth_map[iy, ix]
                if z > 0.1 and z < 100.0:  # Valid depth range
                    X = (x - cx) * z / fx
                    Y = (y - cy) * z / fy
                    pts_3d.append([X, Y, z])
                    valid_2d.append([x, y])
                    
        return np.array(pts_3d, dtype=np.float32), np.array(valid_2d, dtype=np.float32).reshape(-1, 1, 2)

    def track_motion(self, frame_bgr):
        # Convert to grayscale for tracking
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # We need a depth map to initialize tracking features
        if self.current_depth_map is None:
            self.prev_gray = gray
            return
        
        if self.prev_gray is None or self.prev_pts_2d is None or len(self.prev_pts_2d) < 100 or self.frame_idx % 15 == 0:
            self.prev_gray = gray
            # Detect new features to track
            new_pts = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=2000, qualityLevel=0.01, minDistance=10)
            if new_pts is not None:
                self.prev_pts_3d, self.prev_pts_2d = self.get_3d_points(new_pts, self.current_depth_map)
            else:
                self.prev_pts_2d = None
                self.prev_pts_3d = None
            return
            
        if self.prev_pts_2d is None or len(self.prev_pts_2d) == 0:
            self.prev_gray = gray
            return
            
        # Optical Flow
        curr_pts_2d, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts_2d, None, winSize=(21, 21), maxLevel=3)
        
        # Filter valid tracked points
        good_curr_2d = curr_pts_2d[st == 1]
        good_prev_3d = self.prev_pts_3d[st.ravel() == 1]
        
        if len(good_curr_2d) < 12:
            self.prev_gray = gray
            return
            
        # Check if there is actual movement to avoid accumulating noise on static cameras
        motions = np.linalg.norm(good_curr_2d - self.prev_pts_2d[st == 1], axis=1)
        median_motion = np.median(motions)
        
        # If the camera is essentially static, don't update pose
        if median_motion < 1.0:
            self.prev_gray = gray
            self.prev_pts_2d = curr_pts_2d
            return
            
        # Compute Camera Pose using solvePnP
        # Assuming the initial features form our "world" coordinate system for this step
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            good_prev_3d, good_curr_2d, self.K, None,
            iterationsCount=100, reprojectionError=5.0, confidence=0.99
        )
        
        if success and inliers is not None and len(inliers) > 8:
            # We found the transform FROM the local 3D points TO the current camera
            R_cam, _ = cv2.Rodrigues(rvec)
            t_cam = tvec.ravel()
            
            # Construct 4x4 matrix for camera moving relative to previous frame's point cloud
            T_rel = np.eye(4, dtype=np.float32)
            
            if not np.isnan(R_cam).any() and not np.isnan(t_cam).any():
                # We need the relative movement of the camera, not the points
                # Transform point from world to cam: P_c = R * P_w + t
                # We want the transformation that updates the camera's global pose:
                # Camera in world (inverse of P_c transform):
                R_cam_inv = R_cam.T
                t_cam_inv = -R_cam_inv @ t_cam
                
                T_rel[:3, :3] = R_cam_inv
                T_rel[:3, 3] = t_cam_inv
                
                # Sanity check translation so we don't jump huge distances in one frame
                if np.max(np.abs(t_cam_inv)) < 5.0:  
                    # Update global rigid transformation
                    new_pose = self.global_pose @ T_rel
                    
                    # Normalize rotation matrix to prevent drift/skew
                    U, _, Vh = np.linalg.svd(new_pose[:3, :3])
                    new_pose[:3, :3] = U @ Vh
                    
                    if not np.isnan(new_pose).any() and not np.isinf(new_pose).any():
                        if np.max(np.abs(new_pose[:3, 3])) < 500.0:
                            self.global_pose = new_pose
            
        # Keep features updating frame-by-frame (visual tracking approach)
        # Note: True SLAM keeps a global map. We update 3D relative to previous frame.
        self.prev_gray = gray
        self.prev_pts_2d = curr_pts_2d

    def draw_axes(self):
        glBegin(GL_LINES)
        glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(1, 0, 0) # X Red
        glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, 1, 0) # Y Green
        glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, 1) # Z Blue
        glEnd()

    def run(self):
        clock = pygame.time.Clock()
        
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        while True:
            if not self.process_input():
                break
                
            ret, frame = self.cap.read()
            if not ret:
                print("\nEnd of video reached. You can still navigate the scene.")
                # Loop through pygame events to allow navigation without new frames
                while self.process_input():
                    self.render_scene()
                    clock.tick(60)
                break
                
            self.frame_idx += 1
            
            # Frame properties
            h_vid, w_vid = frame.shape[:2]
            
            # --- 1. Compute Depth ALWAYS (Needed for Tracking) ---
            print(f"Processing Frame {self.frame_idx}") # Debug
            
            image, (orig_h, orig_w) = preprocess(frame, INPUT_SIZE)
            x_mlx = mx.array(image)[None]
            depth = self.model(x_mlx)
            mx.eval(depth)
            
            depth_np = np.array(depth[0])
            
            # Resize depth exactly to frame size for pixel-to-pixel projection
            depth_full = cv2.resize(depth_np, (w_vid, h_vid), interpolation=cv2.INTER_LINEAR)
            
            # Inverse Depth Mapping to get Z values for tracking and pointcloud visualization
            d_min, d_max = depth_full.min(), depth_full.max()
            d_norm = (depth_full - d_min) / (d_max - d_min + 1e-6)
            d_norm = np.nan_to_num(d_norm, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Store scaled Z depth map for tracker
            self.current_depth_map = 3.0 / np.clip((d_norm + 0.1), 0.05, 1.5)
            
            # --- 2. Track Camera Motion (RGB-D) ---
            try:
                self.track_motion(frame)
            except Exception as e:
                print(f"Tracking error: {e}")
                
            # Skip point cloud accumulation for every N frames to save memory and performance
            if self.frame_idx % 5 == 0:
                
                # Downsample current depth map specifically for the point cloud visualization buffer
                target_w = 128 # Smaller grid
                target_h = int(target_w * orig_h / orig_w)
                
                z_small = cv2.resize(self.current_depth_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                color_small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                color_small = cv2.cvtColor(color_small, cv2.COLOR_BGR2RGB) / 255.0
                
                xx, yy = np.meshgrid(np.linspace(-1, 1, target_w), np.linspace(-1, 1, target_h))
                
                fov_scale = 1.0 # Adjust view scale X/Y
                local_x = xx * z_small * fov_scale
                local_y = -yy * z_small * fov_scale # Invert Y for correct orientation
                local_z = -z_small      # Negative Z is forward
                
                # Stack to (N, 3)
                pts_local = np.stack([local_x, local_y, local_z], axis=-1).reshape(-1, 3)
                pts_local = np.nan_to_num(pts_local, copy=False)
                clrs = color_small.reshape(-1, 3)
                
                # Transform to Global Coordinate Space
                # Ensure global_pose is float32 matrix
                transform = np.nan_to_num(self.global_pose.astype(np.float32), nan=0.0)
                
                # Apply transformation: pts_global = R * pts_local + t
                R_glob = transform[:3, :3]
                t_glob = transform[:3, 3]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pts_global = (pts_local @ R_glob.T) + t_glob
                
                # Check for NaNs
                if not np.isnan(pts_global).any() and not np.isinf(pts_global).any():
                    # Filter out unreasonably far points (e.g., sky or bad tracking)
                    valid_mask = np.linalg.norm(pts_global, axis=1) < 500.0
                    pts_global = pts_global[valid_mask]
                    clrs_valid = clrs[valid_mask].astype(np.float32)
                    
                    if len(pts_global) > 0:
                        # Ensure memory is contiguous C arrays for OpenGL Safety
                        pts_clean = np.ascontiguousarray(pts_global, dtype=np.float32)
                        clrs_clean = np.ascontiguousarray(clrs_valid, dtype=np.float32)
                        
                        # Add to accumulators
                        self.global_pts.append(pts_clean)
                        self.global_clrs.append(clrs_clean)
                        
                        # Truncate if too large
                        total_pts = sum([len(p) for p in self.global_pts])
                        while total_pts > self.max_points and len(self.global_pts) > 1:
                            removed_pts = len(self.global_pts[0])
                            self.global_pts.pop(0)
                            self.global_clrs.pop(0)
                            total_pts -= removed_pts
                        
                if self.frame_idx % 30 == 0:
                    print(f"Frame {self.frame_idx} | Pts: {sum([len(p) for p in self.global_pts])} | Pos: {self.global_pose[:3, 3].round(2)}")
            
            # Render
            self.render_scene()
            clock.tick(60)

        self.cap.release()
        pygame.quit()

    def render_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Apply camera transformations
        glTranslatef(self.cam_pos[0], self.cam_pos[1], -self.cam_dist + self.cam_pos[2])
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)
        
        self.draw_axes()
        
        # Draw all accumulated point clouds
        if len(self.global_pts) > 0:
            # Flatten lists into single arrays for faster rendering if we had VBOs
            # For immediate arrays, we can draw them batch by batch
            
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            
            for pts, clrs in zip(self.global_pts, self.global_clrs):
                # Extra array safety check
                if len(pts) > 0:
                    glVertexPointer(3, GL_FLOAT, 0, pts)
                    glColorPointer(3, GL_FLOAT, 0, clrs)
                    glDrawArrays(GL_POINTS, 0, len(pts))
                
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            
        pygame.display.flip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Depth Anything V2 Video Point Cloud")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
    args = parser.parse_args()
    
    app = VideoPointCloudApp(args.video)
    app.run()
