import os
import cv2
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_depth_anything_v2.model import DepthAnythingV2

# Configuration
CAMERA_ID = 0
INPUT_SIZE = 518
ENCODER = 'vits'
MODEL_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'depth_anything_v2_vits.npz')
WINDOW_SIZE = (1280, 720)
PROJECTION_MODE = 'ortho' # 'ortho' or 'perspective'

# OpenGL / PyGame Imports
try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("Error: PyGame or PyOpenGL not found. Please install them:")
    print("pip install pygame PyOpenGL PyOpenGL_accelerate")
    exit(1)

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

class PointCloudApp:
    def __init__(self):
        self.mode = PROJECTION_MODE
        
        # Camera Control State
        self.cam_dist = 2.0  # Closer default
        self.cam_rot_x = 0.0
        self.cam_rot_y = 0.0
        self.cam_pos = [0, 0, 0]
        self.fov_scale = 0.6 # Default approx 60 deg vert
        self.ortho_zoom = 1.0 
        self.z_scale = 3.0 # Increase default depth scale for visibility
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        self.init_pygame()
        self.init_model()
        self.init_camera()
        
    def init_pygame(self):
        pygame.init()
        pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Depth Anything V2 - Live Point Cloud")
        
        glEnable(GL_DEPTH_TEST)
        glPointSize(3.0) 
        self.set_projection()

    def set_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        if self.mode == 'ortho':
            # Orthographic: Parallel projection
            # Adjust scaling to fit the [-1, 1] grid nicely
            scale = 2.0 / self.ortho_zoom
            aspect = WINDOW_SIZE[0] / WINDOW_SIZE[1]
            if aspect >= 1.0:
                glOrtho(-scale * aspect, scale * aspect, -scale, scale, -100.0, 100.0)
            else:
                glOrtho(-scale, scale, -scale / aspect, scale / aspect, -100.0, 100.0)
        else:
            # Perspective
            gluPerspective(45, (WINDOW_SIZE[0] / WINDOW_SIZE[1]), 0.1, 50.0)
            
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)

    # ...

    def run(self):
        # ...
        
        while True:
            # ... (capture and inference unchanged)
            
            # INVERSE DEPTH MAPPING
            d_min, d_max = depth_small.min(), depth_small.max()
            d_norm = (depth_small - d_min) / (d_max - d_min + 1e-6)
            
            if self.mode == 'ortho':
                # Orthographic / Height Map Mode
                # Use Z-Scale here
                z = d_norm * self.z_scale 
                scale = 1.0 
            else:
                # Perspective Mode
                z = 1.0 / (d_norm + 0.1)
                scale = self.fov_scale
            
            # Debug stats every 60 frames
            frame_count += 1
            if frame_count % 60 == 0:
                print(f"FPS: {clock.get_fps():.1f} | Pts: {target_w*target_h} | Mode: {self.mode} | Z-Scale: {self.z_scale:.1f}")

            xx, yy = np.meshgrid(np.linspace(-1, 1, target_w), np.linspace(-1, 1, target_h))
            
            if self.mode == 'ortho':
                points_x = xx * 2.0 # Widen grid to fill view
                points_y = -yy * 2.0
                points_z = z * 0.5 # Apply base scaling
            else:
                 points_x = xx * z * scale
                 points_y = -yy * z * scale
                 points_z = -z

            # ... (rendering)
        
    def init_model(self):
        print("Initializing MLX Model...")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.model = DepthAnythingV2(**model_configs[ENCODER])
        weights = mx.load(MODEL_WEIGHTS)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        print("Model Loaded.")

    def init_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {CAMERA_ID}")
            pygame.quit()
            exit(1)
            
    def process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4: # Scroll Up
                    if self.mode == 'ortho':
                        self.ortho_zoom *= 1.1
                        self.set_projection()
                    else:
                        self.cam_dist -= 0.2
                elif event.button == 5: # Scroll Down
                    if self.mode == 'ortho':
                        self.ortho_zoom /= 1.1
                        self.set_projection()
                    else:
                        self.cam_dist += 0.2
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    mx, my = pygame.mouse.get_pos()
                    dx = mx - self.last_mouse_pos[0]
                    dy = my - self.last_mouse_pos[1]
                    self.cam_rot_y += dx * 0.5
                    self.cam_rot_x += dy * 0.5
                    self.last_mouse_pos = (mx, my)
        
        keys = pygame.key.get_pressed()
        if keys[K_w]: self.cam_pos[2] += 0.05
        if keys[K_s]: self.cam_pos[2] -= 0.05
        if keys[K_a]: self.cam_pos[0] += 0.05
        if keys[K_d]: self.cam_pos[0] -= 0.05
        if keys[K_UP]: self.fov_scale += 0.01
        if keys[K_DOWN]: self.fov_scale -= 0.01
        
        # Z Scale adjustment
        if keys[K_z]: self.z_scale += 0.1
        if keys[K_x]: self.z_scale -= 0.1
        
        if keys[K_q]: return False
        
        if keys[K_o]:
             # Simple debounce check or just toggle rapidly (user handles)
             # Let's check event based instead later, but explicit is better
             self.mode = 'perspective' if self.mode == 'ortho' else 'ortho'
             self.set_projection()
             pygame.time.wait(200) # Debounce
             print(f"Switched to {self.mode} mode")
        
        return True

    def draw_grid(self):
        glBegin(GL_LINES)
        glColor3f(0.5, 0.5, 0.5)
        # Draw a small grid on XZ plane (which is XY here locally?)
        # Our world: X right, Y up (inverted), Z into screen
        # Draw grid at Z=0
        for i in range(-5, 6):
            glVertex3f(i, 5, 0)
            glVertex3f(i, -5, 0)
            glVertex3f(5, i, 0)
            glVertex3f(-5, i, 0)
        glEnd()

    def run(self):
        clock = pygame.time.Clock()
        frame_count = 0
        
        # Set background color
        glClearColor(0.2, 0.2, 0.2, 1.0)
        
        while True:
            if not self.process_input():
                break
                
            ret, frame = self.cap.read()
            if not ret: 
                print("Warning: Failed to grab frame from camera.", end='\r')
                pygame.time.wait(100)
                continue
            
            # Flip for mirror effect if webcam
            frame = cv2.flip(frame, 1)
            
            # 1. Inference
            image, (orig_h, orig_w) = preprocess(frame, INPUT_SIZE)
            x_mlx = mx.array(image)[None]
            depth = self.model(x_mlx)
            mx.eval(depth)
            
            # 2. Process Depth & Color for Point Cloud
            depth_np = np.array(depth[0]) # (H, W)
            
            # Resize depth to match a simpler grid for visualization performance
            target_w = 256
            target_h = int(target_w * orig_h / orig_w)
            
            depth_small = cv2.resize(depth_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            color_small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            color_small = cv2.cvtColor(color_small, cv2.COLOR_BGR2RGB) / 255.0
            
            # INVERSE DEPTH MAPPING
            d_min, d_max = depth_small.min(), depth_small.max()
            d_norm = (depth_small - d_min) / (d_max - d_min + 1e-6)
            
            if self.mode == 'ortho':
                # Orthographic / Height Map Mode
                # Use Z-Scale here
                z = d_norm * self.z_scale 
                scale = 1.0 
            else:
                # Perspective Mode
                z = 1.0 / (d_norm + 0.1)
                scale = self.fov_scale
            
            # Debug stats every 60 frames
            frame_count += 1
            if frame_count % 60 == 0:
                print(f"FPS: {clock.get_fps():.1f} | Pts: {target_w*target_h} | Mode: {self.mode} | Z-Scale: {self.z_scale:.1f}")

            xx, yy = np.meshgrid(np.linspace(-1, 1, target_w), np.linspace(-1, 1, target_h))
            
            if self.mode == 'ortho':
                points_x = xx * 2.0 # Widen grid to fill view
                points_y = -yy * 2.0
                points_z = z * 0.5 
            else:
                 points_x = xx * z * scale
                 points_y = -yy * z * scale
                 points_z = -z

            pts = np.stack([points_x, points_y, points_z], axis=-1).reshape(-1, 3)
            # Ensure contiguous float32 for OpenGL
            pts = np.ascontiguousarray(pts, dtype=np.float32)
            
            clrs = color_small.reshape(-1, 3)
            clrs = np.ascontiguousarray(clrs, dtype=np.float32)
            
            # 3. Render
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            glTranslatef(self.cam_pos[0], self.cam_pos[1], -self.cam_dist + self.cam_pos[2])
            glRotatef(self.cam_rot_x, 1, 0, 0)
            glRotatef(self.cam_rot_y, 0, 1, 0)
            
            # Draw Grid to verify visual context
            self.draw_grid()
            
            # Draw Points
            glPointSize(4.0)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            
            glVertexPointer(3, GL_FLOAT, 0, pts)
            glColorPointer(3, GL_FLOAT, 0, clrs)
            
            glDrawArrays(GL_POINTS, 0, len(pts))
            
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            
            pygame.display.flip()
            clock.tick(60)

        self.cap.release()
        pygame.quit()

if __name__ == '__main__':
    PointCloudApp().run()
