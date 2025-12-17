import cv2
import numpy as np

import torch
import torch.nn.functional as F

from ..Settings import *

def preprocess_frame1(frame: np.ndarray, max_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)

    if scale < 1.0:
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

    return frame

def preprocess_frame(frame: np.ndarray, max_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)

    frame_t = (
        torch.from_numpy(frame)
        .cuda()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )
    
    frame_t = F.interpolate(
        frame_t,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False
    )
    
    frame_resized = (
        frame_t.squeeze(0)
        .permute(1, 2, 0)
        .byte()
        .cpu()
        .numpy()
    )
    return frame_resized

def preprocess_frame2(frame: np.ndarray, max_size: int = Settings.defaultMaxFrameSize) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)

    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    gpu_resized = cv2.cuda.resize(
    gpu_frame,
    (int(w * scale), int(h * scale)),
    interpolation=cv2.INTER_LINEAR)

    return gpu_resized
