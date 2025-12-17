from enum import IntEnum
from dataclasses import dataclass

class BackendPreference(IntEnum):
    PROPRIETARY = 0        # e.g CUDA / ROCm / MPS / XPU
    CPU         = 1
    VULKAN      = 2
    OPENCL      = 4

class RaftEnginePreference(IntEnum):
    TORCH    = 0
    TensorRT = 1
    # Not implemented
    NCNN     = 2
    ONNX     = 3

#@dataclass
class Settings:
    # VideoUtils
    defaultMaxFrameSize: int = 640
    # torch
    defaultBackendPreference: BackendPreference  = BackendPreference.PROPRIETARY
    # General
    defaultRaftEnginePreference = RaftEnginePreference.TensorRT
    defaultMaxFrameCount: int = 30;
    defaultMaxLostFrame: int = 1000;
    useOpticalFlow: bool = True;

#Settings = _Settings()
