from pathlib import Path

# -------------------------------
# 1. Grad_CAM
# -------------------------------

from GradCAM.GradCAM import run_gradcam

image_list_fixed = [
    "cat.avif",
    "Resume.pdf"
]

bRunGradCam = False

if bRunGradCam:
    input_folder = Path(r".\inputs")
    image_list = [f.name for f in input_folder.iterdir() if f.is_file()]
    
    run_gradcam(image_list)

# -------------------------------
# 2. OptiTrack
# -------------------------------

from OptiTrack.Settings import *

from OptiTrack.OptiTrack import run_OptiTrack

input_video = 'input.mp4'
output_video = 'output.mp4'

import time

def profile_engine(engine, input_video, output_video, runs=10):
    Settings.defaultRaftEnginePreference = engine
    times = []

    for i in range(runs):
        start = time.time()
        run_OptiTrack(input_video, output_video)
        end = time.time()
        times.append(end - start)
        print(f"Run {i+1}/{runs}: {end - start:.3f}s")

    avg_time = sum(times) / runs

    print(f"\nAverage time using {engine.name}: {avg_time:.3f}s\n")

    return avg_time

Settings.defaultMaxFrameCount = 50

profile_engine(RaftEnginePreference.TensorRT, input_video, output_video)
profile_engine(RaftEnginePreference.TORCH, input_video, output_video)
