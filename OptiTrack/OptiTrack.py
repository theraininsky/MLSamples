from .src.OptiTrackImpl import *
from .Settings import * 
from colorama import init, Fore, Style
from ultralytics import YOLO
import cv2

init(autoreset=True)

# Initialize YOLO (vehicle detection)
# Use yolov8n.pt or any suitable YOLOv8 weights
model = YOLO('yolov8n.pt')  

def run_OptiTrack(input_video, output_video):
    cap    = cv2.VideoCapture(input_video)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out    = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    ret, prev_frame = cap.read()
    prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)

    tracker = Tracker()
    opticalFlow = OpticalFlow()

    for i in range(Settings.defaultMaxFrameCount):
        print(f"{Fore.YELLOW}\rCurrent Frame: {i}\r", end="", flush=True)

        ret, frame = cap.read()
        if not ret:
            break
    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        # Detect vehicles
        results = model.predict(frame, verbose=False)[0]
        detections = []
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2])
    
        flow = None

        if Settings.useOpticalFlow:
        # Compute optical flow
            flow = opticalFlow.Compute(prev_frame_rgb, frame_rgb)
    
        # Update tracker
        tracks = tracker.update(detections, flow)
    
        # Draw results
        for tid, t in tracks.items():
            if t['lost'] == 0:
                x1, y1, x2, y2 = t['bbox']
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f'ID:{tid}', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
    
        # Draw counting line
        #cv2.line(frame, (0,210), (width,210), (0,0,255), 2)
        cv2.putText(frame, f'Count: {tracker.count}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)
    
        out.write(frame)
        prev_frame_rgb = frame_rgb.copy()

    print(f"\n", end="", flush=True)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    opticalFlow.release()
    
    print(f"Total vehicles counted: {tracker.count}")
