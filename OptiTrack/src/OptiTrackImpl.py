import torch
from torch import NoneType, Tensor
import numpy as np

from .ExternalDependencies import *

VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck

from .VideoUtils import preprocess_frame
from .raftImpl.RaftTorch import InitializeRaft
from .raftImpl.RaftTRT import InitializeRaftTRT

from ..Settings import *

class OpticalFlow:
    def __init__(self):
        self.cachedModel = None

    def Compute(self, img1, img2, raftEngineType:RaftEnginePreference = None):

        if raftEngineType is None:
            raftEngineType = Settings.defaultRaftEnginePreference

        # Preprocess frames to reduce memory usage
        img1 = preprocess_frame(img1, Settings.defaultMaxFrameSize)
        img2 = preprocess_frame(img2, Settings.defaultMaxFrameSize)
    
        frameHW: tuple = img1.shape[:2]
    
        # Convert to tensor and move to device
        img1_t: Tensor = torch.from_numpy(img1).permute(2,0,1).float().unsqueeze(0).cuda()#to(device)
        img2_t: Tensor = torch.from_numpy(img2).permute(2,0,1).float().unsqueeze(0).cuda()#to(device)
    
        # Pad to multiple of 8 (RAFT requirement)
        padder = InputPadder(img1_t.shape)
        img1_t, img2_t = padder.pad(img1_t, img2_t)
    
        # Run RAFT
        match raftEngineType:
            case RaftEnginePreference.TORCH:
                if self.cachedModel is None:
                    self.cachedModel = InitializeRaft()
                raft_model = self.cachedModel
                with torch.no_grad():
                    flow_low, flow_up = raft_model(img1_t, img2_t, iters=20, test_mode=True)
            case RaftEnginePreference.TensorRT:
                if self.cachedModel is None:
                    self.cachedModel = InitializeRaftTRT(frameHW)
                trt_model = self.cachedModel
                flow_up = trt_model.infer(img1_t, img2_t)
    
        # Back to CPU numpy array and return
        return flow_up[0].permute(1,2,0).cpu().numpy()

    def release(self)->None:
        del self.cachedModel

class Tracker:
    def __init__(self, max_lost=Settings.defaultMaxLostFrame):
        self.tracks = {}
        self.next_id = 0
        self.count = 0
        self.max_lost = max_lost

    def _apply_flow(self, pos, flow):
        cx, cy = pos
        h, w, _ = flow.shape

        x = int(np.clip(cx, 0, w - 1))
        y = int(np.clip(cy, 0, h - 1))

        dx, dy = flow[y, x]
        return (int(cx + dx), int(cy + dy))

    def update(self, detections, flows):
        new_tracks = {}
        unmatched_tracks = set(self.tracks.keys())
        useFlow = flows is not None

        if useFlow:
            # Predict old track positions using flow
            predicted_tracks = {}
            for tid, track in self.tracks.items():
                pred_pos = self._apply_flow(track['pos'], flows)
                predicted_tracks[tid] = pred_pos

        for det in detections:
            x1, y1, x2, y2 = det
            cx, cy = (x1+x2)//2, (y1+y2)//2

            assigned = False

            # Try to match with existing tracks
            if useFlow:
                best_tid = None
                best_dist = 1e9
                for tid, (px, py) in predicted_tracks.items():
                    dist = np.hypot(cx - px, cy - py)
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                if best_tid is not None and best_dist < 60:
                    track = self.tracks[best_tid]
                    new_tracks[best_tid] = {
                        'pos': (cx, cy),
                        'bbox': det,
                        'counted': track.get('counted', False),
                        'age': track.get('age', 0) + 1,
                        'lost': 0
                    }
                    unmatched_tracks.discard(best_tid)
                    assigned = True
            else:
                # Simple nearest-neighbor assignment using previous tracks
                for tid, track in self.tracks.items():
                    prev_cx, prev_cy = track['pos']
                    dist = np.hypot(cx - prev_cx, cy - prev_cy)
                    if dist < 50:  # threshold
                        new_tracks[tid] = {
                            'pos': (cx, cy), 
                            'bbox': det, 
                            'counted': track.get('counted', False),
                            'age': track.get('age', 0) + 1,
                            'lost': 0
                            }
                        unmatched_tracks.discard(tid)
                        assigned = True
                        break

            # If no match, create NEW track
            if not assigned:
                new_tracks[self.next_id] = {
                    'pos': (cx, cy), 
                    'bbox': det, 
                    'counted': False,
                    'age': 100,
                    'lost': 0
                    }
                self.next_id += 1

        # Handle Lost Tracks
        # Add back tracks that weren't matched, provided they haven't been lost too long
        for tid in unmatched_tracks:
            track = self.tracks[tid]
            track['lost'] += 1
            if track['lost'] <= self.max_lost:
                new_tracks[tid] = track

        # Update tracks for next frame
        self.tracks = new_tracks

        # Only count if it's a stable track (age > 3) to avoid noise
        for tid, track in self.tracks.items():
            if not track['counted'] and track['age'] > 3: 
                self.count += 1
                track['counted'] = True

        # Counting logic: line crossing
        #for tid, track in self.tracks.items():
        #    _, cy = track['pos']
        #    if 200 < cy < 220 and not track.get('counted', False):  # crossing y=210
        #        self.count += 1
        #        track['counted'] = True

        return self.tracks
