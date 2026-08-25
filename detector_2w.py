import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from scipy import ndimage
from ultralytics import YOLO

from train_2w_lcnet import DeadBlockLCNet4L


def grid_to_boxes(prob_map, conf_thresh=0.5, orig_h=1080, orig_w=1920):
    """
    Converts a spatial grid probability map (grid_h x grid_w) to pixel bounding boxes
    using connected component labeling (scipy.ndimage.label).
    Returns list of [x1, y1, x2, y2, confidence].
    """
    grid_h, grid_w = prob_map.shape
    binary = (prob_map >= conf_thresh).astype(np.uint8)
    labelled, n_obj = ndimage.label(binary)

    boxes = []
    for i in range(1, n_obj + 1):
        ys, xs = np.where(labelled == i)
        if len(ys) == 0 or len(xs) == 0:
            continue
        conf = float(prob_map[labelled == i].mean())
        
        # Grid cell coordinates to normalized coordinates [0, 1]
        y1_norm = ys.min() / grid_h
        y2_norm = (ys.max() + 1) / grid_h
        x1_norm = xs.min() / grid_w
        x2_norm = (xs.max() + 1) / grid_w
        
        # Scale to target pixel dimensions
        x1 = x1_norm * orig_w
        y1 = y1_norm * orig_h
        x2 = x2_norm * orig_w
        y2 = y2_norm * orig_h

        boxes.append([float(x1), float(y1), float(x2), float(y2), conf])
    return boxes


def filter_boxes(boxes, frame_shape, min_dim=15, max_dim=600):
    """
    Filters out noise boxes by dimension constraints and border bezel overlaps.
    """
    H, W = frame_shape[:2]
    filtered = []

    for box in boxes:
        x1, y1, x2, y2, conf = box
        w = x2 - x1
        h = y2 - y1

        # Size filter
        if w < min_dim or h < min_dim or w > max_dim or h > max_dim:
            continue

        # Border filter: check if box is heavily touching border margins
        if x1 < 5 and x2 < 30:
            continue
        if y1 < 5 and y2 < 30:
            continue
        if x2 > W - 5 and x1 > W - 30:
            continue
        if y2 > H - 5 and y1 > H - 30:
            continue

        filtered.append(box)
    return filtered


class LCNet2WDetector:
    """
    Two-Wheeler Detector wrapping Model A (DeadBlockLCNet4L).
    Specialized for patch-level feature extraction on 1080p 2-wheeler targets.
    """
    def __init__(self, model_weights="lcnet_2w_best.pt", conf_thresh=0.35, img_h=1080, img_w=1920):
        self.conf_thresh = conf_thresh
        self.img_h = img_h
        self.img_w = img_w
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Initializing LCNet Model A Detector on {self.device}...")
        self.model = DeadBlockLCNet4L(in_channels=1, img_h=img_h, img_w=img_w, c1=56, c2=42, c3=28, c4=7)

        weights_path = Path(model_weights)
        if weights_path.exists():
            print(f"Loading Model A weights from '{weights_path.resolve()}'...")
            sd = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(sd)
        else:
            print(f"Warning: Weights file '{model_weights}' not found. Detector running with random initialization.")

        self.model.to(self.device)
        self.model.eval()
        print("LCNet Model A Detector ready.")

    def detect(self, frame, imgsz=None):
        """
        Input: BGR frame numpy array (H, W, 3)
        Returns: list of detections [[x1, y1, x2, y2, confidence, class_id=3], ...]
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocessing: BGR -> Grayscale -> Resize to (1920, 1080)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (orig_w, orig_h) != (self.img_w, self.img_h):
            resized = cv2.resize(gray, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = gray

        # Convert to Tensor [1, 1, 1080, 1920] normalized to [0, 1]
        img_arr = np.array(resized, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_t)
            prob_map = torch.sigmoid(logits)[0].cpu().numpy()  # (154, 274)

        # Postprocessing: spatial connected component bounding box extraction
        raw_boxes = grid_to_boxes(prob_map, conf_thresh=self.conf_thresh, orig_h=self.img_h, orig_w=self.img_w)
        
        # Scale bounding boxes back to original input frame dimensions if resized
        scaled_boxes = []
        scale_x = orig_w / float(self.img_w)
        scale_y = orig_h / float(self.img_h)

        for (x1, y1, x2, y2, conf) in raw_boxes:
            sx1 = x1 * scale_x
            sy1 = y1 * scale_y
            sx2 = x2 * scale_x
            sy2 = y2 * scale_y
            scaled_boxes.append([sx1, sy1, sx2, sy2, conf])

        # Filter noise boxes
        filtered = filter_boxes(scaled_boxes, (orig_h, orig_w))

        # Format output: [x1, y1, x2, y2, conf, class_id=3] (class_id=3 for motorcycle/2-wheeler)
        detections = []
        for box in filtered:
            x1, y1, x2, y2, conf = box
            detections.append([x1, y1, x2, y2, conf, 3])

        return detections


class SmallVehicleDetector:
    """
    Baseline Detector specialized for small two-wheelers using YOLOv8.
    """
    def __init__(self, model_weights="models/yolov8n.pt", conf_thresh=0.15, iou_thresh=0.45, target_classes=[1, 3]):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target_classes = target_classes

        print(f"Loading YOLO detector weights from '{model_weights}'...")
        self.model = YOLO(model_weights)
        print(f"YOLO Detector initialized. Target classes: {self.target_classes}")

    def detect(self, frame, imgsz=1280):
        results = self.model.predict(
            source=frame,
            imgsz=imgsz,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.target_classes,
            verbose=False
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1

                if w > 0 and h > 0 and w < 600 and h < 600:
                    detections.append([float(x1), float(y1), float(x2), float(y2), conf, cls_id])

        return detections


class TiledYOLOP2Detector:
    """
    YOLO High-Zoom Tiling Architecture Detector for Ultra-Small 2-Wheelers:
    - 1 Global Pass (imgsz=1280)
    - High-Zoom Far Tiles (256x256 cropped from y=20..360, fed to YOLO at imgsz=1024 -> 4x Zoom Boost!)
    - Multi-class recall (0: Person/Rider, 1: Bicycle, 3: Motorcycle)
    - Maps Class 0 (Person/Rider) in far zone (h/w >= 1.0) to Class 3 (Motorcycle)
    - Ultra-low confidence threshold (conf=0.015)
    """
    def __init__(self, model_weights="models/yolov8m.pt", conf_thresh=0.05, iou_thresh=0.45,
                 roi_ymin=0.03, target_classes=[0, 1, 3]):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.roi_ymin = roi_ymin
        self.target_classes = target_classes

        print(f"Initializing YOLO High-Zoom Tiling Detector...")
        print(f"  - Model Weights: {model_weights}")
        print(f"  - Horizon Cutoff: y2 >= {roi_ymin:.2f} * H")
        print(f"  - Target Classes: {target_classes} (Person/Rider, Bicycle, Motorcycle)")

        self.model = YOLO(model_weights)
        print("YOLO High-Zoom Tiling Detector ready.")

    def _generate_adaptive_tiles(self, H, W):
        tiles = []
        roi_top = max(0, int(self.roi_ymin * H))  # ~20 px for 720p
        far_size = 256
        step_x = 140

        # Dense Far Horizon Grid (y from 20 to 280, x stride 140)
        for y_start in [roi_top, roi_top + 100]:
            y_crop = max(0, min(H - far_size, y_start))
            for x in range(0, max(1, W - far_size + 1), step_x):
                x_crop = max(0, min(W - far_size, x))
                tiles.append((x_crop, y_crop, far_size, far_size))

        unique_tiles = list(dict.fromkeys(tiles))
        return unique_tiles

    def detect(self, frame, imgsz=None, **kwargs):
        H, W = frame.shape[:2]
        all_raw_boxes = []

        # 1. Global Full-Frame Pass (imgsz=1280)
        global_results = self.model.predict(
            source=frame,
            imgsz=1280,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.target_classes,
            verbose=False
        )

        if len(global_results) > 0 and global_results[0].boxes is not None:
            boxes = global_results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy
                
                # Map Class 0 (Person/Rider) to Class 3 (Motorcycle) ONLY in far horizon zone (y2 < 0.45*H)
                if cls_id == 0:
                    if y2 < 0.45 * H and 1.0 <= ((y2 - y1) / max(1.0, x2 - x1)) <= 3.8:
                        cls_id = 3
                    else:
                        continue

                if y2 >= self.roi_ymin * H:
                    all_raw_boxes.append([float(x1), float(y1), float(x2), float(y2), conf, cls_id])

        # 2. High-Zoom Far Tile Pass (256x256 crop -> imgsz=1024 -> 4x Zoom Boost!)
        tiles = self._generate_adaptive_tiles(H, W)
        for (x_off, y_off, tw, th) in tiles:
            crop = frame[y_off:y_off + th, x_off:x_off + tw]
            tile_results = self.model.predict(
                source=crop,
                imgsz=1024,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                classes=self.target_classes,
                verbose=False
            )

            if len(tile_results) > 0 and tile_results[0].boxes is not None:
                boxes = tile_results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    gx1 = float(xyxy[0] + x_off)
                    gy1 = float(xyxy[1] + y_off)
                    gx2 = float(xyxy[2] + x_off)
                    gy2 = float(xyxy[3] + y_off)

                    gw = gx2 - gx1
                    gh = gy2 - gy1

                    # Map Class 0 (Person/Rider) to Class 3 (Motorcycle) ONLY in far horizon (gy2 < 0.45*H)
                    if cls_id == 0:
                        aspect_ratio = gh / max(1.0, gw)
                        if gy2 < 0.45 * H and 1.0 <= aspect_ratio <= 3.8:
                            cls_id = 3
                        else:
                            continue

                    # Filter static non-road margin artifacts in upper horizon
                    if gy2 < 0.40 * H and (gx1 < 30 or gx2 > W - 30):
                        continue

                    # Boost confidence of high-zoom far tile 2-wheelers by 1.4x into STRONG tier
                    boosted_conf = min(0.99, conf * 1.40) if (gh / max(1.0, gw) >= 1.0) else conf

                    if gy2 >= self.roi_ymin * H and gw >= 3 and gh >= 4 and gw < 600 and gh < 600:
                        all_raw_boxes.append([gx1, gy1, gx2, gy2, boosted_conf, cls_id])

        if not all_raw_boxes:
            return []

        # 3. Global class-aware NMS across tile boundaries
        final_detections = []
        unique_classes = set(int(b[5]) for b in all_raw_boxes)
        for c_id in unique_classes:
            cls_indices = [i for i, b in enumerate(all_raw_boxes) if int(b[5]) == c_id]
            if not cls_indices:
                continue

            cv_boxes = []
            scores = []
            valid_indices = []
            for idx in cls_indices:
                x1, y1, x2, y2, conf, _ = all_raw_boxes[idx]
                w = x2 - x1
                h = y2 - y1
                if w < 2 or h < 3 or w > 600 or h > 600:
                    continue
                cv_boxes.append([int(x1), int(y1), int(w), int(h)])
                scores.append(float(conf))
                valid_indices.append(idx)

            if not valid_indices:
                continue

            keep = cv2.dnn.NMSBoxes(
                cv_boxes, scores, float(self.conf_thresh), float(self.iou_thresh)
            )
            if len(keep):
                for k in np.asarray(keep).flatten():
                    final_detections.append(all_raw_boxes[valid_indices[int(k)]])

        final_detections.sort(key=lambda b: float(b[4]), reverse=True)
        return final_detections


def get_detector(detector_type="lcnet", weights=None, conf_thresh=0.05):
    """
    Factory function to select detector.
    """
    dtype = detector_type.lower()
    if dtype in ["lcnet", "model_a", "modela"]:
        w = weights if weights else "lcnet_2w_best.pt"
        return LCNet2WDetector(model_weights=w, conf_thresh=conf_thresh)
    elif dtype in ["p2_tiling", "tiled_yolo", "p2", "tiling", "yolo_p2"]:
        w = weights if weights else "models/yolov8m.pt"
        return TiledYOLOP2Detector(model_weights=w, conf_thresh=conf_thresh, iou_thresh=0.45, roi_ymin=0.03)
    else:
        w = weights if weights else "models/yolov8m.pt"
        return SmallVehicleDetector(model_weights=w, conf_thresh=conf_thresh)