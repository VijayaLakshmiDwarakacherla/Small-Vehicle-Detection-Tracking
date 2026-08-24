import os
import csv
import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment


def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


@dataclass
class AMTConfig:
    """
    Centralized configuration dataclass for AMT Tracker v4.2 — Identity Persistence & Restoration.
    """
    roi_ymin: float = 0.18
    birth_conf: float = 0.15          # Default birth confidence fallback
    birth_conf_far: float = 0.15      # y < 0.40H
    birth_conf_medium: float = 0.20   # 0.40H <= y < 0.65H
    birth_conf_near: float = 0.25     # y >= 0.65H

    # Confidence Tiers
    conf_high: float = 0.35
    conf_medium: float = 0.15
    recovery_conf: float = 0.05       # 0.05 <= conf < birth_conf (recovery ONLY)

    min_hits: int = 2
    min_birth_hits: int = 2           # Require 2 hits within 3 frames (Section H)
    birth_window: int = 3

    max_misses_confirmed: int = 5     # Retain track for up to 5 missed frames (Section C)
    max_misses_lost: int = 5          # 5 missed frames recovery horizon
    max_recovery_misses: int = 5
    max_age: int = 30

    base_x: float = 25.0
    base_y: float = 30.0
    kx: float = 1.0
    ky: float = 1.5
    scale_ratio: float = 2.0

    id_hysteresis: float = 0.12       # Identity lock margin for mature tracks
    id_switch_margin: float = 0.10
    ambiguity_margin: float = 0.05
    direction_min_speed: float = 5.0  # Speed threshold for hard direction gating
    direction_stability_thresh: float = 0.75
    stationary_threshold: float = 2.5
    stationary_max_frames: int = 6
    match_cost_threshold: float = 0.65
    lambda_aspect: float = 0.5

    def __post_init__(self):
        if self.min_hits != 2 and self.min_birth_hits == 2:
            self.min_birth_hits = self.min_hits


# =====================================================================
# Reference Baselines (B0, B1, B2) - Implemented 100% From Scratch
# =====================================================================

class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1

    def update(self, detections, **kwargs):
        for t in self.tracks:
            t['age'] += 1
            t['time_since_update'] += 1

        if not detections:
            self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
            return []

        if not self.tracks:
            for d in detections:
                if d[4] >= 0.15:
                    self.tracks.append({
                        'id': self.next_id, 'bbox': d[:4], 'conf': d[4],
                        'age': 1, 'time_since_update': 0, 'hits': 1
                    })
                    self.next_id += 1
            return []

        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                cost_matrix[i, j] = 1.0 - compute_iou(t['bbox'], d[:4])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_trks, unmatched_dets = [], list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= (1.0 - self.iou_threshold):
                matched_trks.append((r, c))
                unmatched_dets.remove(c)

        for r, c in matched_trks:
            self.tracks[r]['bbox'] = detections[c][:4]
            self.tracks[r]['conf'] = detections[c][4]
            self.tracks[r]['time_since_update'] = 0
            self.tracks[r]['hits'] += 1

        for c in unmatched_dets:
            if detections[c][4] >= 0.15:
                self.tracks.append({
                    'id': self.next_id, 'bbox': detections[c][:4], 'conf': detections[c][4],
                    'age': 1, 'time_since_update': 0, 'hits': 1
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        return [[t['bbox'][0], t['bbox'][1], t['bbox'][2], t['bbox'][3], t['id'], 0.0, t['conf'], [], False, "OBS"]
                for t in self.tracks if t['time_since_update'] == 0]


class NNTracker:
    def __init__(self, dist_threshold=50.0, max_age=5):
        self.dist_threshold = dist_threshold
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1

    def update(self, detections, **kwargs):
        for t in self.tracks:
            t['age'] += 1
            t['time_since_update'] += 1

        if not detections:
            self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
            return []

        if not self.tracks:
            for d in detections:
                if d[4] >= 0.15:
                    cx = (d[0] + d[2]) / 2.0
                    cy = (d[1] + d[3]) / 2.0
                    self.tracks.append({
                        'id': self.next_id, 'bbox': d[:4], 'center': (cx, cy),
                        'conf': d[4], 'age': 1, 'time_since_update': 0, 'hits': 1
                    })
                    self.next_id += 1
            return []

        cost_matrix = np.full((len(self.tracks), len(detections)), 1e5, dtype=np.float32)
        for i, t in enumerate(self.tracks):
            tcx, tcy = t['center']
            for j, d in enumerate(detections):
                dcx = (d[0] + d[2]) / 2.0
                dcy = (d[1] + d[3]) / 2.0
                dist = math.sqrt((tcx - dcx)**2 + (tcy - dcy)**2)
                cost_matrix[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_trks, unmatched_dets = [], list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.dist_threshold:
                matched_trks.append((r, c))
                unmatched_dets.remove(c)

        for r, c in matched_trks:
            d = detections[c]
            dcx = (d[0] + d[2]) / 2.0
            dcy = (d[1] + d[3]) / 2.0
            self.tracks[r]['bbox'] = d[:4]
            self.tracks[r]['center'] = (dcx, dcy)
            self.tracks[r]['conf'] = d[4]
            self.tracks[r]['time_since_update'] = 0
            self.tracks[r]['hits'] += 1

        for c in unmatched_dets:
            if detections[c][4] >= 0.15:
                d = detections[c]
                dcx = (d[0] + d[2]) / 2.0
                dcy = (d[1] + d[3]) / 2.0
                self.tracks.append({
                    'id': self.next_id, 'bbox': d[:4], 'center': (dcx, dcy),
                    'conf': d[4], 'age': 1, 'time_since_update': 0, 'hits': 1
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        return [[t['bbox'][0], t['bbox'][1], t['bbox'][2], t['bbox'][3], t['id'], 0.0, t['conf'], [], False, "OBS"]
                for t in self.tracks if t['time_since_update'] == 0]


class SORTTracker:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections, **kwargs):
        for t in self.tracks:
            t['cx'] += t['vx']
            t['cy'] += t['vy']
            t['age'] += 1
            t['time_since_update'] += 1

        if not detections:
            self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
            return []

        if not self.tracks:
            for d in detections:
                if d[4] >= 0.15:
                    x1, y1, x2, y2 = d[:4]
                    w, h = x2 - x1, y2 - y1
                    self.tracks.append({
                        'id': self.next_id, 'cx': (x1+x2)/2.0, 'cy': (y1+y2)/2.0,
                        'w': w, 'h': h, 'vx': 0.0, 'vy': 0.0, 'conf': d[4],
                        'age': 1, 'time_since_update': 0, 'hits': 1, 'history': [((x1+x2)/2.0, (y1+y2)/2.0)]
                    })
                    self.next_id += 1
            return []

        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(self.tracks):
            tb = [t['cx']-t['w']/2, t['cy']-t['h']/2, t['cx']+t['w']/2, t['cy']+t['h']/2]
            for j, d in enumerate(detections):
                cost_matrix[i, j] = 1.0 - compute_iou(tb, d[:4])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_trks, unmatched_dets = [], list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= (1.0 - self.iou_threshold):
                matched_trks.append((r, c))
                unmatched_dets.remove(c)

        for r, c in matched_trks:
            d = detections[c]
            dcx = (d[0] + d[2]) / 2.0
            dcy = (d[1] + d[3]) / 2.0
            dw = max(1.0, d[2] - d[0])
            dh = max(1.0, d[3] - d[1])

            dx = dcx - self.tracks[r]['cx']
            dy = dcy - self.tracks[r]['cy']

            self.tracks[r]['vx'] = 0.5 * self.tracks[r]['vx'] + 0.5 * dx
            self.tracks[r]['vy'] = 0.5 * self.tracks[r]['vy'] + 0.5 * dy
            self.tracks[r]['cx'] = dcx
            self.tracks[r]['cy'] = dcy
            self.tracks[r]['w'] = dw
            self.tracks[r]['h'] = dh
            self.tracks[r]['conf'] = d[4]
            self.tracks[r]['time_since_update'] = 0
            self.tracks[r]['hits'] += 1
            self.tracks[r]['history'].append((dcx, dcy))

        for c in unmatched_dets:
            if detections[c][4] >= 0.15:
                d = detections[c]
                x1, y1, x2, y2 = d[:4]
                w, h = x2 - x1, y2 - y1
                self.tracks.append({
                    'id': self.next_id, 'cx': (x1+x2)/2.0, 'cy': (y1+y2)/2.0,
                    'w': w, 'h': h, 'vx': 0.0, 'vy': 0.0, 'conf': d[4],
                    'age': 1, 'time_since_update': 0, 'hits': 1, 'history': [((x1+x2)/2.0, (y1+y2)/2.0)]
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        return [[t['cx']-t['w']/2, t['cy']-t['h']/2, t['cx']+t['w']/2, t['cy']+t['h']/2,
                 t['id'], math.sqrt(t['vx']**2 + t['vy']**2), t['conf'], t['history'], False, "OBS"]
                for t in self.tracks if t['time_since_update'] == 0 and t['hits'] >= self.min_hits]


# =====================================================================
# Production AMT — Adaptive Motion and Geometry Tracker (v4.2)
# =====================================================================

class AMTTrack:
    """
    AMT Track Container (v4.2 Identity Persistence & Restoration):
    - Lifecycle: Candidate -> Confirmed -> Lost -> Confirmed / Deleted.
    - Track Identity Confidence (Section G) in [0.0, 1.0].
    - Explicit previous_state & association_event tracking.
    """
    _count = 0

    def __init__(self, bbox, confidence, class_id=0, min_birth_hits=2, custom_id=None):
        if custom_id is not None:
            self.track_id = custom_id
        else:
            AMTTrack._count += 1
            self.track_id = AMTTrack._count

        x1, y1, x2, y2 = bbox
        self.w = max(1.0, float(x2 - x1))
        self.h = max(1.0, float(y2 - y1))
        self.cx = float((x1 + x2) / 2.0)
        self.cy = float((y1 + y2) / 2.0)

        self.anchor_center = (self.cx, self.cy)
        self.anchor_size = (self.w, self.h)
        self.anchor_velocity = (0.0, 0.0)
        self.anchor_direction = (0.0, 0.0)
        self.anchor_frame = 0

        self.identity_locked = False
        self.track_identity_confidence = 0.50  # Section G: track identity confidence [0, 1]

        self.dw = 0.0
        self.dh = 0.0
        self.pred_w = self.w
        self.pred_h = self.h

        self.last_observation = (self.cx, self.cy)
        self.predicted_position = (self.cx, self.cy)

        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0

        self.trajectory = deque([(self.cx, self.cy)], maxlen=6)

        self.sigma_pos = 5.0
        self.sigma_vel = 2.0
        self.sigma_max = 50.0

        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.min_birth_hits = min_birth_hits

        self.hits = 1
        self.age = 1
        self.missed_frames = 0
        self.time_since_update = 0
        self.hit_history = deque([1], maxlen=3)

        self.state = "Candidate"
        self.previous_state = "NONE"
        self.association_event = "NONE"
        self.matched_det_idx = -1
        self.last_matched_cost = 0.0

        self.stationary_frames = 0
        self.avg_confidence = float(confidence)
        self.continuity_score = 0.50

    def get_continuity_status(self):
        if self.continuity_score >= 0.75:
            return "HIGH"
        elif self.continuity_score >= 0.45:
            return "MEDIUM"
        else:
            return "LOW"

    def adjust_continuity(self, delta):
        self.continuity_score = max(0.0, min(1.0, self.continuity_score + delta))

    def get_track_quality(self):
        q_det = self.avg_confidence
        q_motion = 1.0 / (1.0 + 0.05 * self.sigma_pos)
        q_age = min(1.0, self.hits / 5.0)
        q_miss = 0.20 * self.missed_frames
        return max(0.0, min(1.0, 0.40 * q_det + 0.35 * q_motion + 0.25 * q_age - q_miss))

    def get_direction_stability(self):
        if len(self.trajectory) < 4:
            return 0.0
        pts = list(self.trajectory)
        vectors = []
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i-1][0]
            dy = pts[i][1] - pts[i-1][1]
            mag = math.sqrt(dx*dx + dy*dy)
            if mag >= 0.5:
                vectors.append((dx/mag, dy/mag))
        if len(vectors) < 2:
            return 0.0
        dot_products = [vectors[i][0]*vectors[i+1][0] + vectors[i][1]*vectors[i+1][1] for i in range(len(vectors)-1)]
        return float(np.mean(dot_products))

    def predict(self, dt=1.0):
        ax_clip = max(-5.0, min(5.0, self.ax))
        ay_clip = max(-5.0, min(5.0, self.ay))

        if self.missed_frames > 0:
            damp = 0.5 ** self.missed_frames
            self.cx += self.vx * dt * damp
            self.cy += self.vy * dt * damp
        else:
            self.cx += self.vx * dt + 0.5 * ax_clip * (dt ** 2)
            self.cy += self.vy * dt + 0.5 * ay_clip * (dt ** 2)

        # Velocity is NOT re-estimated during misses to prevent fake confidence drift
        if self.missed_frames == 0:
            self.vx += ax_clip * dt
            self.vy += ay_clip * dt

        self.pred_w = max(4.0, self.w + self.dw * dt)
        self.pred_h = max(6.0, self.h + self.dh * dt)

        self.predicted_position = (self.cx, self.cy)
        self.sigma_pos = min(self.sigma_max, self.sigma_pos + self.sigma_vel * dt + 3.0)
        self.age += 1
        self.hit_history.append(0)

        # Track quality and identity confidence decrease smoothly during prediction (Section G)
        self.avg_confidence *= 0.97
        self.track_identity_confidence = max(0.0, self.track_identity_confidence - 0.05)

        return self.get_bbox()

    def update(self, bbox, confidence, is_strong=True):
        x1, y1, x2, y2 = bbox
        det_w = max(1.0, float(x2 - x1))
        det_h = max(1.0, float(y2 - y1))
        det_cx = float((x1 + x2) / 2.0)
        det_cy = float((y1 + y2) / 2.0)

        # Position Error & Scale Error for Identity Confidence Update (Section G)
        dist_err = math.sqrt((det_cx - self.cx)**2 + (det_cy - self.cy)**2)
        scale_err = max(det_w / max(1.0, self.pred_w), self.pred_w / max(1.0, det_w))

        if is_strong and dist_err < 15.0 and scale_err < 1.3:
            self.track_identity_confidence = min(1.0, self.track_identity_confidence + 0.15)
        else:
            self.track_identity_confidence = min(1.0, self.track_identity_confidence + 0.05)

        alpha = 0.65
        smoothed_cx = alpha * det_cx + (1.0 - alpha) * self.cx
        smoothed_cy = alpha * det_cy + (1.0 - alpha) * self.cy

        elapsed = float(max(1, self.missed_frames + 1))
        last_obs_cx, last_obs_cy = self.last_observation

        obs_vx = (smoothed_cx - last_obs_cx) / elapsed
        obs_vy = (smoothed_cy - last_obs_cy) / elapsed

        new_vx = 0.60 * self.vx + 0.40 * obs_vx
        new_vy = 0.60 * self.vy + 0.40 * obs_vy

        new_ax = new_vx - self.vx
        new_ay = new_vy - self.vy

        self.vx = new_vx
        self.vy = new_vy
        self.ax = max(-5.0, min(5.0, 0.70 * self.ax + 0.30 * new_ax))
        self.ay = max(-5.0, min(5.0, 0.70 * self.ay + 0.30 * new_ay))

        self.dw = 0.70 * self.dw + 0.30 * ((det_w - self.w) / elapsed)
        self.dh = 0.70 * self.dh + 0.30 * ((det_h - self.h) / elapsed)

        self.cx = smoothed_cx
        self.cy = smoothed_cy
        self.last_observation = (smoothed_cx, smoothed_cy)
        self.predicted_position = (smoothed_cx, smoothed_cy)

        self.w = 0.70 * self.w + 0.30 * det_w
        self.h = 0.70 * self.h + 0.30 * det_h
        self.pred_w = self.w
        self.pred_h = self.h

        if confidence >= 0.35:
            self.anchor_center = (smoothed_cx, smoothed_cy)
            self.anchor_size = (self.w, self.h)
            self.anchor_velocity = (self.vx, self.vy)
            self.anchor_frame = self.age

        self.sigma_pos = max(2.0, 0.50 * self.sigma_pos)
        self.confidence = float(confidence)
        self.avg_confidence = 0.70 * self.avg_confidence + 0.30 * float(confidence)
        self.hits += 1

        was_lost = (self.previous_state == "Lost")
        self.missed_frames = 0
        self.time_since_update = 0

        if self.hit_history:
            self.hit_history[-1] = 1

        if was_lost:
            self.adjust_continuity(-0.04)
        elif is_strong:
            self.adjust_continuity(0.12)
        else:
            self.adjust_continuity(0.04)

        # Motion-Gated Confirmation: Moving bikes (net_disp >= 2.5px over 2 hits) confirm instantly.
        # Static background noise (net_disp < 1.5px after 3 hits) is rejected as false positive.
        net_disp = math.sqrt((self.cx - self.anchor_center[0])**2 + (self.cy - self.anchor_center[1])**2)
        if self.previous_state in ["Candidate", "Lost"]:
            if sum(self.hit_history) >= self.min_birth_hits:
                if net_disp >= 2.0 or self.hits >= 3 or self.confidence >= 0.25:
                    self.state = "Confirmed"

        if self.age >= 10 and self.hits >= 5:
            self.identity_locked = True

        self.trajectory.append((self.cx, self.cy))

        speed = self.get_speed_px()
        if speed < 1.5:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0

    def get_speed_px(self):
        return float(math.sqrt(self.vx**2 + self.vy**2))

    def get_bbox(self):
        x1 = self.cx - self.w / 2.0
        y1 = self.cy - self.h / 2.0
        x2 = self.cx + self.w / 2.0
        y2 = self.cy + self.h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)


class AMTTracker:
    """
    AMT — Adaptive Motion and Geometry Tracker (v4.2 Identity Persistence & Restoration).
    Fulfills all Sections A through M:
    - Real Identity Fragmentation Metrics (Section A)
    - Replacement Detection & Restoration (Section I & L)
    - Expanded Recovery Gate & Multi-Feature Cost (Section E & F)
    - Candidate Birth & Confirmation (Section H)
    - Expanded CSV Export (Section K)
    """

    def __init__(self,
                 config: AMTConfig = None,
                 max_age=30,
                 min_hits=2,
                 match_cost_threshold=0.65,
                 roi_ymin=0.18,
                 img_h=720,
                 img_w=1280,
                 enable_perspective=True,
                 enable_direction=True,
                 enable_recovery=True,
                 csv_path="tracker_diagnostic.csv"):

        if config is not None:
            self.cfg = config
        else:
            self.cfg = AMTConfig(
                max_age=max_age,
                min_birth_hits=min_hits,
                match_cost_threshold=match_cost_threshold,
                roi_ymin=roi_ymin
            )

        self.img_h = img_h
        self.img_w = img_w
        self.enable_perspective = enable_perspective
        self.enable_direction = enable_direction
        self.enable_recovery = enable_recovery

        self.tracks = []
        self.all_historical_tracks = []
        self.recently_lost_or_deleted = []
        self.suspected_id_switches = []
        self.suspected_replacements = []     # Automated ID fragmentation log (Section L)
        self.frame_counter = 0

        # Cumulative Diagnostic Counters (Section A)
        self.total_raw_detections = 0
        self.total_fused_detections = 0
        self.total_candidates_created = 0
        self.total_candidates_confirmed = 0
        self.total_tracks_deleted = 0
        self.total_recoveries = 0
        self.stats_births = 0
        self.stats_recoveries = 0
        self.stats_duplicate_suppressions = 0
        self.total_id_switches = 0
        self.detections_rejected_by_filters = 0

        self.fragmentation_events = 0
        self.replacement_events = 0
        self.orphaned_tracks = 0

        # CSV Diagnostic Export (Section K)
        self.csv_path = csv_path
        if self.csv_path:
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame', 'track_id', 'event', 'det_x1', 'det_y1', 'det_x2', 'det_y2',
                    'confidence', 'source', 'zone', 'state', 'missed_frames',
                    'pred_cx', 'pred_cy', 'actual_cx', 'actual_cy', 'distance_error',
                    'scale_ratio', 'track_identity_confidence', 'matched_track_id', 'association_cost'
                ])

    def _detect_suspected_id_switches(self, detections, matched_trks):
        """
        Explicit ID_SWITCH_SUSPECTED detection engine.
        Logs every suspected identity jump, candidate spatial steal, or trajectory exchange.
        """
        # 1. Check spatial trajectory exchange between active confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.state == "Confirmed"]
        for i in range(len(confirmed_tracks)):
            t1 = confirmed_tracks[i]
            for j in range(i + 1, len(confirmed_tracks)):
                t2 = confirmed_tracks[j]
                dist = math.sqrt((t1.cx - t2.cx)**2 + (t1.cy - t2.cy)**2)
                if dist < 40.0:
                    v1_mag, v2_mag = t1.get_speed_px(), t2.get_speed_px()
                    if v1_mag >= 2.0 and v2_mag >= 2.0:
                        cos_sim = (t1.vx * t2.vx + t1.vy * t2.vy) / (v1_mag * v2_mag + 1e-5)
                        if cos_sim < 0.20:
                            self.suspected_id_switches.append({
                                'frame': self.frame_counter,
                                'old_id': t1.track_id,
                                'new_id': t2.track_id,
                                'distance': dist,
                                'iou': compute_iou(t1.get_bbox(), t2.get_bbox()),
                                'velocity_cosine': cos_sim,
                                'scale_ratio': max(t1.w / max(1.0, t2.w), t2.w / max(1.0, t1.w)),
                                'association_cost': t1.last_matched_cost,
                                'competing_track_id': t2.track_id,
                                'reason': 'Spatial Trajectory Exchange (<40px)'
                            })

        # 2. Check if newly created track is spawned at a location where a track was lost <= 3 frames ago
        new_tracks = [t for t in self.tracks if t.association_event == "NEW"]
        for new_t in new_tracks:
            for record in self.recently_lost_or_deleted:
                gap = self.frame_counter - record['frame']
                if 1 <= gap <= 3:
                    pred_cx = record['last_cx'] + record['last_vx'] * gap
                    pred_cy = record['last_cy'] + record['last_vy'] * gap
                    dist = math.sqrt((new_t.cx - pred_cx)**2 + (new_t.cy - pred_cy)**2)
                    if dist <= 60.0:
                        v_mag = math.sqrt(record['last_vx']**2 + record['last_vy']**2) + 1e-5
                        cos_sim = (record['last_vx'] * new_t.vx + record['last_vy'] * new_t.vy) / (v_mag * new_t.get_speed_px() + 1e-5)
                        scale_r = max(new_t.w / max(1.0, record['last_w']), record['last_w'] / max(1.0, new_t.w))
                        self.suspected_id_switches.append({
                            'frame': self.frame_counter,
                            'old_id': record['track_id'],
                            'new_id': new_t.track_id,
                            'distance': dist,
                            'iou': 0.0,
                            'velocity_cosine': cos_sim,
                            'scale_ratio': scale_r,
                            'association_cost': 0.0,
                            'competing_track_id': record['track_id'],
                            'reason': f'New ID spawned near lost track (Gap {gap}f, {dist:.1f}px)'
                        })

    def print_summary_diagnostics(self):
        """
        Prints the complete Diagnostic Summary Table & Suspected ID Switches.
        """
        all_trks = self.all_historical_tracks if self.all_historical_tracks else self.tracks
        confirmed_trks = [t for t in all_trks if t.hits >= 2 or t.state == "Confirmed"]

        confirmed_lifetimes = [t.age for t in confirmed_trks] if confirmed_trks else [0]
        avg_confirmed_lifetime = float(np.mean(confirmed_lifetimes))
        median_confirmed_lifetime = float(np.median(confirmed_lifetimes))

        gte_10_hits = sum(1 for t in confirmed_trks if t.hits >= 10)
        gte_30_hits = sum(1 for t in confirmed_trks if t.hits >= 30)
        gte_60_hits = sum(1 for t in confirmed_trks if t.hits >= 60)

        conf_cnt = sum(1 for t in self.tracks if t.state == "Confirmed")
        lost_cnt = sum(1 for t in self.tracks if t.state == "Lost")

        print("\n=================================================================")
        print("  AMT TRACKER ADVANCED DIAGNOSTICS SUMMARY")
        print("=================================================================")
        print(f"  Total Processed Frames:            {self.frame_counter}")
        print(f"  Raw Detections Accumulated:       {self.total_raw_detections}")
        print(f"  Fused Detections Accumulated:     {self.total_fused_detections}")
        print(f"  Detections Rejected by Filters:   {self.detections_rejected_by_filters}")
        print(f"  Total Candidates Created:         {self.total_candidates_created}")
        print(f"  Total Candidates Confirmed:       {self.total_candidates_confirmed}")
        print(f"  Current Active Confirmed Tracks:  {conf_cnt}")
        print(f"  Current Active Lost Tracks:       {lost_cnt}")
        print(f"  Total Recoveries (Lost -> Conf):  {self.total_recoveries}")
        print(f"  Total Tracks Deleted:             {self.total_tracks_deleted}")
        print(f"  Orphaned Tracks:                  {self.orphaned_tracks}")
        print(f"  Replacement Events Restored:      {self.replacement_events}")
        print(f"  Suspected ID Switches:            {len(self.suspected_id_switches)}")
        print(f"  Average Confirmed Track Lifetime: {avg_confirmed_lifetime:.1f} frames")
        print(f"  Median Confirmed Track Lifetime:  {median_confirmed_lifetime:.1f} frames")
        print(f"  Tracks with >= 10 Hits:           {gte_10_hits}")
        print(f"  Tracks with >= 30 Hits:           {gte_30_hits}")
        print(f"  Tracks with >= 60 Hits:           {gte_60_hits}")
        print(f"  Diagnostic CSV Export:            {self.csv_path}")
        print("=================================================================")

        print("\n=================================================================")
        print("  EXPLICIT SUSPECTED ID SWITCH ANALYSIS TABLE")
        print("=================================================================")
        print(f"  {'Frame':<6} | {'Old ID':<8} | {'New ID':<8} | {'Dist(px)':<9} | {'IoU':<6} | {'CosSim':<7} | {'ScaleR':<7} | {'Reason':<30}")
        print("  " + "-" * 95)
        if self.suspected_id_switches:
            for item in self.suspected_id_switches:
                f_s = f"{item['frame']}"
                old_s = f"#{item['old_id']}"
                new_s = f"#{item['new_id']}"
                dist_s = f"{item['distance']:.1f}"
                iou_s = f"{item['iou']:.2f}"
                cos_s = f"{item['velocity_cosine']:.2f}"
                scale_s = f"{item['scale_ratio']:.2f}"
                reason_s = item['reason']
                print(f"  {f_s:<6} | {old_s:<8} | {new_s:<8} | {dist_s:<9} | {iou_s:<6} | {cos_s:<7} | {scale_s:<7} | {reason_s:<30}")
        else:
            print("  No suspected ID switch events detected during sequence.")
        print("=================================================================\n")

    def _plausibly_belongs_to_existing_track(self, det_box, conf):
        det_cx = (det_box[0] + det_box[2]) / 2.0
        det_cy = (det_box[1] + det_box[3]) / 2.0
        det_w = max(1.0, det_box[2] - det_box[0])
        det_h = max(1.0, det_box[3] - det_box[1])

        for trk in self.tracks:
            if trk.state in ["Confirmed", "Lost"]:
                elliptical_val = self._compute_elliptical_val(trk, det_cx, det_cy, is_recovery=True)
                scale_r_w = max(det_w / trk.w, trk.w / det_w)
                scale_r_h = max(det_h / trk.h, trk.h / det_h)

                if elliptical_val <= 1.2 and scale_r_w <= 2.0 and scale_r_h <= 2.0:
                    return True
        return False

    def _suppress_duplicate_tracks(self):
        if len(self.tracks) < 2:
            return

        to_delete = set()
        for i in range(len(self.tracks)):
            if i in to_delete:
                continue
            t1 = self.tracks[i]
            for j in range(i + 1, len(self.tracks)):
                if j in to_delete:
                    continue
                t2 = self.tracks[j]

                dist = math.sqrt((t1.cx - t2.cx)**2 + (t1.cy - t2.cy)**2)
                scale_w = max(t1.w / t2.w, t2.w / t1.w)
                scale_h = max(t1.h / t2.h, t2.h / t1.h)

                if dist < 15.0 and scale_w < 1.3 and scale_h < 1.3:
                    v1_mag = t1.get_speed_px()
                    v2_mag = t2.get_speed_px()
                    if v1_mag >= 0.5 and v2_mag >= 0.5:
                        cos_sim = (t1.vx * t2.vx + t1.vy * t2.vy) / (v1_mag * v2_mag + 1e-5)
                        if cos_sim < 0.5:
                            continue

                    if t1.get_track_quality() >= t2.get_track_quality():
                        to_delete.add(j)
                    else:
                        to_delete.add(i)

        if to_delete:
            self.stats_duplicate_suppressions += len(to_delete)
            self.tracks = [t for idx, t in enumerate(self.tracks) if idx not in to_delete]

    def _compute_elliptical_val(self, trk, det_cx, det_cy, is_recovery=False):
        dx = abs(det_cx - trk.cx)
        dy = abs(det_cy - trk.cy)

        norm_y = trk.cy / float(self.img_h)
        if norm_y < 0.48:
            d_base = 95.0
        elif norm_y < 0.68:
            d_base = 110.0
        else:
            d_base = 140.0

        m = max(1, trk.missed_frames)
        gate_mult = 1.0 + 0.25 * (m - 1) if is_recovery else 1.0

        dx_limit = (d_base * 0.8) * gate_mult + self.cfg.kx * abs(trk.vx) * m + 0.5 * trk.sigma_pos
        dy_limit = (d_base * 1.0) * gate_mult + self.cfg.ky * abs(trk.vy) * m + 0.5 * trk.sigma_pos

        if norm_y < 0.48:
            dynamic_limit = min(95.0, 35.0 + 1.5 * trk.get_speed_px() + 12.0 * (m - 1))
            dx_limit = max(dx_limit, dynamic_limit)
            dy_limit = max(dy_limit, dynamic_limit)

        if is_recovery:
            dx_limit = min(110.0, dx_limit)
            dy_limit = min(110.0, dy_limit)

        val = (dx / max(1.0, dx_limit))**2 + (dy / max(1.0, dy_limit))**2
        return float(val)

    def _multi_priority_associate(self, det_boxes, det_confs, det_tiers):
        if not self.tracks or not det_boxes:
            return [], list(range(len(det_boxes))), list(range(len(self.tracks)))

        available_dets = set(range(len(det_boxes)))

        def is_mature(t):
            return t.age >= 10 and (t.hits >= 5 or t.track_identity_confidence >= 0.75)

        # Stage 1: Mature Confirmed vs STRONG & TRACKABLE detections
        p1_trks = [i for i, t in enumerate(self.tracks) if t.previous_state == "Confirmed" and is_mature(t)]
        p1_dets = [j for j in available_dets if det_tiers[j] in ["STRONG", "TRACKABLE"]]
        matched1, _, _ = self._associate_subset(p1_trks, p1_dets, det_boxes, det_confs, is_recovery_stage=False)
        for _, d_idx in matched1:
            available_dets.discard(d_idx)

        # Stage 2: Young Confirmed vs remaining STRONG & TRACKABLE detections
        p2_trks = [i for i, t in enumerate(self.tracks) if t.previous_state == "Confirmed" and not is_mature(t)]
        p2_dets = [j for j in available_dets if det_tiers[j] in ["STRONG", "TRACKABLE"]]
        matched2, _, _ = self._associate_subset(p2_trks, p2_dets, det_boxes, det_confs, is_recovery_stage=False)
        for _, d_idx in matched2:
            available_dets.discard(d_idx)

        # Stage 3: Lost Mature vs remaining detections (including WEAK)
        p3_trks = [i for i, t in enumerate(self.tracks) if t.previous_state == "Lost" and is_mature(t)]
        p3_dets = list(available_dets)
        matched3, _, _ = self._associate_subset(p3_trks, p3_dets, det_boxes, det_confs, is_recovery_stage=True)
        for _, d_idx in matched3:
            available_dets.discard(d_idx)

        # Stage 4: Lost Young vs remaining detections (including WEAK)
        p4_trks = [i for i, t in enumerate(self.tracks) if t.previous_state == "Lost" and not is_mature(t)]
        p4_dets = list(available_dets)
        matched4, _, _ = self._associate_subset(p4_trks, p4_dets, det_boxes, det_confs, is_recovery_stage=True)
        for _, d_idx in matched4:
            available_dets.discard(d_idx)

        # Stage 5: Candidate Tracks vs remaining STRONG & TRACKABLE detections
        p5_trks = [i for i, t in enumerate(self.tracks) if t.previous_state == "Candidate"]
        p5_dets = [j for j in available_dets if det_tiers[j] in ["STRONG", "TRACKABLE"]]
        matched5, _, _ = self._associate_subset(p5_trks, p5_dets, det_boxes, det_confs, is_recovery_stage=False)
        for _, d_idx in matched5:
            available_dets.discard(d_idx)

        all_matched = matched1 + matched2 + matched3 + matched4 + matched5
        all_unmatched_dets = list(available_dets)
        all_unmatched_trks = list(set(range(len(self.tracks))) - set([m[0] for m in all_matched]))

        return all_matched, all_unmatched_dets, all_unmatched_trks

    def _associate_subset(self, trk_indices, det_indices, det_boxes, det_confs, is_recovery_stage=False):
        if not trk_indices or not det_indices:
            return [], det_indices, trk_indices

        cost_matrix = np.full((len(trk_indices), len(det_indices)), 1e5, dtype=np.float32)

        for row_i, trk_idx in enumerate(trk_indices):
            trk = self.tracks[trk_idx]
            trk_box = trk.get_bbox()
            trk_speed = trk.get_speed_px()
            is_lost = (trk.previous_state == "Lost")
            trk_norm_y = trk.cy / float(self.img_h)

            for col_j, det_idx in enumerate(det_indices):
                det_box = det_boxes[det_idx]
                det_c = det_confs[det_idx]

                det_cx = (det_box[0] + det_box[2]) / 2.0
                det_cy = (det_box[1] + det_box[3]) / 2.0
                det_w = max(1.0, det_box[2] - det_box[0])
                det_h = max(1.0, det_box[3] - det_box[1])

                elliptical_val = self._compute_elliptical_val(trk, det_cx, det_cy, is_recovery=is_lost)
                if elliptical_val > 1.0:
                    cost_matrix[row_i, col_j] = 1e5
                    continue

                max_scale = 1.60 if (is_lost and trk.missed_frames == 1) else self.cfg.scale_ratio
                scale_r_w = max(det_w / trk.pred_w, trk.pred_w / det_w)
                scale_r_h = max(det_h / trk.pred_h, trk.pred_h / det_h)
                if scale_r_w > max_scale or scale_r_h > max_scale:
                    cost_matrix[row_i, col_j] = 1e5
                    continue

                c_m = math.sqrt(elliptical_val)

                obs_vx = (det_cx - trk.cx) / max(1.0, trk.missed_frames)
                obs_vy = (det_cy - trk.cy) / max(1.0, trk.missed_frames)
                v_diff = math.sqrt((obs_vx - trk.vx)**2 + (obs_vy - trk.vy)**2)

                obs_ax = obs_vx - trk.vx
                obs_ay = obs_vy - trk.vy
                a_diff = math.sqrt((obs_ax - trk.ax)**2 + (obs_ay - trk.ay)**2)

                v_gate = max(15.0, 10.0 + 0.5 * trk.sigma_vel)
                a_gate = max(10.0, 8.0 + 0.3 * trk.sigma_vel)
                c_v = (v_diff / v_gate) + 0.3 * (a_diff / a_gate)

                disp_mag = math.sqrt((det_cx - trk.cx)**2 + (det_cy - trk.cy)**2)
                if self.enable_direction and trk_speed >= self.cfg.direction_min_speed and disp_mag >= 0.5:
                    cos_sim = (trk.vx * (det_cx - trk.cx) + trk.vy * (det_cy - trk.cy)) / (trk_speed * disp_mag + 1e-5)
                    c_d = 0.3 * max(0.0, min(2.0, 1.0 - cos_sim))

                    if len(trk.trajectory) >= 5 and trk_speed >= 5.0 and trk.get_direction_stability() >= 0.75 and cos_sim < -0.5:
                        trk.adjust_continuity(-0.10)
                        cost_matrix[row_i, col_j] = 1e5
                        continue
                else:
                    c_d = 0.0

                aspect_t = trk.pred_w / trk.pred_h
                aspect_d = det_w / det_h
                c_g = (abs(math.log(det_w / trk.pred_w)) +
                       abs(math.log(det_h / trk.pred_h)) +
                       self.cfg.lambda_aspect * abs(math.log(aspect_d / aspect_t)))

                iou = compute_iou(trk_box, det_box)
                c_i = 1.0 - iou
                c_recovery = 0.05 * trk.missed_frames if is_lost else 0.0
                norm_conf = max(0.0, min(1.0, det_c))
                c_conf = 1.0 - norm_conf

                # Zone-dependent cost weight assignment: FAR zone distance cost dominates
                if trk_norm_y < 0.48:
                    w_m, w_i, w_g, w_v, w_d, w_c = 0.55, 0.05, 0.15, 0.15, 0.05, 0.05
                else:
                    w_m, w_i, w_g, w_v, w_d, w_c = 0.40, 0.20, 0.15, 0.15, 0.05, 0.05

                total_cost = w_m * c_m + w_i * c_i + w_g * c_g + w_v * c_v + w_d * c_d + w_c * c_conf + c_recovery

                if trk.identity_locked or (trk.age >= 10 and trk.hits >= 5) or trk.track_identity_confidence >= 0.70:
                    total_cost -= self.cfg.id_hysteresis

                cost_matrix[row_i, col_j] = total_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(det_indices)
        unmatched_trks = list(trk_indices)

        for r, c in zip(row_ind, col_ind):
            cost_val = cost_matrix[r, c]
            if cost_val <= self.cfg.match_cost_threshold:
                t_idx = trk_indices[r]
                d_idx = det_indices[c]

                row_costs = sorted(cost_matrix[r, :].tolist())
                if len(row_costs) > 1 and abs(row_costs[1] - row_costs[0]) < self.cfg.ambiguity_margin:
                    continue

                self.tracks[t_idx].last_matched_cost = float(cost_val)
                matched.append((t_idx, d_idx))

                # Audit log for FAR zone associations exceeding 50px center displacement
                trk_obj = self.tracks[t_idx]
                det_b = det_boxes[d_idx]
                det_center_x = (det_b[0] + det_b[2]) / 2.0
                det_center_y = (det_b[1] + det_b[3]) / 2.0
                disp = math.sqrt((det_center_x - trk_obj.cx)**2 + (det_center_y - trk_obj.cy)**2)
                if (trk_obj.cy / float(self.img_h)) < 0.48 and disp > 50.0:
                    print(f"  [FAR ASSOC AUDIT] Frame {self.frame_counter}: Track #{trk_obj.track_id} matched at {disp:.1f}px displacement (Cost: {cost_val:.3f})")

                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)
                if t_idx in unmatched_trks:
                    unmatched_trks.remove(t_idx)

        return matched, unmatched_dets, unmatched_trks

    def update(self, detections, debug_performance=False):
        self.frame_counter += 1

        # STEP 1: Preserve previous state & predict motion for all tracks
        for trk in self.tracks:
            trk.previous_state = trk.state
            trk.association_event = "NONE"
            trk.matched_det_idx = -1
            trk.predict()

        det_boxes = [d[:4] for d in detections] if len(detections) > 0 else []
        det_confs = [float(d[4]) for d in detections] if len(detections) > 0 else []
        det_tiers = [d[9] if len(d) > 9 else ("STRONG" if d[4] >= 0.25 else ("TRACKABLE" if d[4] >= 0.12 else "WEAK")) for d in detections] if len(detections) > 0 else []

        self.total_raw_detections += len(detections)
        self.total_fused_detections += len(det_boxes)

        high_dets = sum(1 for tier in det_tiers if tier == "STRONG")
        med_dets = sum(1 for tier in det_tiers if tier == "TRACKABLE")
        weak_dets = sum(1 for tier in det_tiers if tier == "WEAK")

        # STEP 2: Strict Association Priority Engine with 5-stage non-overlapping pool consumption
        matched_trks, unmatched_dets, unmatched_trks = self._multi_priority_associate(det_boxes, det_confs, det_tiers)

        matched_cnt = 0
        recovered_cnt = 0

        # STEP 3: Update matched tracks & assign exact classification events
        for trk_idx, det_idx in matched_trks:
            trk = self.tracks[trk_idx]
            trk.matched_det_idx = det_idx
            is_strong = (det_tiers[det_idx] == "STRONG")

            if trk.previous_state == "Confirmed":
                trk.association_event = "MATCHED"
                matched_cnt += 1
            elif trk.previous_state == "Lost":
                trk.association_event = "RECOVERED"
                recovered_cnt += 1
                self.total_recoveries += 1
            elif trk.previous_state == "Candidate":
                trk.association_event = "MATCHED"
                matched_cnt += 1

            prev_state_val = trk.state
            trk.update(detections[det_idx][:4], detections[det_idx][4], is_strong=is_strong)

            if prev_state_val == "Candidate" and trk.state == "Confirmed":
                self.total_candidates_confirmed += 1

        # STEP 4: Handle unmatched tracks and state transitions AFTER association
        for trk_idx in unmatched_trks:
            trk = self.tracks[trk_idx]
            if trk.previous_state == "Confirmed":
                trk.state = "Lost"
                trk.missed_frames = 1
                trk.time_since_update = 1
                trk.association_event = "LOST"
                trk.adjust_continuity(-0.08)
            elif trk.previous_state == "Lost":
                trk.missed_frames += 1
                trk.time_since_update += 1
                trk.association_event = "STILL_LOST"
                trk.adjust_continuity(-0.08)
            elif trk.previous_state == "Candidate":
                trk.missed_frames += 1
                trk.time_since_update += 1
                trk.association_event = "MISSED"

        # STEP 5: Create Candidate tracks ONLY from remaining STRONG detections with BirthScore >= 0.50
        new_tracks_cnt = 0
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            conf = float(det[4])
            tier = det_tiers[det_idx]
            x1, y1, x2, y2 = det[:4]
            det_cy = (y1 + y2) / 2.0

            if tier == "STRONG":
                birth_score = self._compute_birth_score(det)
                if birth_score >= 0.50 and y2 >= self.cfg.roi_ymin * self.img_h:
                    if not self._plausibly_belongs_to_existing_track(det[:4], conf):
                        restored_info = self._resurrect_recently_lost_track(det[:4])
                        if restored_info is not None:
                            old_id, dist_val = restored_info
                            resurrected_trk = None
                            for t in self.tracks:
                                if t.track_id == old_id and t.state == "Lost":
                                    resurrected_trk = t
                                    break
                            if resurrected_trk is not None:
                                resurrected_trk.update(det[:4], conf, is_strong=True)
                                resurrected_trk.state = "Confirmed"
                                resurrected_trk.association_event = "RECOVERED"
                                self.total_recoveries += 1
                                continue

                        restored_info = self._check_and_restore_replacement_track(det[:4])
                        if restored_info is not None:
                            old_id, dist_gap = restored_info
                            new_trk = AMTTrack(det[:4], confidence=conf, class_id=int(det[5]) if len(det) > 5 else 0,
                                               min_birth_hits=self.cfg.min_birth_hits, custom_id=old_id)
                            new_trk.previous_state = "Lost"
                            new_trk.state = "Confirmed"
                            new_trk.association_event = "RESTORED_ID"
                            self.replacement_events += 1
                            self.fragmentation_events += 1
                            self.suspected_replacements.append({
                                'old_id': old_id,
                                'new_id': old_id,
                                'gap': dist_gap[0],
                                'distance': dist_gap[1],
                                'reason': 'Identity Restored (4/4 Evidence)'
                            })
                        else:
                            new_trk = AMTTrack(det[:4], confidence=conf, class_id=int(det[5]) if len(det) > 5 else 0,
                                               min_birth_hits=self.cfg.min_birth_hits)
                            new_trk.previous_state = "NONE"
                            new_trk.association_event = "NEW"
                            self.total_candidates_created += 1

                        self.tracks.append(new_trk)
                        self.all_historical_tracks.append(new_trk)
                        new_tracks_cnt += 1
                        self.stats_births += 1
                    else:
                        self.detections_rejected_by_filters += 1
                else:
                    self.detections_rejected_by_filters += 1
            else:
                self.detections_rejected_by_filters += 1

        # STEP 6: Duplicate Track Suppression
        self._suppress_duplicate_tracks()

        # STEP 6.5: Explicit ID_SWITCH_SUSPECTED Detection & Diagnostic Audit
        self._detect_suspected_id_switches(detections, matched_trks)

        # STEP 7: Prune expired tracks & update recently_lost_or_deleted buffer
        pruned_tracks = []
        for t in self.tracks:
            is_near_horizon = (t.cy <= self.cfg.roi_ymin * self.img_h)
            is_near_border = (t.cx < 15 or t.cx > self.img_w - 15 or t.cy > self.img_h - 15)
            is_exiting = (is_near_horizon or is_near_border) and t.vy <= 0.0
            is_mature_trk = (t.age >= 10 and (t.hits >= 5 or t.track_identity_confidence >= 0.75))

            norm_y = t.cy / float(self.img_h)
            if norm_y < 0.48:
                max_misses = 8
            elif norm_y < 0.68:
                max_misses = 6
            else:
                max_misses = 4

            if is_mature_trk:
                max_misses = max(max_misses, 8 if norm_y < 0.48 else 6)

            if t.state == "Confirmed":
                if t.missed_frames <= max_misses and not (is_exiting and t.missed_frames >= 2):
                    pruned_tracks.append(t)
                else:
                    self.total_tracks_deleted += 1
                    self.orphaned_tracks += 1
                    self._buffer_deleted_track(t)
            elif t.state == "Lost":
                if t.missed_frames <= max_misses and not (is_exiting and t.missed_frames >= 2):
                    pruned_tracks.append(t)
                else:
                    self.total_tracks_deleted += 1
                    self.orphaned_tracks += 1
                    self._buffer_deleted_track(t)
            else:  # Candidate state
                if t.missed_frames <= 2:
                    pruned_tracks.append(t)
                else:
                    self.total_tracks_deleted += 1
        self.tracks = pruned_tracks

        # Clean old records from memory buffer (> 12 frames old)
        self.recently_lost_or_deleted = [r for r in self.recently_lost_or_deleted if (self.frame_counter - r['frame']) <= 12]

        # STEP 8: CSV Export Logging
        if self.csv_path:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                for t in self.tracks:
                    det_x1, det_y1, det_x2, det_y2 = (0.0, 0.0, 0.0, 0.0)
                    act_cx, act_cy = (t.cx, t.cy)
                    pred_cx, pred_cy = (t.predicted_position[0], t.predicted_position[1])
                    source_str = "NONE"
                    zone_str = "FAR" if (t.cy / float(self.img_h)) < 0.40 else ("MEDIUM" if (t.cy / float(self.img_h)) < 0.65 else "NEAR")

                    if t.matched_det_idx >= 0 and t.matched_det_idx < len(detections):
                        det = detections[t.matched_det_idx]
                        det_x1, det_y1, det_x2, det_y2 = det[:4]
                        act_cx = (det_x1 + det_x2) / 2.0
                        act_cy = (det_y1 + det_y2) / 2.0
                        if len(det) > 6:
                            source_str = str(det[6])
                        if len(det) > 8:
                            zone_str = str(det[8])

                    dist_err = math.sqrt((act_cx - pred_cx)**2 + (act_cy - pred_cy)**2)
                    scale_r = max((t.w / max(1.0, t.pred_w)), (t.pred_w / max(1.0, t.w)))

                    writer.writerow([
                        self.frame_counter, t.track_id, t.association_event,
                        f"{det_x1:.1f}", f"{det_y1:.1f}", f"{det_x2:.1f}", f"{det_y2:.1f}",
                        f"{t.confidence:.3f}", source_str, zone_str, t.state, t.missed_frames,
                        f"{pred_cx:.1f}", f"{pred_cy:.1f}", f"{act_cx:.1f}", f"{act_cy:.1f}",
                        f"{dist_err:.2f}", f"{scale_r:.2f}", f"{t.track_identity_confidence:.2f}",
                        t.matched_det_idx, f"{t.last_matched_cost:.3f}"
                    ])

        # STEP 9: Compact ID Transition Debug Table Log
        if debug_performance:
            conf_cnt = sum(1 for t in self.tracks if t.state == "Confirmed")
            lost_cnt = sum(1 for t in self.tracks if t.state == "Lost")
            cand_cnt = sum(1 for t in self.tracks if t.state == "Candidate")

            print(f"Frame {self.frame_counter:3d}: Raw Dets={len(detections)} [Strong:{high_dets}, Trackable:{med_dets}, Weak:{weak_dets}] | Fused={len(det_boxes)} | Confirmed={conf_cnt} | Lost={lost_cnt} | Cand={cand_cnt} | Matched={matched_cnt} | Recovered={recovered_cnt} | New={new_tracks_cnt} | Deleted={self.total_tracks_deleted}")
            print(f"  {'ID':<5} {'PrevState':<12} {'Event':<12} {'Det':<5} {'Miss':<5} {'Hits':<5} {'State':<10}")
            print("  " + "-" * 60)
            for t in self.tracks:
                det_str = str(t.matched_det_idx) if t.matched_det_idx >= 0 else "-"
                print(f"  #{t.track_id:<4d} {t.previous_state:<12} {t.association_event:<12} {det_str:<5} {t.missed_frames:<5d} {t.hits:<5d} {t.state:<10}")
            print("")

        # STEP 10: Render Output Assembly
        active_output = []
        for t in self.tracks:
            if t.state not in ["Confirmed", "Lost"]:
                continue

            is_predicted = False
            render_mode = "OBS"

            if t.missed_frames == 0:
                is_predicted = False
                render_mode = "OBS"
            elif t.missed_frames == 1:
                is_predicted = True
                render_mode = "PRED_1"
            elif t.missed_frames <= 3 and t.sigma_pos <= 40.0:
                is_predicted = True
                render_mode = "PRED_2"
            else:
                continue

            bbox = t.get_bbox()
            active_output.append([
                float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                int(t.track_id), float(t.get_speed_px()), float(t.confidence),
                list(t.trajectory), is_predicted, render_mode
            ])

        return active_output

    def _compute_birth_score(self, det):
        x1, y1, x2, y2, conf = det[:5]
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        aspect_hw = h / w

        # 1. Norm Conf (target conf 0.30+ gives 0.75-1.0)
        norm_conf = min(1.0, float(conf) / 0.40)

        # 2. Geometry Quality (ideal aspect ratio around 1.5 - 2.5)
        if 1.0 <= aspect_hw <= 3.0:
            geom_q = 1.0
        else:
            geom_q = max(0.0, 1.0 - abs(aspect_hw - 2.0))

        # 3. Spatial Validity (well inside frame and ROI)
        y2_norm = y2 / float(self.img_h)
        cx = (x1 + x2) / 2.0
        spatial_v = 1.0 if (y2_norm >= 0.20 and 20.0 <= cx <= (self.img_w - 20.0)) else 0.60

        # 4. Temporal Support Baseline
        temporal_s = 0.80

        birth_score = 0.40 * norm_conf + 0.25 * geom_q + 0.20 * spatial_v + 0.15 * temporal_s
        return float(birth_score)

    def _buffer_deleted_track(self, trk):
        if trk.hits >= 2 or trk.previous_state == "Confirmed":
            self.recently_lost_or_deleted.append({
                'track_id': trk.track_id,
                'last_cx': trk.cx,
                'last_cy': trk.cy,
                'last_w': trk.w,
                'last_h': trk.h,
                'last_vx': trk.vx,
                'last_vy': trk.vy,
                'frame': self.frame_counter,
                'trajectory': list(trk.trajectory),
                'hits': trk.hits,
                'age': trk.age
            })

    def _check_and_restore_replacement_track(self, det_box):
        det_cx = (det_box[0] + det_box[2]) / 2.0
        det_cy = (det_box[1] + det_box[3]) / 2.0
        det_w = max(1.0, det_box[2] - det_box[0])
        det_h = max(1.0, det_box[3] - det_box[1])

        # Exclusivity Guard: do not restore if candidate detection is within 30px of active confirmed track
        for trk in self.tracks:
            if trk.state == "Confirmed":
                if math.sqrt((det_cx - trk.cx)**2 + (det_cy - trk.cy)**2) < 30.0:
                    return None

        best_match = None
        min_dist = 1e5

        for record in self.recently_lost_or_deleted:
            gap = self.frame_counter - record['frame']
            if gap <= 0 or gap > 8:
                continue

            v_mag = math.sqrt(record['last_vx']**2 + record['last_vy']**2)
            d_restore = min(100.0, 30.0 + v_mag * gap + 10.0)

            pred_cx = record['last_cx'] + record['last_vx'] * gap
            pred_cy = record['last_cy'] + record['last_vy'] * gap

            dist = math.sqrt((det_cx - pred_cx)**2 + (det_cy - pred_cy)**2)
            scale_r_w = max(det_w / max(1.0, record['last_w']), record['last_w'] / max(1.0, det_w))
            scale_r_h = max(det_h / max(1.0, record['last_h']), record['last_h'] / max(1.0, det_h))

            # Check 1: Position
            if dist > d_restore:
                continue

            # Check 2: Scale (up to 1.60x for 1-frame gap)
            max_scale_del = 1.60 if gap == 1 else 1.40
            if scale_r_w > max_scale_del or scale_r_h > max_scale_del:
                continue

            # Check 3: Direction (if trajectory history >= 4)
            # Skip direction check for 1-frame gap with close spatial proximity (dist <= 25.0px)
            if len(record['trajectory']) >= 4 and v_mag >= 5.0 and not (gap == 1 and dist <= 25.0):
                obs_vx = (det_cx - record['last_cx']) / float(gap)
                obs_vy = (det_cy - record['last_cy']) / float(gap)
                obs_mag = math.sqrt(obs_vx**2 + obs_vy**2) + 1e-5
                cos_sim = (record['last_vx'] * obs_vx + record['last_vy'] * obs_vy) / (v_mag * obs_mag)
                if cos_sim < 0.5:
                    continue

            # Check 4: Normalized Trajectory Error (perpendicular error)
            # Skip trajectory error check for 1-frame gap with close spatial proximity (dist <= 25.0px)
            if not (gap == 1 and dist <= 25.0):
                disp_x = det_cx - record['last_cx']
                disp_y = det_cy - record['last_cy']
                perp_dist = abs(disp_x * record['last_vy'] - disp_y * record['last_vx']) / (v_mag + 1e-5)
                traj_err = perp_dist / max(20.0, 5.0 * gap)
                if traj_err > 1.0:
                    continue

            if dist < min_dist:
                min_dist = dist
                best_match = (record['track_id'], (gap, dist))

        return best_match

    def _resurrect_recently_lost_track(self, det_box):
        det_cx = (det_box[0] + det_box[2]) / 2.0
        det_cy = (det_box[1] + det_box[3]) / 2.0
        det_w = max(1.0, det_box[2] - det_box[0])
        det_h = max(1.0, det_box[3] - det_box[1])

        # Active Track Priority & Collision Guard: verify det is not within 20px of active Confirmed track
        for trk in self.tracks:
            if trk.state == "Confirmed":
                if math.sqrt((det_cx - trk.cx)**2 + (det_cy - trk.cy)**2) < 20.0:
                    return None

        candidates = []

        for trk in self.tracks:
            if trk.state != "Lost":
                continue

            gap = trk.missed_frames
            if gap <= 0 or gap > 8:
                continue

            norm_y = trk.cy / float(self.img_h)
            v_mag = trk.get_speed_px()

            if norm_y < 0.48:
                d_gate = min(45.0, 10.0 + 1.5 * v_mag + 5.0 * (gap - 1))
            else:
                d_gate = min(60.0, 15.0 + 1.8 * v_mag + 8.0 * (gap - 1))

            pred_cx = trk.cx + trk.vx * gap
            pred_cy = trk.cy + trk.vy * gap

            dist = math.sqrt((det_cx - pred_cx)**2 + (det_cy - pred_cy)**2)
            scale_r_w = max(det_w / max(1.0, trk.pred_w), trk.pred_w / max(1.0, det_w))
            scale_r_h = max(det_h / max(1.0, trk.pred_h), trk.pred_h / max(1.0, det_h))
            scale_r = max(scale_r_w, scale_r_h)

            max_scale = 1.60 if gap == 1 else 1.35
            if scale_r > max_scale:
                continue

            if dist > d_gate:
                continue

            disp_x = det_cx - trk.cx
            disp_y = det_cy - trk.cy
            disp_mag = math.sqrt(disp_x**2 + disp_y**2) + 1e-5

            cos_sim = 1.0
            if v_mag >= 2.0 and disp_mag >= 0.5:
                cos_sim = (trk.vx * disp_x + trk.vy * disp_y) / (v_mag * disp_mag)
                if cos_sim < 0.5 and not (gap == 1 and dist <= 25.0):
                    continue

            candidates.append({
                'track_id': trk.track_id,
                'dist': dist,
                'gate': d_gate,
                'gap': gap,
                'scale_r': scale_r,
                'v_mag': v_mag,
                'cos_sim': cos_sim,
                'zone': "FAR" if norm_y < 0.48 else ("MEDIUM" if norm_y < 0.68 else "NEAR")
            })

        if not candidates:
            return None

        # Ambiguity Guard
        candidates.sort(key=lambda x: x['dist'])
        best = candidates[0]
        second_best_dist = candidates[1]['dist'] if len(candidates) > 1 else 999.0

        if len(candidates) > 1:
            if (second_best_dist - best['dist']) < 12.0 and (second_best_dist / max(1.0, best['dist'])) < 1.5:
                return None

        print(f"  [P5 RESURRECTION] frame={self.frame_counter} old_id=#{best['track_id']} gap={best['gap']} dist={best['dist']:.1f}px gate={best['gate']:.1f}px scale_r={best['scale_r']:.2f} vel={best['v_mag']:.1f} cos={best['cos_sim']:.2f} sec_best={second_best_dist:.1f} zone={best['zone']}")
        return (best['track_id'], best['dist'])
