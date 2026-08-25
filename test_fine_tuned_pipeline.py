#!/usr/bin/env python3
"""
test_fine_tuned_pipeline.py

Testing script that integrates a fine-tuned YOLOv8 model with the existing AMT (Adaptive Motion & Geometry) Tracker.
Features:
  1. Loads fine-tuned YOLOv8 weights (or baseline YOLO weights).
  2. Runs detection (with optional Far-Horizon Zoom Tiling for sub-20px targets).
  3. Feeds detections into the existing `AMTTracker` from `custom_tracker.py`.
  4. Evaluates tracking performance, computes statistics, and renders annotated output video.
  5. Includes automated integration unit tests.
"""

import os
import cv2
import time
import argparse
import unittest
import numpy as np
from pathlib import Path

from detector_2w import get_detector, TiledYOLOP2Detector, SmallVehicleDetector
from custom_tracker import AMTTracker, AMTConfig, compute_iou


def test_detector_and_tracker(model_weights, input_video, output_video, conf_thresh=0.05,
                               detector_type="p2_tiling", max_frames=0):
    """
    Runs end-to-end testing of fine-tuned detector + existing AMT Tracker on a video sequence.
    """
    input_path = Path(input_video)
    output_path = Path(output_video)

    if not input_path.exists():
        print(f"Error: Input video '{input_path}' not found.")
        return False

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Unable to open video '{input_path}'")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    print("\n" + "=" * 70)
    print("  Testing Fine-Tuned Detector + Existing AMT Tracker Pipeline")
    print("=" * 70)
    print(f"  - Input Video     : {input_path.name}")
    print(f"  - Resolution      : {width}x{height} @ {fps:.2f} FPS")
    print(f"  - Model Weights   : {model_weights}")
    print(f"  - Detector Mode   : {detector_type}")
    print(f"  - Conf Threshold  : {conf_thresh}")
    print(f"  - Output Video    : {output_path.name}")
    print("=" * 70 + "\n")

    # Initialize fine-tuned detector using factory
    detector = get_detector(detector_type=detector_type, weights=model_weights, conf_thresh=conf_thresh)

    # Initialize existing production AMT Tracker from custom_tracker.py
    cfg = AMTConfig(
        max_age=15,
        min_hits=2,
        match_cost_threshold=0.65,
        roi_ymin=0.03
    )
    tracker = AMTTracker(config=cfg, img_h=height, img_w=width)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    total_det_time = 0.0
    total_trk_time = 0.0
    total_detections_count = 0
    active_tracks_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break

        # 1. Detector Pass
        t0 = time.time()
        raw_detections = detector.detect(frame)
        t1 = time.time()
        det_latency = (t1 - t0) * 1000.0
        total_det_time += det_latency

        total_detections_count += len(raw_detections)

        # 2. Existing AMT Tracker Update Pass
        t2 = time.time()
        active_tracks = tracker.update(raw_detections)
        t3 = time.time()
        trk_latency = (t3 - t2) * 1000.0
        total_trk_time += trk_latency

        active_tracks_history[frame_idx] = len(active_tracks)

        # 3. Render Annotations on Output Video Frame
        vis_frame = frame.copy()

        # Draw Raw Detections (thin orange bounding boxes)
        for det in raw_detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 1)

        # Draw Confirmed/Active Tracker Bounding Boxes & Trajectories
        for trk in active_tracks:
            x1, y1, x2, y2, trk_id, hits, misses, state, is_pred = trk[:9]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            color = (0, 255, 0) if not is_pred else (0, 255, 255)  # Green if detected, Yellow if predicted
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw Track ID label
            label = f"ID:{trk_id}"
            if is_pred:
                label += f" [P{misses}]"

            cv2.putText(vis_frame, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(vis_frame, (cx, cy), 3, color, -1)

        # Draw Frame Statistics Overlay
        avg_det_ms = total_det_time / frame_idx
        avg_trk_ms = total_trk_time / frame_idx
        fps_curr = 1000.0 / max(1.0, det_latency + trk_latency)

        cv2.putText(vis_frame, f"Frame: {frame_idx}/{total_frames} | FPS: {fps_curr:.1f}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Detections: {len(raw_detections)} | Tracks: {len(active_tracks)}",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Det Latency: {det_latency:.1f}ms | Trk Latency: {trk_latency:.2f}ms",
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(vis_frame)

        if frame_idx % 25 == 0 or frame_idx == total_frames:
            print(f"Processed Frame {frame_idx}/{total_frames} | Dets: {len(raw_detections)} | "
                  f"Active Tracks: {len(active_tracks)} | Latency: {det_latency:.1f}ms (det) + {trk_latency:.2f}ms (trk)")

    cap.release()
    out.release()

    # Print Summary Report
    avg_det_ms = total_det_time / max(1, frame_idx)
    avg_trk_ms = total_trk_time / max(1, frame_idx)
    total_fps = 1000.0 / max(1.0, avg_det_ms + avg_trk_ms)

    print("\n" + "=" * 70)
    print("  Test Pipeline Execution Summary")
    print("=" * 70)
    print(f"  Total Processed Frames : {frame_idx}")
    print(f"  Total Raw Detections   : {total_detections_count}")
    print(f"  Avg Detector Latency   : {avg_det_ms:.2f} ms/frame ({1000.0 / avg_det_ms:.1f} FPS)")
    print(f"  Avg Tracker Latency    : {avg_trk_ms:.2f} ms/frame ({1000.0 / avg_trk_ms:.1f} FPS)")
    print(f"  End-to-End Throughput  : {total_fps:.1f} FPS")
    print(f"  Output Video Saved To  : {output_path.resolve()}")
    print("=" * 70 + "\n")

    return True


class TestFineTunedPipelineUnit(unittest.TestCase):
    """
    Unit test cases for validating detector and existing AMT Tracker integration.
    """
    def setUp(self):
        self.cfg = AMTConfig(max_age=15, min_hits=2, match_cost_threshold=0.65, roi_ymin=0.03)
        self.tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)

    def test_tracker_initialization(self):
        """Test existing tracker initialization with custom config."""
        self.assertIsNotNone(self.tracker)
        self.assertEqual(len(self.tracker.tracks), 0)

    def test_detection_to_tracker_flow(self):
        """Test feeding mock detector outputs into existing AMT Tracker."""
        # Simulated sequence of moving two-wheeler detections from fine-tuned detector
        detections_f1 = [[500.0, 200.0, 530.0, 240.0, 0.88, 3]]
        detections_f2 = [[505.0, 201.0, 535.0, 241.0, 0.90, 3]]

        # Frame 1: Candidate track birth (min_hits=2 required for confirmed display)
        active1 = self.tracker.update(detections_f1)
        self.assertEqual(len(active1), 0, "Frame 1 candidate track should not be confirmed yet")

        # Frame 2: Confirmed track promotion
        active2 = self.tracker.update(detections_f2)
        self.assertEqual(len(active2), 1, "Frame 2 candidate should be promoted to confirmed track")
        self.assertEqual(active2[0][4], 1, "First track ID should be 1")

    def test_missed_detection_velocity_dampening(self):
        """Test tracker state persistence during temporary detector loss."""
        # Confirm track on frames 1 and 2
        self.tracker.update([[400.0, 300.0, 430.0, 340.0, 0.85, 3]])
        active = self.tracker.update([[410.0, 300.0, 440.0, 340.0, 0.85, 3]])
        trk_id = active[0][4]

        # Simulate 2 missed detector frames
        miss_active1 = self.tracker.update([])
        miss_active2 = self.tracker.update([])

        self.assertEqual(len(miss_active1), 1)
        self.assertTrue(miss_active1[0][8], "Predicted track should have is_predicted=True")

        # Re-detect vehicle on frame 5
        recovered = self.tracker.update([[430.0, 300.0, 460.0, 340.0, 0.85, 3]])
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0][4], trk_id, "Track ID must be preserved after recovery")


def main():
    parser = argparse.ArgumentParser(description="Test Fine-Tuned YOLO Detector + Existing AMT Tracker")
    parser.add_argument("--weights", type=str, default="models/best_yolo_custom.pt",
                        help="Path to fine-tuned YOLO model weights (or models/yolov8s.pt / models/yolov8m.pt)")
    parser.add_argument("--input", type=str, default="13105476_720p_10fps_35s.mp4",
                        help="Input video file to test")
    parser.add_argument("--output", type=str, default="test_fine_tuned_result.mp4",
                        help="Output annotated video path")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="Detection confidence threshold")
    parser.add_argument("--detector", type=str, default="p2_tiling",
                        help="Detector mode: 'p2_tiling' (recommended for small objects) or 'yolo'")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to process (0 for full video)")
    parser.add_argument("--run-unit-tests", action="store_true",
                        help="Run automated integration unit tests")

    args = parser.parse_args()

    if args.run_unit_tests:
        print("Running pipeline integration unit tests...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestFineTunedPipelineUnit)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
        return

    # Determine model weights to use
    weights = args.weights
    if not os.path.exists(weights):
        for candidate in ["models/best_yolo_custom.pt", "custom_model/best_yolo_custom.pt", "models/yolov8m.pt", "models/yolov8s.pt", "yolo/yolov8m.pt", "yolo/yolov8s.pt"]:
            if os.path.exists(candidate):
                weights = candidate
                break
        else:
            weights = "yolov8s.pt"


    test_detector_and_tracker(
        model_weights=weights,
        input_video=args.input,
        output_video=args.output,
        conf_thresh=args.conf,
        detector_type=args.detector,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
