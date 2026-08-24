import time
import cv2
import csv
import argparse
import numpy as np
from pathlib import Path

from detector_2w import get_detector
from custom_tracker import AMTTracker, AMTConfig


def run_small_vehicle_pipeline(input_video="input_720p_10fps.mp4",
                                output_video="annotated_output_720p_10fps.mp4",
                                detector_type="p2_tiling",
                                weights=None,
                                conf_thresh=0.05,
                                show_detections=False,
                                debug_performance=False,
                                debug_detector=False,
                                max_frames=0):
    input_path = Path(input_video)
    output_path = Path(output_video)

    if not input_path.exists():
        print(f"Error: Input video '{input_path.resolve()}' not found.")
        return

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{input_path}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    det_name = "YOLOv8 + Far Tiling" if detector_type.lower() in ["p2_tiling", "tiled_yolo", "p2", "tiling", "yolo_p2", "yolo_tiling", "tiled_yolov8"] else detector_type.upper()

    print(f"\n{'='*65}")
    print(f"  AMT (Adaptive Motion & Geometry) 2-Wheeler Pipeline")
    print(f"{'='*65}")
    print(f"  Input:             {input_path.name}")
    print(f"  Resolution:        {width}x{height} @ {fps:.2f} FPS")
    print(f"  Total Frames:      {total_frames}")
    print(f"  Detector:          {det_name} (1 Global + 4 Far Tiles)")
    print(f"  Tracker:           AMT (Adaptive Motion and Geometry Tracker)")
    print(f"  Show Raw Dets:     {show_detections}")
    print(f"  Debug Performance: {debug_performance}")
    print(f"  Debug Detector:    {debug_detector}")
    print(f"  Output Video:      {output_path.name}")
    print(f"{'='*65}\n")

    # Initialize Detector & AMT Production Tracker
    detector = get_detector(detector_type=detector_type, weights=weights, conf_thresh=conf_thresh)
    cfg = AMTConfig(max_age=15, min_hits=2, match_cost_threshold=0.65, roi_ymin=0.05)
    tracker = AMTTracker(config=cfg, img_h=height, img_w=width)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Initialize CSV Diagnostic Export files if debug_detector is True
    diag_csv_file = None
    frames_csv_file = None
    nms_csv_file = None
    diag_writer = None
    frames_writer = None
    nms_writer = None

    if debug_detector:
        diag_csv_file = open("detector_recall_diagnostic.csv", mode='w', newline='')
        diag_writer = csv.writer(diag_csv_file)
        diag_writer.writerow([
            'Frame', 'Raw_ID', 'Source', 'X1', 'Y1', 'X2', 'Y2', 'W', 'H',
            'Aspect_HW', 'Conf', 'Zone', 'Tier', 'Result', 'Reason'
        ])

        frames_csv_file = open("detector_recall_frames.csv", mode='w', newline='')
        frames_writer = csv.writer(frames_csv_file)
        frames_writer.writerow([
            'Frame', 'Global_Raw', 'Far_Tile_01_Raw', 'Far_Tile_02_Raw', 'Far_Tile_03_Raw', 'Far_Tile_04_Raw',
            'Total_Raw', 'Geometry_Passed', 'Fused_Passed', 'Tracker_Associated',
            'ROI_Rejects', 'Border_Rejects', 'Min_Size_Rejects', 'Aspect_Rejects', 'Low_Conf_Rejects', 'NMS_Suppressed'
        ])

        nms_csv_file = open("detector_nms_diagnostic.csv", mode='w', newline='')
        nms_writer = csv.writer(nms_csv_file)
        nms_writer.writerow([
            'Frame', 'Suppressed_Raw_ID', 'Kept_Raw_ID', 'IoU', 'Center_Distance',
            'Conf_Suppressed', 'Conf_Kept', 'Source_Suppressed', 'Source_Kept'
        ])

    frame_count = 0
    total_inference_time = 0.0
    total_tracking_time = 0.0

    # Cumulative Diagnostic Counters across all frames
    cum_stats = {
        'total_raw': 0, 'pass_global': 0, 'pass_tile1': 0, 'pass_tile2': 0, 'pass_tile3': 0, 'pass_tile4': 0,
        'geom_passed': 0, 'fused_passed': 0, 'tracker_associated': 0,
        'ROI': 0, 'BORDER': 0, 'MIN_SIZE': 0, 'ASPECT': 0, 'LOW_CONF': 0, 'NMS_SUPPRESSED': 0
    }

    print("Processing video frames...")
    start_pipeline_time = time.time()

    progress_interval = 10 if (debug_performance or debug_detector) else 50

    while True:
        if max_frames > 0 and frame_count >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t0 = time.time()

        # 1. Detect 2-wheelers using selected detector
        is_first = (frame_count == 1)
        detections = detector.detect(frame, debug_performance=debug_performance, debug_detector=debug_detector, is_first_frame=is_first)
        t1 = time.time()

        # 2. Update AMT production tracker
        active_tracks = tracker.update(detections, debug_performance=debug_performance)
        t2 = time.time()

        det_time = (t1 - t0) * 1000.0
        trk_time = (t2 - t1) * 1000.0
        total_frame_ms = (t2 - t0) * 1000.0

        total_inference_time += det_time
        total_tracking_time += trk_time

        # Count how many fused detections were associated by tracker
        assoc_cnt = sum(1 for t in tracker.tracks if t.association_event in ["MATCHED", "RECOVERED"])

        if debug_detector and hasattr(detector, 'last_diagnostic_records'):
            diag_records = detector.last_diagnostic_records
            stats = detector.last_frame_diagnostic_stats
            pass_counts = stats.get('pass_counts', {})
            rej_counts = stats.get('rejection_counts', {})

            cum_stats['total_raw'] += stats.get('raw_total', 0)
            cum_stats['pass_global'] += pass_counts.get('GLOBAL', 0)
            cum_stats['pass_tile1'] += pass_counts.get('FAR_TILE_01', 0)
            cum_stats['pass_tile2'] += pass_counts.get('FAR_TILE_02', 0)
            cum_stats['pass_tile3'] += pass_counts.get('FAR_TILE_03', 0)
            cum_stats['pass_tile4'] += pass_counts.get('FAR_TILE_04', 0)
            cum_stats['geom_passed'] += stats.get('geometry_passed', 0)
            cum_stats['fused_passed'] += stats.get('fused_passed', 0)
            cum_stats['tracker_associated'] += assoc_cnt

            for r_key in ['ROI', 'BORDER', 'MIN_SIZE', 'ASPECT', 'LOW_CONF', 'NMS_SUPPRESSED']:
                cum_stats[r_key] += rej_counts.get(r_key, 0)

            # Export candidates to detector_recall_diagnostic.csv
            for rec in diag_records:
                x1, y1, x2, y2 = rec['box']
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                aspect = h / w
                diag_writer.writerow([
                    frame_count, rec['raw_id'], rec['source'],
                    f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                    f"{w:.1f}", f"{h:.1f}", f"{aspect:.2f}",
                    f"{rec['conf']:.4f}", rec['zone'], rec['tier'], rec['result'], rec['reason']
                ])

            # Export NMS suppression records to detector_nms_diagnostic.csv
            if hasattr(detector, 'last_nms_records'):
                for nms_rec in detector.last_nms_records:
                    nms_writer.writerow([
                        frame_count, nms_rec['suppressed_raw_id'], nms_rec['kept_raw_id'],
                        f"{nms_rec['iou']:.4f}", f"{nms_rec['center_distance']:.2f}",
                        f"{nms_rec['conf_suppressed']:.4f}", f"{nms_rec['conf_kept']:.4f}",
                        nms_rec['source_suppressed'], nms_rec['source_kept']
                    ])

            # Export per-frame summary to detector_recall_frames.csv
            frames_writer.writerow([
                frame_count, pass_counts.get('GLOBAL', 0),
                pass_counts.get('FAR_TILE_01', 0), pass_counts.get('FAR_TILE_02', 0),
                pass_counts.get('FAR_TILE_03', 0), pass_counts.get('FAR_TILE_04', 0),
                stats.get('raw_total', 0), stats.get('geometry_passed', 0),
                stats.get('fused_passed', 0), assoc_cnt,
                rej_counts.get('ROI', 0), rej_counts.get('BORDER', 0),
                rej_counts.get('MIN_SIZE', 0), rej_counts.get('ASPECT', 0),
                rej_counts.get('LOW_CONF', 0), rej_counts.get('NMS_SUPPRESSED', 0)
            ])

        # 3. Render Annotations
        annotated_frame = frame.copy()

        # Deterministic per-track color palette
        COLOR_PALETTE = [
            (0, 255, 127),    # Bright Spring Green
            (255, 165, 0),    # Bright Orange
            (0, 215, 255),    # Gold / Yellow
            (255, 105, 180),  # Deep Pink
            (50, 205, 50),    # Lime Green
            (0, 191, 255),    # Deep Sky Blue
            (238, 130, 238),  # Violet
            (255, 215, 0),    # Gold
            (127, 255, 212),  # Aquamarine
            (255, 140, 0)     # Dark Orange
        ]

        def get_track_color(tid):
            return COLOR_PALETTE[int(tid) % len(COLOR_PALETTE)]

        # Render 3-Color Diagnostic Boxes when debug_detector is True
        if debug_detector and hasattr(detector, 'last_diagnostic_records'):
            for rec in detector.last_diagnostic_records:
                x1, y1, x2, y2 = [int(v) for v in rec['box']]
                res = rec['result']
                reas = rec['reason']
                c_val = rec['conf']

                if res == "PASSED":
                    box_color = (0, 255, 0)  # GREEN: Passed fused detection
                    thick = 2
                elif reas in ["ASPECT", "MIN_SIZE", "ROI", "BORDER", "NMS_SUPPRESSED"]:
                    box_color = (0, 255, 255)  # YELLOW: Filter / NMS rejected
                    thick = 1
                else:  # LOW_CONF
                    box_color = (0, 0, 255)  # RED: Low confidence rejected
                    thick = 1

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thick)
                cv2.putText(annotated_frame, f"{reas[:4]} {c_val:.2f}", (x1, max(0, y1 - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, box_color, 1)

        elif show_detections:
            for det in detections:
                dx1, dy1, dx2, dy2, dconf, dcls = det[:6]
                dsrc = det[6] if len(det) > 6 else "DET"
                color = (180, 180, 180)
                cv2.rectangle(annotated_frame, (int(dx1), int(dy1)), (int(dx2), int(dy2)), color, 1)
                cv2.putText(annotated_frame, f"RAW [{dsrc[:3]}] {dconf:.2f}", (int(dx1), max(0, int(dy1) - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Clean, consistent high-visibility color scheme
        CONFIRMED_COLOR = (0, 255, 127)   # Bright Spring/Emerald Green for all confirmed bikes
        PREDICTED_COLOR = (0, 165, 255)   # Amber/Orange for predicted/lost state

        for trk in active_tracks:
            x1, y1, x2, y2, track_id, velocity, conf, trajectory, is_predicted, render_mode = trk
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            track_color = PREDICTED_COLOR if is_predicted else get_track_color(track_id)

            if len(trajectory) > 1:
                pts = [np.array(p, dtype=np.int32) for p in trajectory]
                for k in range(1, len(pts)):
                    cv2.line(annotated_frame, tuple(pts[k-1]), tuple(pts[k]), track_color, 2, cv2.LINE_AA)

            line_thickness = 1 if is_predicted else 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), track_color, line_thickness)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(annotated_frame, (cx, cy), 3, track_color, -1)

            if debug_performance:
                status_tag = " [PRED]" if is_predicted else ""
                label = f"ID #{track_id}{status_tag} | {velocity:.1f} px/f"
            else:
                label = f"ID #{track_id}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_y1 = max(0, y1 - th - 6)
            cv2.rectangle(annotated_frame, (x1, badge_y1), (x1 + tw + 6, y1), track_color, -1)
            cv2.putText(annotated_frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        inst_fps = 1.0 / max(1e-5, (t2 - t0))
        if debug_detector:
            info_text = f"Frame {frame_count}/{total_frames} | Raw: {detector.last_frame_diagnostic_stats.get('raw_total', 0)} | Fused: {len(detections)} | Active Bikes: {len(active_tracks)}"
        elif debug_performance:
            info_text = f"Frame {frame_count}/{total_frames} | Det [{det_name}]: {det_time:.1f}ms | AMT Trk: {trk_time:.1f}ms | Active: {len(active_tracks)}"
        else:
            info_text = f"Frame {frame_count}/{total_frames} | Active Bikes: {len(active_tracks)}"

        cv2.rectangle(annotated_frame, (10, 10), (600, 42), (0, 0, 0), -1)
        cv2.putText(annotated_frame, info_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        out.write(annotated_frame)

        if debug_detector:
            st = detector.last_frame_diagnostic_stats
            print(f"Frame {frame_count:2d}/{total_frames}: RAW={st.get('raw_total', 0)} [Global:{st['pass_counts'].get('GLOBAL',0)}, Far1:{st['pass_counts'].get('FAR_TILE_01',0)}, Far2:{st['pass_counts'].get('FAR_TILE_02',0)}, Far3:{st['pass_counts'].get('FAR_TILE_03',0)}, Far4:{st['pass_counts'].get('FAR_TILE_04',0)}] | GeomPass={st.get('geometry_passed', 0)} | FusedPass={st.get('fused_passed', 0)} | ActiveTrks={len(active_tracks)}")
        elif debug_performance:
            print(f"Frame {frame_count:2d}/{total_frames}: Total={total_frame_ms:.1f}ms (Det={det_time:.1f}ms, Trk={trk_time:.1f}ms) Active={len(active_tracks)} Dets={len(detections)}")
        elif frame_count % progress_interval == 0 or frame_count == total_frames:
            print(f"  Frame {frame_count}/{total_frames}: Processed at {inst_fps:.1f} FPS (Active: {len(active_tracks)})")

    cap.release()
    out.release()

    if diag_csv_file:
        diag_csv_file.close()
    if frames_csv_file:
        frames_csv_file.close()
    if nms_csv_file:
        nms_csv_file.close()

    total_elapsed = time.time() - start_pipeline_time
    avg_fps = frame_count / max(1e-5, total_elapsed)
    avg_det_ms = total_inference_time / max(1, frame_count)
    avg_trk_ms = total_tracking_time / max(1, frame_count)

    print(f"\n{'='*65}")
    print(f"  AMT Pipeline Processing Complete!")
    print(f"{'='*65}")
    print(f"  Total Processed Frames: {frame_count}")
    print(f"  Total Elapsed Time:    {total_elapsed:.2f} seconds")
    print(f"  Average Throughput:    {avg_fps:.2f} FPS")
    print(f"  Avg Detection Latency: {avg_det_ms:.2f} ms / frame")
    print(f"  Avg Tracking Latency:  {avg_trk_ms:.2f} ms / frame")
    print(f"  Annotated Video Saved: {output_path.resolve()}")
    print(f"{'='*65}\n")

    if debug_detector:
        print("========================================================")
        print(f"DETECTOR RECALL DIAGNOSTIC — {frame_count} FRAMES")
        print("========================================================")
        print(f"Raw YOLO detections (conf=0.001):   {cum_stats['total_raw']}")
        print(f"  - GLOBAL Pass:                    {cum_stats['pass_global']}")
        print(f"  - FAR_TILE_01:                    {cum_stats['pass_tile1']}")
        print(f"  - FAR_TILE_02:                    {cum_stats['pass_tile2']}")
        print(f"  - FAR_TILE_03:                    {cum_stats['pass_tile3']}")
        print(f"  - FAR_TILE_04:                    {cum_stats['pass_tile4']}")
        print("\nGeometry & Pre-Filtering Breakdown:")
        print(f"  - ROI rejected (y2 < 0.18*H):     {cum_stats['ROI']}")
        print(f"  - Border edge rejected:           {cum_stats['BORDER']}")
        print(f"  - Min/Max size rejected:          {cum_stats['MIN_SIZE']}")
        print(f"  - Aspect ratio rejected:          {cum_stats['ASPECT']}")
        print(f"  - Low confidence (< 0.05):        {cum_stats['LOW_CONF']}")
        print(f"  - Geometry filter passed:         {cum_stats['geom_passed']}")
        print("\nFusion & Tracker Breakdown:")
        print(f"  - NMS suppressed during fusion:   {cum_stats['NMS_SUPPRESSED']}")
        print(f"  - Final accepted fused:           {cum_stats['fused_passed']}")
        print(f"  - Tracker associated:             {cum_stats['tracker_associated']}")
        print("========================================================\n")
        print(f"Detailed Diagnostic CSV Exports:")
        print(f"  - detector_recall_diagnostic.csv")
        print(f"  - detector_recall_frames.csv")
        print(f"  - detector_nms_diagnostic.csv\n")

    if hasattr(tracker, 'print_summary_diagnostics'):
        tracker.print_summary_diagnostics()


def main():
    parser = argparse.ArgumentParser(description="AMT 2-Wheeler Detection and Tracking Pipeline.")
    parser.add_argument("--input", type=str, default="input_720p_10fps.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="annotated_output_720p_10fps.mp4", help="Path to output annotated video")
    parser.add_argument("--detector", type=str, default="p2_tiling", choices=["p2_tiling", "tiled_yolo", "lcnet", "yolo"], help="Detector model architecture")
    parser.add_argument("--weights", type=str, default=None, help="Custom detector model weights path")
    parser.add_argument("--conf", type=float, default=0.05, help="Detection confidence threshold")
    parser.add_argument("--show-detections", action="store_true", help="Show raw detector candidate boxes alongside tracks")
    parser.add_argument("--debug-performance", action="store_true", help="Enable detailed tile-by-tile and per-frame observable timing")
    parser.add_argument("--debug-detector", action="store_true", help="Enable ultra-low conf detector recall diagnostic mode")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to process (0 = process all)")

    args = parser.parse_args()
    run_small_vehicle_pipeline(input_video=args.input, output_video=args.output,
                                detector_type=args.detector, weights=args.weights,
                                conf_thresh=args.conf, show_detections=args.show_detections,
                                debug_performance=args.debug_performance, debug_detector=args.debug_detector,
                                max_frames=args.max_frames)

if __name__ == "__main__":
    main()

