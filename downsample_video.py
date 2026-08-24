import cv2
import argparse
from pathlib import Path

def downsample_video(input_path, output_path, target_w=1280, target_h=720, target_fps=10.0):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Error: Input video '{input_path.resolve()}' not found.")
        return False

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_path}'")
        return False

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input Video: {input_path.name}")
    print(f"  - Original Resolution: {orig_w}x{orig_h}")
    print(f"  - Original FPS: {orig_fps:.2f}")
    print(f"  - Original Frame Count: {orig_total_frames}")

    # Determine frame step to achieve ~target_fps
    frame_step = max(1, int(round(orig_fps / target_fps)))
    actual_fps = orig_fps / frame_step
    
    print(f"Target Video:")
    print(f"  - Target Resolution: {target_w}x{target_h}")
    print(f"  - Sampling Step: Every {frame_step} frame(s)")
    print(f"  - Output FPS: {actual_fps:.2f}")
    print(f"  - Output File: {output_path.resolve()}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, actual_fps, (target_w, target_h))

    frame_idx = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            resized_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            out.write(resized_frame)
            saved_frames += 1

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done downsampling! Saved {saved_frames} frames to '{output_path.name}'. Duration: {saved_frames / actual_fps:.1f}s\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="Downsample video to 720p (1280x720) at 10 FPS.")
    parser.add_argument("--input", type=str, default="video_20260822_095839.mp4", help="Input video path")
    parser.add_argument("--output", type=str, default="video_20260822_095839_720p_10fps.mp4", help="Output 720p 10fps video path")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS (default: 10.0)")
    parser.add_argument("--width", type=int, default=1280, help="Target width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target height (default: 720)")

    args = parser.parse_args()
    downsample_video(args.input, args.output, target_w=args.width, target_h=args.height, target_fps=args.fps)

if __name__ == "__main__":
    main()
