#!/usr/bin/env python3
"""
train_yolov8_custom.py

Script to fine-tune YOLOv8 (e.g., YOLOv8s / YOLOv8m) on the custom small two-wheeler dataset.
Features:
  1. Parses VGG Image Annotator (VIA) CSV format annotations (`via_project_22Aug2026_10h43m_csv.csv`).
  2. Converts VIA bounding boxes into normalized YOLO format (.txt files).
  3. Splits frames into train/val subsets and creates the YOLO dataset structure and dataset.yaml.
  4. Fine-tunes YOLOv8 using Ultralytics API.
"""

import os
import csv
import json
import shutil
import random
import argparse
import cv2
import torch
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


def parse_via_csv(csv_path):
    """
    Parses a VIA CSV export file and returns annotations grouped by image filename.
    Returns: dict mapping filename -> list of [x, y, w, h]
    """
    annotations = defaultdict(list)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"VIA CSV file not found: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('filename', '').strip()
            if not fname:
                continue

            try:
                region_shape = json.loads(row.get('region_shape_attributes', '{}'))
                shape_name = region_shape.get('name', '')
                if shape_name == 'rect':
                    x = float(region_shape.get('x', 0))
                    y = float(region_shape.get('y', 0))
                    w = float(region_shape.get('width', region_shape.get('w', 0)))
                    h = float(region_shape.get('height', region_shape.get('h', 0)))

                    if w > 0 and h > 0:
                        annotations[fname].append([x, y, w, h])
            except Exception as e:
                # Ignore invalid rows or empty region attributes
                continue

    return annotations


def prepare_yolo_dataset(csv_path, frames_dir, output_dir, val_split=0.2, seed=42):
    """
    Converts VIA CSV annotations to YOLO format and splits frames into train/val directories.
    Creates dataset.yaml for Ultralytics YOLO training.
    """
    print(f"--> Parsing VIA CSV annotations from: {csv_path}")
    annotations = parse_via_csv(csv_path)
    print(f"Found annotations for {len(annotations)} images.")

    frames_path = Path(frames_dir)
    if not frames_path.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Gather all image files in frames_dir
    all_image_files = sorted(list(frames_path.glob("*.jpg")) + list(frames_path.glob("*.png")))
    if not all_image_files:
        raise ValueError(f"No image files found in {frames_dir}")

    print(f"Total image files found in directory: {len(all_image_files)}")

    output_path = Path(output_dir).resolve()

    # Create dataset directories
    train_img_dir = output_path / "images" / "train"
    val_img_dir = output_path / "images" / "val"
    train_lbl_dir = output_path / "labels" / "train"
    val_lbl_dir = output_path / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Perform deterministic train/val split
    random.seed(seed)
    annotated_files = [f for f in all_image_files if f.name in annotations]
    if not annotated_files:
        # Fallback to all images if CSV filenames don't strictly match
        annotated_files = all_image_files

    random.shuffle(annotated_files)
    num_val = int(len(annotated_files) * val_split)
    val_files = set(annotated_files[:num_val])
    train_files = set(annotated_files[num_val:])

    print(f"Train/Val Split: {len(train_files)} train images, {len(val_files)} val images.")

    # Image dimension cache to prevent repeated reading
    img_size_cache = {}

    converted_count = 0
    box_count = 0

    for img_file in all_image_files:
        fname = img_file.name
        is_val = img_file in val_files
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        # Copy image file to destination
        dst_img_path = target_img_dir / fname
        shutil.copy2(img_file, dst_img_path)

        # Process annotations for this image
        boxes = annotations.get(fname, [])
        lbl_file = target_lbl_dir / f"{img_file.stem}.txt"

        if boxes:
            if fname not in img_size_cache:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                img_size_cache[fname] = (img_h, img_w)
            else:
                img_h, img_w = img_size_cache[fname]

            yolo_lines = []
            for (x, y, w, h) in boxes:
                # Convert to normalized center_x, center_y, width, height
                cx = (x + w / 2.0) / img_w
                cy = (y + h / 2.0) / img_h
                nw = w / img_w
                nh = h / img_h

                # Clip to valid [0.0, 1.0] range
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                nw = min(max(nw, 0.0), 1.0)
                nh = min(max(nh, 0.0), 1.0)

                # Class 0: two_wheeler
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                box_count += 1

            with open(lbl_file, 'w', encoding='utf-8') as f_lbl:
                f_lbl.write("\n".join(yolo_lines) + "\n")
            converted_count += 1
        else:
            # Create an empty label file for negative samples
            with open(lbl_file, 'w', encoding='utf-8') as f_lbl:
                pass

    print(f"YOLO dataset conversion completed: {converted_count} labeled images ({box_count} total bounding boxes).")

    # Generate dataset.yaml
    yaml_path = output_path / "dataset.yaml"
    yaml_content = f"""# YOLOv8 Custom Two-Wheeler Dataset Configuration
path: '{output_path.as_posix()}'
train: images/train
val: images/val

names:
  0: two_wheeler
"""
    with open(yaml_path, 'w', encoding='utf-8') as f_yaml:
        f_yaml.write(yaml_content)

    print(f"Created dataset YAML at: {yaml_path}")
    return yaml_path


def train_yolov8(dataset_yaml, model_weights="yolov8s.pt", epochs=50, imgsz=1280, batch=8, device=None):
    """
    Fine-tunes a YOLOv8 model using Ultralytics API on the converted dataset.
    """
    print("=" * 70)
    print("Starting YOLOv8 Fine-Tuning Process")
    print(f"  - Base Model Weights : {model_weights}")
    print(f"  - Dataset Config     : {dataset_yaml}")
    print(f"  - Image Resolution   : {imgsz}")
    print(f"  - Epochs             : {epochs}")
    print(f"  - Batch Size         : {batch}")

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"  - Compute Device     : {device}")
    print("=" * 70)

    # Initialize YOLO model
    model = YOLO(model_weights)

    # Run fine-tuning
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name="yolov8_2w_custom",
        single_cls=True,  # Fine-tune specifically for single two-wheeler target class
        patience=15,
        save=True,
        verbose=True
    )

    print("\nFine-tuning completed!")
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Custom VIA Small Two-Wheeler Dataset")
    parser.add_argument("--csv-path", type=str, default="custom_dataset/via_project_22Aug2026_10h43m_csv.csv",
                        help="Path to VIA CSV annotation file")
    parser.add_argument("--frames-dir", type=str, default="custom_dataset/frames",
                        help="Path to directory containing input frame images")
    parser.add_argument("--output-dir", type=str, default="custom_dataset/yolo_format",
                        help="Directory to save converted YOLO dataset structure")
    parser.add_argument("--model", type=str, default="yolo/yolov8s.pt",
                        help="Pretrained YOLOv8 weights (e.g. yolov8s.pt or yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="YOLO input image resolution")
    parser.add_argument("--batch", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio (0.0 - 1.0)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only convert dataset and generate YAML without running training")

    args = parser.parse_args()

    # Resolve paths relative to working directory if needed
    base_dir = Path.cwd()
    csv_path = Path(args.csv_path) if Path(args.csv_path).is_absolute() else base_dir / args.csv_path
    frames_dir = Path(args.frames_dir) if Path(args.frames_dir).is_absolute() else base_dir / args.frames_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else base_dir / args.output_dir

    # Fallback to models/ or yolo/ directories if model_weights doesn't exist directly
    model_weights = args.model
    if not os.path.exists(model_weights):
        for candidate in ["models/best_yolo_custom.pt", "models/yolov8m.pt", "models/yolov8s.pt", "custom_model/best_yolo_custom.pt", "yolo/yolov8s.pt"]:
            if os.path.exists(candidate):
                model_weights = candidate
                break
        else:
            model_weights = "yolov8s.pt"


    # Step 1: Convert VIA CSV to YOLO dataset format
    yaml_path = prepare_yolo_dataset(
        csv_path=str(csv_path),
        frames_dir=str(frames_dir),
        output_dir=str(output_dir),
        val_split=args.val_split
    )

    # Step 2: Train YOLOv8 model if not prepare-only mode
    if not args.prepare_only:
        train_yolov8(
            dataset_yaml=yaml_path,
            model_weights=model_weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch
        )
    else:
        print("Prepare-only flag set. Dataset prepared successfully; skipping training step.")


if __name__ == "__main__":
    main()
