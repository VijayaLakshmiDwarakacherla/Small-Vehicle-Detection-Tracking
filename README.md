# Small Vehicle Detection & Tracking

## Detection and Custom Tracking for Small Two-Wheelers

The system is designed specifically for:

- **Object Detection:** YOLOv8m (Ultralytics), fine-tuned for small two-wheeler detection
- **Initial Detection Baseline:** YOLOv8s pretrained model
- **Small-Object Detection Strategy:** Global detection + high-zoom far-horizon tiling
- **Tracking:** Custom Adaptive Motion & Geometry (AMT) tracker
- **Motion Model:** Linear position/velocity prediction with velocity dampening
- **Data Association:** Custom geometry-aware cost matrix + Hungarian assignment

The main challenge is that the targets are very small relative to the 1280×720 frame and can move a significant distance between consecutive frames because the input video is only 10 FPS.

---

# 1. Input Video

The main test video used for this project is:

```text
13105476_720p_10fps_35s.mp4
```

Video properties:

```text
Resolution : 1280 × 720
Frame rate : 10 FPS
Duration   : approximately 35 seconds
Frames     : approximately 350
```

The footage was from pexel.com and prepared at 1280×720 and 10 FPS for this assignment.
link: https://www.pexels.com/video/busy-mumbai-road-traffic-scene-on-sunny-day-30608914/

The footage contains:

* Small motorcycles and scooters
* Distant two-wheelers
* Vehicles entering and leaving the scene
* Partial/temporary occlusions
* Stationary motorcycles/background vehicles

The data therefore provides the required non-trivial tracking conditions rather than being a synthetic or completely static scene.

---

# 2. Dataset Development

## 2.1 Initial Experiment with VisDrone MOT

My first approach was to use the public **VisDrone MOT dataset** for detector training.

VisDrone is useful for traffic and aerial/road-object detection, but during experimentation I found that it did not provide enough examples matching the specific target distribution in this task.

In particular:

* Many of the required motorcycles weren't extremely small.
* There were not enough training examples of the specific 6×8 to 20×20 pixel targets encountered in my footage.
* The camera viewpoint and traffic appearance differed from the target footage.
* The trained detector did not reliably detect the smallest distant motorcycles.

Therefore, VisDrone was useful as an initial training source and baseline, but it was not sufficient for the final small-object detection problem.

---

## 2.2 Custom Traffic Dataset

To address the domain mismatch, I collected my own traffic footage.

I manually annotated the relevant two-wheelers using the **VGG Image Annotator (VIA)**.

The annotations include very small distant objects, including targets down to approximately:

```text
6×8 pixels
8×10 pixels
10×10 pixels
15×15 pixels
```

The goal was to train/evaluate the detector on examples that actually resemble the target deployment footage.

---

## 2.3 Limitations of the Custom Dataset

Because the dataset was collected using a real camera rather than a controlled recording setup, it contains several disadvantages:

### Camera micro-jitter

Small camera movements cause the background to shift between frames. This can make stationary objects appear to move and makes motion-based filtering more difficult.

### Motion blur

Fast motorcycles and scooters can become blurred, especially at the low frame rate.

### Low contrast

Distant motorcycles can visually blend into the road surface and surrounding background.

---

# 3. Detector

## 3.1 Initial YOLOv8s Experiment

The first detector used was a pretrained **YOLOv8s** model.

YOLOv8s was attractive because it provides relatively fast inference on consumer hardware.

However, testing showed that standard full-frame inference was not sufficient for the smallest and farthest motorcycles.

The primary problem was not only model capacity, but also the number of pixels representing the target.

For example, a motorcycle occupying approximately 10×10 pixels in the original frame contains very little visual information. After the detector's resizing and feature extraction stages, the object can become extremely difficult to distinguish from road texture and compression noise.

Therefore, simply increasing confidence thresholds or using standard full-frame inference did not solve the problem.

---

# 4. YOLOv8m and Small-Object Strategy

I therefore moved to **YOLOv8m** and combined it with a high-zoom tiling strategy.

YOLOv8m provides greater model capacity than YOLOv8s and gave better detection results on the difficult distant targets in my experiments.

The trade-off is higher computational cost and slower inference.

---

## 4.1 Global Detection Pass

The full 1280×720 frame is processed to detect medium and large vehicles.

This pass is useful for motorcycles that are close enough to occupy a reasonable number of pixels.

---

## 4.2 High-Zoom Far-Horizon Pass

The most important small-object improvement is the far-horizon tiling strategy.

The upper/far-road region is divided into pixel tiles, each tile is resized before being passed to YOLOv8m.

Conceptually:

```text
Original Frame
1280 × 720
      │
      ▼
Far-Horizon Region
      │
      ▼
a x b px Tile
      │
      │ 4× digital enlargement
      ▼
4a × 4b px YOLO Input
      │
      ▼
Small motorcycle becomes significantly larger
inside the detector input
```

For example:

```text
Original motorcycle : ~10 pixels
After 4× tile resize : ~40 pixels
```

This does not create new information, but it allows the detector to operate on a much larger representation of the available pixels.

This significantly improves the opportunity to detect distant motorcycles compared with processing the entire 1280×720 frame at a conventional detector input size.

---

# 5. Detection Post-Processing

The detection stage applies several task-specific filters.

## Far-Horizon Handling

The far-horizon region is treated differently because the expected vehicle size and appearance are different from nearby vehicles.

## Rider/Motorcycle Handling

In the far-horizon region, person/rider detections can be associated with the corresponding two-wheeler region when the geometry is consistent.

The objective is to track the **vehicle/rider unit** rather than generating unrelated IDs for the rider and motorcycle.

## Geometric Filtering

Detections close to known non-road image margins can be rejected when their position and geometry are inconsistent with the traffic region.

This reduces false positives caused by roadside structures and background objects.

---

# 6. Custom AMT Tracker

The tracker is called:

**Adaptive Motion & Geometry (AMT) Tracker**

The tracker was implemented using Python, NumPy, and SciPy.

It does **not** use:

* SORT
* DeepSORT
* ByteTrack
* Norfair
* `cv2.legacy.Tracker*`
* Any equivalent off-the-shelf tracking implementation

---

# 7. Tracker Architecture

The tracking pipeline is:

```text
Detector
   │
   ▼
Current-frame detections
   │
   ▼
Motion prediction
   │
   ▼
Track-to-detection cost matrix
   │
   ▼
Distance/geometry gating
   │
   ▼
Hungarian assignment
   │
   ▼
Track lifecycle management
   │
   ├── Candidate
   ├── Confirmed
   ├── Lost
   └── Terminated
   │
   ▼
Stable track IDs
```

---

# 8. Motion Model

The tracker maintains a state containing position, object scale/geometry, and velocity.

Conceptually:

```text
x = [cx, cy, area, aspect_ratio, vx, vy, v_area]
```

where:

```text
cx, cy       = object center
area         = bounding-box area
aspect_ratio = bounding-box aspect ratio
vx, vy       = estimated image-plane velocity
v_area       = scale change velocity
```

The next position is predicted from the estimated velocity:

```text
x(t+1) = x(t) + v(t) × Δt
```

This is important because a two-wheeler can move significantly between two 10 FPS frames.

A simple IoU-only tracker is particularly vulnerable when the target moves 15–40 pixels between frames.

---

# 9. Velocity Dampening During Detector Dropouts

Detector predictions are not guaranteed to be available on every frame.

A motorcycle may temporarily disappear because of:

* Occlusion
* Motion blur
* Low contrast
* Detector uncertainty
* Compression artifacts
* Very small object size

Simply extrapolating the previous velocity indefinitely can cause a predicted track to drift far away from the true vehicle.

Therefore, AMT applies velocity dampening during consecutive missed detections:

```text
v_damped = v × 0.5^missed_frames
```

This means the prediction gradually becomes more conservative as the number of missed frames increases.

The purpose is to keep the predicted location sufficiently close to the last known position while still allowing short-term motion prediction.

---

# 10. Association Cost

Track-to-detection association uses a task-specific cost rather than only bounding-box IoU.

The cost combines:

1. Center-position distance
2. Object-scale consistency

The association cost is:

```text
Cij =
    0.70 × normalized_center_distance
  + 0.30 × scale_difference
```

More specifically:

```text
Cij =
0.70 × ||p_i - d_j|| / D_gate(y_i)
+
0.30 × |log(Area_j / Area_i)|
```

where:

```text
p_i = predicted track center
d_j = detection center
Area = bounding-box area
```

The distance gate is:

```text
D_gate(y) = max(35, 12 + 0.12 × y)
```

This allows the tracker to account for the fact that expected image-plane motion differs with image position/perspective.

Associations outside the allowed distance gate are rejected before assignment.

---

# 11. Hungarian Assignment

After computing the track-to-detection cost matrix and applying invalid-pair gating, valid associations are solved using the Hungarian assignment algorithm.

This provides a global one-to-one assignment between active tracks and current detections.

The Hungarian algorithm is used only as an optimization method for the custom association cost. The tracker itself, including the state representation, motion model, gating, lifecycle, and association design, is implemented specifically for this project.

---

# 12. Track Lifecycle

A major requirement of the task is:

> Only moving vehicles should produce tracks.

Therefore, a detection is not immediately promoted to a visible track.

The lifecycle is:

```text
             Detection
                 │
                 ▼
          ┌─────────────┐
          │  Candidate  │
          └──────┬──────┘
                 │
       Hits >= 2 AND
       displacement >= 2 px
                 │
                 ▼
          ┌─────────────┐
          │  Confirmed  │
          └──────┬──────┘
                 │
           Detector miss
                 │
                 ▼
             ┌───────┐
             │ Lost  │
             └───┬───┘
                 │
        ┌────────┴────────┐
        │                 │
   Re-detection       > 8 misses
        │                 │
        ▼                 ▼
   Confirmed          Terminated
```

---

# 13. Track Birth Policy

A new detection initially becomes a **Candidate**.

A candidate must satisfy both:

```text
Detection hits >= 2
```

and

```text
Cumulative spatial displacement >= 2 pixels
```

before becoming a Confirmed track.

This is intentionally designed to prevent stationary motorcycles from immediately becoming tracks.

For example:

```text
Parked motorcycle
      │
      ▼
Repeated detections
      │
      ▼
Very small displacement
      │
      ▼
Candidate
      │
      ▼
Not promoted
```

A moving motorcycle, on the other hand, should accumulate measurable displacement and become confirmed.

---

# 14. Track Death Policy

Confirmed tracks are allowed to survive temporary detector failures.

The current maximum number of consecutive missed frames is:

```text
8 frames
```

At 10 FPS this corresponds to approximately:

```text
0.8 seconds
```

of temporary detection loss.

If the vehicle is detected again within the allowed period, the existing track ID is recovered.

If the track remains missing for more than 8 consecutive frames, it is terminated.

This provides a compromise between:

* Maintaining identity during short occlusions/dropouts
* Avoiding tracks that persist indefinitely after a vehicle leaves the scene

---

# 15. Frame-by-Frame Debugging

A significant part of the development process was debugging the tracker frame by frame.

The pipeline can report information such as:

```text
ID
Previous State
Event
Matched Detection
Miss Count
Hit Count
Current State
Predicted Position
Number of Detections
```

Example conceptual output:

```text
Frame: 124

ID   PrevState   Event       Det   Miss   Hits   State       Prediction
---------------------------------------------------------------------------
3    Confirmed   MATCHED     2     0      31     Confirmed   (542.4, 186.2)
5    Lost        RECOVERED   4     0      18     Confirmed   (711.7, 203.8)
7    Candidate   NEW         6     0      1      Candidate   (824.1, 221.5)
9    Confirmed   MISSED      -     2      42     Lost        (392.8, 198.3)
```

This debugging information was particularly useful for identifying:

* Incorrect track associations
* Detector dropouts
* Excessive motion prediction
* Stationary-object false tracks
* Track recovery failures
* Incorrect track termination

---

# 16. Assumptions and Operational Definitions

## Small Vehicle

For this project, a small vehicle is defined as a vehicle whose bounding box is approximately:

```text
< 40 × 40 pixels
```

The most difficult targets are significantly smaller, with examples around:

```text
6 × 8 pixels
10 × 10 pixels
15 × 15 pixels
```

These are primarily located in the far-horizon region.

---

## Moving Vehicle

A vehicle is considered moving when its detections demonstrate measurable spatial displacement over multiple observations.

The current confirmation rule requires:

```text
At least 2 detection hits
AND
At least 2 pixels cumulative displacement
```

This threshold was selected because distant motorcycles can move only a few pixels between frames at 10 FPS.

---

## Stationary Vehicle

A stationary motorcycle that repeatedly appears at approximately the same location should remain a Candidate rather than producing a confirmed track.

This prevents parked motorcycles from being reported as moving vehicles.

---

# 17. Running the Pipeline

## Requirements

Recommended environment:

```text
Python 3.9+
Python 3.10 recommended
```

Hardware:

```text
Consumer CPU
or
Single NVIDIA consumer GPU
```

No multi-GPU or cluster processing is required.

---

# 18. Standard Pipeline Run

From the project directory:

```bash
python run_pipeline.py --input 13105476_720p_10fps.mp4 --output final_result.mp4 --detector p2_tiling --conf 0.05
```

The output video will contain:

* Detection bounding boxes
* Track IDs
* Confirmed moving vehicles
* Frame-by-frame tracking results

---

# 19. Performance Debugging

To inspect timing and tracking statistics:

```bash
python run_pipeline.py \
    --input 13105476_720p_10fps.mp4 \
    --output final_result.mp4 \
    --detector p2_tiling \
    --conf 0.05 \
    --debug-performance
```

---

# 20. Detector Diagnostic Mode

For detailed detector diagnostics:

```bash
python run_pipeline.py \
    --input 13105476_720p_10fps.mp4 \
    --output diagnostic_output.mp4 \
    --detector p2_tiling \
    --conf 0.05 \
    --debug-detector
```

This mode is useful for inspecting:

* Raw detections
* Candidate track births
* Detection dropouts
* NMS behavior
* Track lifecycle events

---

# 21. Throughput Measurement

The pipeline was measured on a single consumer-computer setup consisting of:

```text
CPU : Intel Core i7 / AMD Ryzen class 8-core CPU
GPU : NVIDIA RTX consumer GPU
RAM : 16 GB
```

Measured component timings:

| Pipeline Stage | Configuration                | Device |       Latency | Approx. Throughput |
| -------------- | ---------------------------- | -----: | ------------: | -----------------: |
| Detection      | YOLOv8m + 4 far tiles @ 1024 |    CPU | ~754 ms/frame |          ~1.33 FPS |
| Detection      | YOLOv8m + 4 far tiles @ 1024 |    GPU |  ~45 ms/frame |          ~22.2 FPS |
| AMT Tracker    | Custom NumPy/SciPy tracker   |    CPU | ~7.3 ms/frame |         ~136.8 FPS |

The conversion used is:

```text
FPS = 1000 / latency_ms
```

For example:

```text
238 ms/frame

1000 / 238 = 4.20 FPS
```

Therefore, **238 ms/frame corresponds to approximately 4.2 FPS**.

---

## Important Throughput Note

The detector is currently the main computational bottleneck.

The custom tracker itself is substantially faster than the detector and is not the limiting component.

The end-to-end throughput should be measured using the actual wall-clock runtime of the complete pipeline, including:

```text
Video decoding
+
Pre-processing
+
Global detection
+
Tiled detection
+
NMS/post-processing
+
Tracking
+
Video rendering
+
Video encoding
```

Therefore, component-level FPS numbers should not be interpreted as the final end-to-end FPS unless all pipeline stages are included in the measurement.

The reported measurements above are intentionally separated by component so that the computational bottleneck is clear.

---

# 22. Why the Current Detector Is a Trade-Off

YOLOv8m + high-zoom tiling improves small-object detection, but it increases inference cost.

The trade-off is:

```text
YOLOv8s
   │
   ├── Faster
   └── Weaker on very small/far targets
             │
             ▼
YOLOv8m
   │
   ├── Better small-object representation
   ├── Better detection recall in my experiments
   └── More computationally expensive
             │
             ▼
YOLOv8m + Far-Horizon Tiling
   │
   ├── Highest focus on tiny distant targets
   └── Highest computational cost
```

For this assignment, I prioritized recovering very small two-wheelers over maximum raw FPS because missing the target objects completely is more damaging to tracking quality than a moderate detector slowdown.

---

# 23. Current Failure Modes

The system does not claim perfect tracking. The main observed failure modes are:

## 1. Long Occlusion

If a motorcycle is completely occluded by a bus, truck, or another large vehicle for more than approximately 8 frames, the current track is terminated.

When the motorcycle reappears, a new ID may be assigned.

### Why

The tracker intentionally has a finite track lifetime after detector loss to avoid stale tracks remaining indefinitely.

### Improvement

A future version could use a stronger re-identification mechanism and longer-term motion/trajectory reasoning.

---

## 2. Dense Parallel or Crossing Motorcycles

When two motorcycles are extremely close, their detections may overlap.

For example:

```text
Motorcycle A █████
             █████ Motorcycle B
```

The detector's NMS can occasionally merge them into one detection.

This is primarily a detector limitation rather than an association-only problem.

### Improvement

A detector specifically trained for crowded small two-wheelers, combined with softer/cluster-aware suppression, could improve separation.

---

# 24. What I Would Improve With Another Week

If I had another week, I would focus primarily on improving the **small-object detector and camera-motion handling**, rather than replacing the existing tracker with an off-the-shelf tracker.

The current custom AMT tracker is lightweight, while the detector is the main computational and accuracy bottleneck.

---

# 29. Proposed Custom Ultra-Small Vehicle Detector

The most important improvement would be a dedicated detector designed specifically for 5–20 pixel two-wheelers.

Instead of repeatedly running a relatively large YOLO model over high-resolution tiles, I would investigate a lightweight two-stage architecture.

Proposed pipeline:

```text
1280 × 720 Frame
       │
       ▼
Cheap Motion / ROI Proposal
       │
       ▼
Candidate Small Regions
       │
       ▼
Overlapping Local Tiles
       │
       ▼
Lightweight CNN
       │
       ▼
Tiny Two-Wheeler Detection
       │
       ▼
AMT Tracker
```

The goal would be to preserve the high recall of the current tiled detector while significantly reducing inference cost.

---

# 25. Sparse ROI Proposal

A first-stage lightweight process could identify regions where a moving vehicle is
likely to exist using motion, appearance, and temporal consistency.

For example:

```text
P(candidate) =
    α × motion_score
  + β × appearance_score
  + γ × temporal_consistency

---

# 31. Overlapping Small Tiles

I would investigate overlapping local patches, for example:

```text
32 × 32 input tile
        │
        ▼
14 × 14 valid detection region
```

Overlapping tiles are useful because a tiny motorcycle located at the boundary of one tile can otherwise be split between neighboring tiles.

The overlapping strategy would reduce boundary-related detection failures.

---

# 26. Lightweight Backbone

For the dedicated detector, I would investigate:

* Depthwise-separable convolutions
* Residual blocks
* Lightweight feature pyramids
* Small detection heads
* Quantization
* ONNX/TensorRT deployment where appropriate

The goal would be to reduce the computational cost compared with YOLOv8m while preserving the ability to recognize very small objects.

---

# 27. Temporal Information for Detection

Another improvement would be to exploit the fact that this is video rather than independent images.

A very small motorcycle that is ambiguous in one frame can become obvious when several consecutive frames are considered.

I would investigate causal temporal feature memory:

```text
F(t-4)
   │
F(t-3)
   │
F(t-2)
   │
F(t-1)
   │
F(t)
   │
▼
Temporal feature aggregation
   │
   ▼
Small-object detection
```

A lightweight recurrent or exponential feature memory could improve detection stability without requiring future frames.

---

# 28. Synthetic Small-Object Augmentation

The custom dataset could also be expanded using realistic degradation.

Motorcycle crops could be resized to:

```text
5 px
8 px
12 px
18 px
25 px
```

and then augmented with:

* Motion blur
* Gaussian blur
* JPEG compression
* Noise
* Contrast changes
* Haze
* Brightness changes
* Downsampling artifacts

This would specifically train the detector for the conditions under which the smallest targets become difficult.

---

# 20. Hard-Negative Mining

I would also collect difficult non-vehicle patches such as:

```text
Road markings
Shadows
Guardrails
Poles
Signs
Reflections
Road texture
Compression artifacts
```

These would be used as hard negatives so the detector learns that not every tiny high-contrast blob is a motorcycle.

---

# 30. Camera Motion Compensation

Before object-level tracking, I would estimate global camera motion using stable background features.

Conceptually:

```text
Frame t-1
   │
   ▼
Background feature matching
   │
   ▼
Global camera transform
   │
   ▼
Compensate frame t
   │
   ▼
Object motion estimation
```

This would reduce false motion caused by camera vibration.

It would also make the moving-versus-stationary decision more reliable.

---

# 31. Improved Re-Identification

For long occlusions, I would add a lightweight appearance representation.

The tracker could combine:

```text
Motion
+
Geometry
+
Appearance
```

For example:

```text
Association Cost =
    motion cost
  + geometry cost
  + appearance similarity
```

The appearance representation would need to be designed carefully because extremely small motorcycles often contain too few pixels for reliable appearance matching.

For this reason, appearance would be used as a supporting signal rather than the primary association mechanism.

---

# 32. Improved Track Lifecycle

The current lifecycle uses a fixed 8-frame missed-detection threshold.

A future version could make this adaptive.

For example:

```text
High-confidence track
        │
        ▼
Longer allowed miss period

Low-confidence/new track
        │
        ▼
Shorter allowed miss period
```

The maximum allowed gap could also depend on:

* Estimated velocity
* Object location
* Direction of travel
* Scene geometry
* Historical detector confidence
* Previous track stability

This would reduce unnecessary track termination while avoiding stale tracks.

---

# 33. Summary

The final system consists of:

```text
1280 × 720 @ 10 FPS Video
            │
            ▼
     Global YOLOv8m Pass
            │
            +
            │
            ▼
   Far-Horizon Zoom Tiling
            │
            ▼
     Detection Filtering
            │
            ▼
  Custom AMT Motion Prediction
            │
            ▼
 Perspective-Aware Association
            │
            ▼
    Hungarian Assignment
            │
            ▼
 Candidate → Confirmed
            │
            ▼
       Lost / Recovery
            │
            ▼
     Track Termination
            │
            ▼
 Stable Moving-Vehicle IDs
```

The key design decisions were made specifically for the characteristics of the provided traffic footage:

* **YOLOv8m** was selected after YOLOv8s showed insufficient performance on the smallest/farthest targets.
* **Far-horizon tiling and digital enlargement** were introduced to improve the effective representation of tiny vehicles.
* A **custom AMT tracker** was implemented rather than using SORT, DeepSORT, ByteTrack, Norfair, or another equivalent tracking library.
* **Motion prediction** handles the large frame-to-frame displacement caused by 10 FPS input.
* **Velocity dampening** prevents excessive prediction drift during detector dropouts.
* **Geometry-aware association** reduces incorrect matches between nearby vehicles.
* **Candidate/Confirmed/Lost/Terminated lifecycle states** prevent stationary motorcycles from immediately becoming tracks and allow temporary detector failures.
* **Frame-level debugging** was used to inspect previous state, current state, predictions, detections, hit counts, miss counts, and recovery behavior.
* The main current bottleneck is **small-object detection**, especially the computational cost of high-resolution tiled YOLOv8m inference.

The next major improvement would therefore be a **lightweight detector specifically optimized for sub-20-pixel two-wheelers**, combined with camera-motion compensation and stronger long-term track recovery.

---
> **Note:** This repository contains only the final working implementation used for the submission. Intermediate experiments, unsuccessful model versions, and development iterations are not included. The README documents the major experiments and design decisions that led to the final system.
