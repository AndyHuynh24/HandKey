# HandFlow — Comprehensive Project & Edge AI Summary

> Prepared for lab discussion. Covers project architecture, code internals, ML concepts, edge AI principles, and engineering decisions.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Edge AI — Concepts & Relevance](#2-edge-ai--concepts--relevance)
3. [System Architecture — End-to-End Pipeline](#3-system-architecture--end-to-end-pipeline)
4. [Hand Tracking with MediaPipe](#4-hand-tracking-with-mediapipe)
5. [Handedness Tracking — Solving MediaPipe's Label Instability](#5-handedness-tracking--solving-mediapipes-label-instability)
6. [Feature Engineering — 96 Dimensions](#6-feature-engineering--96-dimensions)
7. [Temporal Convolutional Network (TCN)](#7-temporal-convolutional-network-tcn)
8. [Alternative Model Architectures](#8-alternative-model-architectures)
9. [TFLite — On-Device Inference](#9-tflite--on-device-inference)
10. [Prediction Smoothing & Stability](#10-prediction-smoothing--stability)
11. [ArUco Markers & Homography — Virtual Touchscreen](#11-aruco-markers--homography--virtual-touchscreen)
12. [Paper Macro Pad — Physical Interface from CV](#12-paper-macro-pad--physical-interface-from-cv)
13. [Screen Overlay Macro Pad](#13-screen-overlay-macro-pad)
14. [Action Execution System](#14-action-execution-system)
15. [Mouse Control — Smooth Cursor Tracking](#15-mouse-control--smooth-cursor-tracking)
16. [Signal Processing — OneEuro Filter](#16-signal-processing--oneeuro-filter)
17. [Training Pipeline](#17-training-pipeline)
18. [Data Augmentation — Geometric Perturbation](#18-data-augmentation--geometric-perturbation)
19. [Data Pipeline & Caching](#19-data-pipeline--caching)
20. [Evaluation & Visualization](#20-evaluation--visualization)
21. [Application Layer — GUI & Real-Time Loop](#21-application-layer--gui--real-time-loop)
22. [Performance Characteristics](#22-performance-characteristics)
23. [Key Engineering Decisions & Trade-offs](#23-key-engineering-decisions--trade-offs)
24. [Glossary of Key Terms](#24-glossary-of-key-terms)

---

## 1. Project Overview

**HandFlow** is an end-to-end, real-time hand gesture recognition system that turns a standard webcam into a touchless human-computer interface. It runs entirely on CPU at 25-30 FPS.

### What it does

- **Free-space gesture control** — 7 hand gestures (swipe, click, scroll, zoom, pinch) that trigger OS-level actions (keyboard shortcuts, app launches, mouse control)
- **Virtual touchscreen** — Uses ArUco markers at screen corners to transform any non-touch display into a touch surface via homography
- **12-button on-screen macro pad** — Floating overlay with ArUco markers detected by the camera
- **24-button paper macro pad** — A printed A4 sheet that folds into a triangular prism with 3 faces x 8 buttons, tracked purely by camera

### Tech stack

| Component | Technology |
|-----------|------------|
| Hand tracking | MediaPipe Hands (21 3D landmarks) |
| Gesture model | Temporal Convolutional Network (TCN) |
| Inference runtime | TensorFlow Lite (907 KB model) |
| Computer vision | OpenCV (ArUco marker detection, homography) |
| GUI | CustomTkinter |
| Config | Pydantic + YAML |
| Experiment tracking | Weights & Biases + TensorBoard |
| Platform integration | macOS Quartz (native mouse), PyAutoGUI (cross-platform) |

### 11 Gesture Classes

| Gesture | Description | Use Case |
|---------|-------------|----------|
| `none` | Idle / no gesture | Baseline class |
| `horizontal_swipe` | Lateral hand movement | Navigation, switching |
| `swipeup` | Upward vertical motion | Scroll, open app |
| `thumb_index_swipe` | Thumb+index directional swipe | Fine control |
| `thumb_middle_swipe` | Thumb+middle directional swipe | Macro trigger |
| `5_fingers_close` | All fingers pinch together | Zoom, close |
| `pointyclick` | Index finger point and click | Selection |
| `middleclick` | Middle finger click | Context action |
| `touch_hover` | Finger hovering near surface | Cursor movement |
| `touch_hold` | Sustained contact | Drag operation |
| `touch` | Brief contact | Click |

---

## 2. Edge AI — Concepts & Relevance

### What is Edge AI?

Edge AI refers to running artificial intelligence algorithms **locally on a device** (the "edge" of the network) rather than in the cloud. The "edge" can be a phone, laptop, microcontroller, camera, or any device close to where data is generated.

### Why Edge AI matters

| Factor | Cloud AI | Edge AI |
|--------|----------|---------|
| **Latency** | 50-500ms round-trip | <10ms local inference |
| **Privacy** | Data leaves device | Data stays on device |
| **Connectivity** | Requires internet | Works offline |
| **Cost** | Per-request API cost | One-time model deployment |
| **Bandwidth** | Streams raw data | Processes locally |
| **Reliability** | Depends on network | Always available |

### Edge AI in HandFlow

HandFlow is a textbook example of edge AI:
- **All computation is local** — camera frames never leave the machine
- **Real-time constraint** — gesture recognition must happen in <50ms per frame (20 FPS target)
- **Resource-constrained** — must run on a laptop CPU (no GPU required)
- **Privacy-preserving** — no hand/face data is transmitted anywhere
- **Offline-capable** — works without internet

### Key Edge AI Techniques Used

#### Model Compression — TFLite Quantization

The gesture model is compressed from **2.7 MB (Keras)** to **907 KB (TFLite)** — a 66% reduction. TFLite (TensorFlow Lite) is Google's framework for on-device ML inference.

**How quantization works:**
- Full TensorFlow models use 32-bit floating point weights
- TFLite can quantize to 16-bit float, 8-bit integer, or dynamic range
- Smaller model = faster loading, less memory, often faster inference
- Trade-off: small accuracy loss (usually <1% for well-designed models)

#### Lightweight Architecture Design

The TCN was specifically chosen over heavier alternatives because:
- **No recurrence** — Unlike LSTMs/GRUs, TCNs use only convolutions (parallelizable, no sequential dependency)
- **Fixed receptive field** — Dilated convolutions cover the full 12-frame window with just 3 layers
- **Small parameter count** — ~88K parameters total
- **Predictable latency** — convolution operations have consistent timing (no variable-length state updates)

#### Efficient Feature Extraction Pipeline

Instead of feeding raw pixels (computationally expensive), HandFlow uses a multi-stage feature reduction:
```
Camera frame (1280x720x3 = 2.7M values)
    ↓ MediaPipe (runs on 320x180)
21 landmarks x 4 values = 84 values per hand
    ↓ Feature Engineering
96 engineered features per frame
    ↓ Sliding Window
12 frames x 96 features = 1,152 values → Model input
```

This is a **2,400x data reduction** from raw pixels to model input.

#### Frame Rate Management

Edge devices have variable compute speed. HandFlow handles this with:
- **Adaptive frame sampling** — Accumulator-based algorithm that maintains consistent 20 FPS input to the model regardless of actual camera frame rate
- **FPS-normalized velocities** — Feature values are scaled by `reference_dt / actual_dt` so the model behaves identically on a 60 FPS desktop and a 15 FPS laptop
- **Linear interpolation** — On slow devices, missing frames are interpolated to fill the window

### Edge AI Deployment Spectrum

```
Cloud GPU ←————————————————————————————→ Microcontroller
   │                                          │
   │  HandFlow sits here:                     │
   │  ┌─────────────────┐                     │
   │  │ Laptop CPU      │                     │
   │  │ ~907KB model    │                     │
   │  │ ~25-30 FPS      │                     │
   │  │ Real-time       │                     │
   │  └─────────────────┘                     │
   │                                          │
   Server GPU    Laptop GPU    Mobile    IoT/MCU
   (PyTorch)     (CUDA)        (CoreML)  (TinyML)
```

### Edge AI vs Traditional Approaches

| Approach | How it would work | Why HandFlow chose edge |
|----------|-------------------|------------------------|
| Cloud API | Stream video to server, get predictions back | Too slow (100ms+ latency), privacy concerns, requires internet |
| GPU inference | Run full PyTorch model on GPU | Not all laptops have discrete GPUs, overkill for this task |
| **Edge/CPU inference** | **TFLite model on CPU with engineered features** | **Fast, portable, private, works everywhere** |
| Microcontroller | Port to Arduino/ESP32 | Insufficient compute for MediaPipe + TCN |

---

## 3. System Architecture — End-to-End Pipeline

### Real-Time Inference Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                        CAMERA INPUT                                │
│                     (1280x720 @ 30 FPS)                           │
└────────────────────────┬───────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   Frame Preprocessing │
              │   - Horizontal flip   │
              │   - Vertical flip     │
              │   - Resize to 320x180 │
              └──────────┬───────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌────────────────┐ ┌──────────┐ ┌──────────────────┐
│  MediaPipe     │ │  ArUco   │ │  MacroPad        │
│  Hands         │ │  Screen  │ │  Detection       │
│  (every frame) │ │  Detect  │ │  (every 3 frames)│
│                │ │  (3 frm) │ │                  │
│  21 landmarks  │ │  4 corner│ │  8 markers       │
│  per hand      │ │  markers │ │  → button grid   │
└───────┬────────┘ └────┬─────┘ └────────┬─────────┘
        │               │               │
        ▼               │               │
┌────────────────┐      │               │
│  HandTracker   │      │               │
│  - Centroid    │      │               │
│    matching    │      │               │
│  - Majority    │      │               │
│    vote L/R    │      │               │
│  - Phantom     │      │               │
│    filter      │      │               │
└───────┬────────┘      │               │
        │               │               │
        ▼               │               │
┌────────────────┐      │               │
│  OneEuro       │      │               │
│  Filter        │      │               │
│  (finger tips) │      │               │
└───────┬────────┘      │               │
        │               │               │
        ▼               │               │
┌────────────────┐      │               │
│  Feature       │      │               │
│  Engineer      │      │               │
│  84 → 96 dims  │      │               │
│  per frame     │      │               │
└───────┬────────┘      │               │
        │               │               │
        ▼               │               │
┌────────────────┐      │               │
│  Sliding Window│      │               │
│  12 frames     │      │               │
│  (12, 96)      │      │               │
└───────┬────────┘      │               │
        │               │               │
        ▼               │               │
┌────────────────┐      │               │
│  TFLite TCN    │      │               │
│  907 KB model  │      │               │
│  11 classes    │      │               │
└───────┬────────┘      │               │
        │               │               │
        ▼               │               │
┌────────────────┐      │               │
│  Majority Vote │      │               │
│  + Confidence  │      │               │
│    Threshold   │      │               │
└───────┬────────┘      │               │
        │               │               │
        ▼               ▼               ▼
┌──────────────────────────────────────────────────┐
│              ACTION EXECUTOR                      │
│  Gesture → Settings Lookup → OS Action           │
│  Touch → Homography → Screen Coords → Click      │
│  MacroPad → Button Index → Action Sequence       │
│                                                  │
│  Actions: keyboard shortcuts, mouse events,      │
│           app launches, text paste, media keys    │
└──────────────────────────────────────────────────┘
```

### Threading Model

```
┌─────────────────────────────────────────┐
│  Main Thread (UI)                       │
│  - CustomTkinter event loop             │
│  - Settings GUI (tabbed interface)      │
│  - Frame display (PIL → CTkLabel)       │
└────────────────────┬────────────────────┘
                     │ _latest_frame
┌────────────────────▼────────────────────┐
│  Capture Thread (Background)            │
│  - Camera read                          │
│  - MediaPipe + Feature Engineering      │
│  - TFLite inference                     │
│  - ArUco detection                      │
│  - MacroPad detection                   │
│  - Action dispatch                      │
└────────────────────┬────────────────────┘
                     │ (optional)
┌────────────────────▼────────────────────┐
│  Action Sequence Thread (Daemon)        │
│  - Multi-step action chains             │
│  - Delayed keyboard/mouse sequences     │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│  Video Writer Thread (Optional)         │
│  - Queue-based frame recording          │
│  - Constant 30 FPS with frame dup/drop  │
└─────────────────────────────────────────┘
```

---

## 4. Hand Tracking with MediaPipe

### What is MediaPipe?

MediaPipe is Google's open-source framework for building perception pipelines. **MediaPipe Hands** specifically detects and tracks hand landmarks in real-time from RGB images.

### How it works

1. **Palm detection** — A lightweight SSD (Single Shot Detector) locates hands in the frame
2. **Landmark regression** — A second model predicts 21 3D landmarks (keypoints) per detected hand
3. **Tracking** — Between frames, the system uses the previous landmark positions to seed the next detection (faster than re-detecting from scratch)

### The 21 Landmarks

```
        MIDDLE_FINGER_TIP (12)
              │
        MIDDLE_FINGER_DIP (11)
              │
        MIDDLE_FINGER_PIP (10)
              │
        MIDDLE_FINGER_MCP (9)
              │
THUMB_TIP(4)──┼──INDEX_FINGER_TIP(8)──RING_FINGER_TIP(16)──PINKY_TIP(20)
   │          │          │                    │                   │
THUMB_IP(3)   │   INDEX_DIP(7)         RING_DIP(15)        PINKY_DIP(19)
   │          │          │                    │                   │
THUMB_MCP(2)  │   INDEX_PIP(6)         RING_PIP(14)        PINKY_PIP(18)
   │          │          │                    │                   │
THUMB_CMC(1)  │   INDEX_MCP(5)         RING_MCP(13)        PINKY_MCP(17)
              │
          WRIST (0)
```

Each landmark has 4 values: `x, y, z, visibility`
- `x, y` — Normalized screen coordinates [0, 1]
- `z` — Depth relative to wrist (negative = closer to camera)
- `visibility` — Confidence that the landmark is visible

### HandFlow's MediaPipe Configuration

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `min_detection_confidence` | 0.5 | 50% confidence needed to detect a new hand |
| `min_tracking_confidence` | 0.3 | 30% confidence to maintain tracking (lower = more persistent) |
| `max_num_hands` | 2 | Track up to 2 hands simultaneously |
| `model_complexity` | 1 | Balanced speed/accuracy (0=fast, 1=balanced, 2=accurate) |

### Why not use MediaPipe's built-in gesture recognition?

MediaPipe has a gesture recognizer, but HandFlow builds its own because:
- MediaPipe gestures are **static** (pose-based) — HandFlow needs **temporal** gestures (motion-based)
- Custom gestures like `thumb_index_swipe` aren't in MediaPipe's vocabulary
- The TCN model processes **sequences** of 12 frames, capturing motion patterns that single-frame classifiers cannot

---

## 5. Handedness Tracking — Solving MediaPipe's Label Instability

### The Problem

MediaPipe's handedness classification (`Left`/`Right`) is unreliable:
- Labels flip randomly between frames
- When hands cross, labels become chaotic
- Confidence scores are inconsistent

This is a real engineering problem because each hand has **independent gesture sequences and action mappings**.

### The Solution: `HandTracker`

HandFlow implements a custom handedness stabilization system using centroid-based spatial tracking and majority voting.

#### Algorithm

```
For each frame:
  1. EXTRACT — Get all hand detections (centroids, landmarks, MediaPipe labels)

  2. FILTER PHANTOMS — If two detections are < 0.12 apart in normalized
     coordinates, keep only the higher-confidence one
     (prevents double-detection of the same hand)

  3. MATCH — Greedy nearest-neighbor matching:
     - For each tracked hand slot, find the closest new detection
     - If distance < 0.25 (normalized), it's the same hand
     - Unmatched detections create new hand slots

  4. VOTE — Majority voting over 15-frame history:
     - Each frame's MediaPipe label is a "vote"
     - Need 60% agreement to assign a stable label
     - If both hands get the same label, resolve by screen position
       (in mirror mode: left side of screen = left hand)

  5. CLEANUP — Remove hands missing for > 10 consecutive frames

  6. OUTPUT — Return (right_keypoints, left_keypoints)
     - Left hand keypoints are X-flipped so the model sees
       both hands in the same orientation (symmetry)
```

### Why this works

- **Centroid tracking** is more stable than label-based tracking because hand positions change smoothly
- **Majority voting** absorbs momentary label flips
- **Phantom filtering** handles MediaPipe's tendency to duplicate detections
- **X-flipping** for left hand means you only need **one model** for both hands

---

## 6. Feature Engineering — 96 Dimensions

### Why engineer features?

Raw MediaPipe output is 84 values (21 landmarks x 4). These raw coordinates are:
- **Translation-dependent** — same gesture at different screen positions looks different
- **Missing motion information** — no velocity, acceleration
- **Missing structural information** — no joint angles, finger distances

Feature engineering transforms raw coordinates into a representation that is **invariant to irrelevant factors** and **rich in gesture-discriminative information**.

### Feature Breakdown

#### Group 1: Relative Positions (63 dims, indices 0-62)

```python
# Subtract wrist position from all landmarks → translation invariance
for each landmark in 21 landmarks:
    relative_xyz = landmark_xyz - wrist_xyz  # 3 values each
```

**Why:** The same gesture performed at the top-left or bottom-right of the frame should be recognized identically. Wrist-relative coordinates achieve this.

#### Group 2: Inter-Finger Distances (5 dims, indices 63-67)

```
thumb_tip ↔ index_tip     (pinch detection)
thumb_tip ↔ middle_tip    (middle finger gestures)
thumb_tip ↔ ring_tip      (hand openness)
thumb_tip ↔ pinky_tip     (hand openness)
thumb_tip ↔ index_PIP     (thumb position relative to hand)
```

**Why:** These distances directly encode hand pose — open hand vs. closed fist vs. pinch.

#### Group 3: Absolute Positions (9 dims, indices 68-76)

```
thumb_tip xyz, index_MCP xyz, index_tip xyz
```

**Why:** Needed for screen-space operations (cursor mapping, touch detection). Translation invariance is intentionally NOT applied here.

#### Group 4: Velocities (9 dims, indices 77-85)

```python
velocity = (current_position - previous_position) * (reference_fps / actual_fps)
```

Applied to: index MCP (3), index tip (3), thumb tip (3)

**Why:** Swipe and motion gestures are defined by movement, not position. FPS normalization ensures a swipe at 15 FPS looks the same as at 30 FPS to the model.

**FPS normalization formula:**
```
time_scale = reference_dt / actual_dt
           = (1/20) / (actual_delta_time)
velocity_normalized = raw_velocity * time_scale
```

#### Group 5: Finger Bending Angles (5 dims, indices 83-87)

```python
# Angle at PIP joint (or IP for thumb) using 3-point arccos
for each finger:
    vec1 = MCP - PIP  # bone before joint
    vec2 = TIP - PIP  # bone after joint
    angle = arccos(dot(vec1, vec2) / (|vec1| * |vec2|))
```

**Why:** Distinguishes bent fingers from straight fingers — critical for gestures like `pointyclick` (index extended, others bent).

#### Group 6: Pinch Dynamics (3 dims, indices 91-93)

```
- Pinch aperture velocity: d(thumb-index distance)/dt
- Pinch aperture acceleration: d²(thumb-index distance)/dt²
- Thumb-index Z difference: depth gap between thumb and index tips
```

**Why:** Pinch dynamics are the key differentiator between `touch`, `touch_hover`, and `touch_hold`. The velocity tells you if the fingers are closing (about to touch) or stable (holding).

#### Group 7: Thumb Posture (2 dims, indices 94-95)

```
- Thumb abduction angle: angle between thumb direction and palm direction
- Thumb-to-wrist distance: how far the thumb extends from the hand
```

**Why:** Thumb position varies significantly across gestures and provides global hand posture information.

---

## 7. Temporal Convolutional Network (TCN)

### Why TCN over RNN/LSTM?

| Property | TCN | LSTM/GRU |
|----------|-----|----------|
| Parallelism | Fully parallelizable (all convolutions independent) | Sequential (each timestep depends on previous) |
| Receptive field | Controlled by dilation, covers exact window needed | Theoretically infinite but practically limited by vanishing gradients |
| Memory | Fixed, predictable | Hidden state can grow/shrink |
| Inference speed | Very fast on CPU (cache-friendly convolutions) | Slower (sequential matrix multiplications) |
| Training stability | BatchNorm + residual connections | Gradient clipping needed, more hyperparameter sensitive |

### Architecture in Detail

```
Input: (batch, 12, 96)
  │
  ▼
1x1 Convolution: 96 → 128 channels
  │  Purpose: Project features to a higher-dimensional space
  │  (like an embedding layer for temporal features)
  │
  ▼
Residual Block (dilation=1, receptive field=3 frames):
  ┌─────────────────────────────────────────────┐
  │ Conv1D(128, kernel=3, dilation=1, causal)   │
  │ BatchNorm → ReLU                            │
  │ Conv1D(128, kernel=1) [pointwise]           │
  │ BatchNorm → Dropout(0.1)                    │
  │              +                              │
  │         [input]────────→ residual add       │
  │              → ReLU                         │
  └─────────────────────────────────────────────┘
  │
  ▼
Residual Block (dilation=2, receptive field=5 additional frames):
  │  Same structure, but Conv1D dilation=2
  │  Each filter "sees" every 2nd frame → wider temporal context
  │
  ▼
Residual Block (dilation=4, receptive field=5 more frames):
  │  Dilation=4: sees every 4th frame
  │  Total receptive field: 3 + 4 + 4 = 13 frames (covers full 12-frame input)
  │
  ▼
┌─────────────────┐  ┌─────────────────┐
│ Global AvgPool  │  │ Global MaxPool  │
│ (128-dim)       │  │ (128-dim)       │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └──────┬─────────────┘
                │ Concatenate
                ▼
          (256-dim vector)
                │
                ▼
         Dense(64, ReLU)
         Dropout(0.1)
                │
                ▼
         Dense(11, Softmax)
                │
                ▼
         11 gesture probabilities
```

### Key Concepts Explained

#### Dilated Causal Convolutions

Standard 1D convolution with kernel size 3 looks at 3 consecutive time steps. **Dilation** skips time steps:

```
Dilation=1: looks at t, t-1, t-2          (3 frames)
Dilation=2: looks at t, t-2, t-4          (spans 5 frames)
Dilation=4: looks at t, t-4, t-8          (spans 9 frames)
```

By stacking dilations [1, 2, 4], the network's **receptive field** grows exponentially with depth while using fewer parameters than widening the kernel.

**Causal** means the convolution only looks backward in time (no future frames), which is essential for real-time inference.

#### Residual Connections

```
output = ReLU(Conv_block(input) + input)
```

The `+ input` skip connection means the network only needs to learn the **residual** (difference) from identity. Benefits:
- Prevents vanishing gradients in deeper networks
- Allows information to flow unchanged through layers
- Makes it easier to learn identity mappings when the block adds no useful transformation

#### Combined Pooling

Using **both** Global Average Pooling and Global Max Pooling:
- **Average** captures the overall signal strength across time
- **Max** captures the peak activations (most distinctive moments)
- Concatenating both gives the classifier richer information than either alone

### Model Size

| Format | Size | Notes |
|--------|------|-------|
| Keras (.keras) | 2.7 MB | Full 32-bit float weights |
| TFLite (.tflite) | 907 KB | Quantized, 66% smaller |

---

## 8. Alternative Model Architectures

HandFlow supports 5 architectures, all swappable via `config.yaml`:

### LSTM (Long Short-Term Memory)

```
Input → LSTM(128, return_sequences=True) → LSTM(64) → Dense(64) → Softmax
```

- **Gated recurrence**: Input gate, forget gate, output gate control information flow
- Processes frames **sequentially** (slower inference than TCN)
- Good at capturing long-range dependencies
- More parameters than TCN for same capacity

### GRU (Gated Recurrent Unit)

```
Input → GRU(128, return_sequences=True) → GRU(64) → Dense(64) → Softmax
```

- Simplified LSTM with only 2 gates (reset, update)
- Fewer parameters than LSTM
- Similar performance for short sequences like 12 frames

### 1D-CNN

```
Input → Conv1D(64) → BN → MaxPool → Conv1D(128) → BN → MaxPool → Conv1D(128) → BN → GlobalAvgPool → Dense(64) → Softmax
```

- Simple stacked convolutions with pooling
- Fast but limited temporal context (MaxPool reduces sequence length)
- No dilations or residual connections

### Transformer

```
Input → Positional Encoding → 2x [MultiHeadAttention(4 heads) → FFN(128)] → GlobalAvgPool → Dense(64) → Softmax
```

- Self-attention mechanism can capture arbitrary temporal relationships
- Overkill for 12-frame sequences (attention is most useful for long sequences)
- Higher computational cost due to quadratic attention complexity

### Why TCN Won

For this specific task (12-frame gesture classification on CPU):
- TCN has the **best speed/accuracy trade-off**
- Receptive field exactly covers the input window
- BatchNorm + residual connections make it stable to train
- Convolutions are highly optimized on modern CPUs

---

## 9. TFLite — On-Device Inference

### What is TFLite?

TensorFlow Lite is Google's framework for running ML models on edge devices. It provides:
- A **converter** that transforms TensorFlow/Keras models into an optimized flatbuffer format
- A lightweight **interpreter** that runs the model without the full TensorFlow runtime
- Support for **quantization** (reducing weight precision)
- **Hardware delegation** (GPU, NNAPI, CoreML, etc.)

### Conversion Process

```python
# 1. Save Keras model as SavedModel (TF format)
model.save(temp_dir)

# 2. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,    # Standard TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS       # Fallback to full TF ops (needed for some layers)
]
tflite_model = converter.convert()

# 3. Write to file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

### Runtime Inference

```python
# Load once at startup
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Each frame:
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_data)  # (1, 12, 96)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])     # (1, 11)
```

### Size Reduction

```
Keras model:  2.7 MB (32-bit float, full TF graph)
TFLite model: 907 KB (optimized flatbuffer, quantized)
Reduction:    66% smaller
```

---

## 10. Prediction Smoothing & Stability

Raw model predictions are noisy. HandFlow uses multiple smoothing techniques:

### Majority Vote Smoothing

```python
# Keep a buffer of the last N predictions (stability_window = 4-5)
prediction_buffer = deque(maxlen=stability_window)

# Each frame: add the argmax prediction to the buffer
prediction_buffer.append(predicted_class)

# The output gesture is the most common prediction in the buffer
majority_class = Counter(prediction_buffer).most_common(1)[0][0]

# Only accept if majority matches current prediction AND confidence > threshold
if majority_class == predicted_class and confidence > 0.5:
    return gesture
```

**Why:** Prevents single-frame misclassifications from triggering actions. A gesture must be consistently predicted for ~4-5 frames to be accepted.

### Confidence Thresholding

```
Confidence threshold: 0.5 (50%)
```

If the model's softmax probability for the top class is below 0.5, the prediction is discarded. This prevents ambiguous gestures from triggering actions.

### Cooldown System

```
Cooldown: 12 frames (~600ms at 20 FPS)
```

After a gesture triggers an action, no new gestures can trigger for 12 frames. This prevents:
- Repeated firing of the same gesture
- Adjacent gestures triggering in rapid succession

### Touch-Specific: Finger Tip Lookback

```python
# Cache finger tip positions for 8 frames
tip_cache = deque(maxlen=8)

# For clicks, use the position from 6 frames ago, not the current frame
click_position = tip_cache[-6]  # 6 frames in the past
```

**Why:** At the exact moment of a touch, the finger is still moving and the tip position is jittery. The position from 6 frames ago (before the touch motion began) is more stable.

---

## 11. ArUco Markers & Homography — Virtual Touchscreen

### What are ArUco Markers?

ArUco markers are binary square fiducial markers used in computer vision for pose estimation and tracking. Each marker has:
- A unique ID encoded in a binary grid pattern
- Known physical size
- Fast detection algorithm in OpenCV

HandFlow uses `DICT_4X4_50` — a dictionary of 50 markers with 4x4 internal grids.

### Virtual Touchscreen — How It Works

```
Physical Setup:
┌──────────────────────────────┐
│  [Marker 0]     [Marker 1]  │  ← ArUco markers at screen corners
│                              │
│       Non-touch display      │
│                              │
│  [Marker 3]     [Marker 2]  │
└──────────────────────────────┘
         ▲
         │ Camera looks at this
```

1. **Detect 4 markers** in the camera frame → get their pixel coordinates
2. **Apply calibration offsets** → adjust from marker center to actual screen corner
3. **Compute homography matrix** — a 3x3 transformation matrix that maps any point in the camera's view of the screen to normalized [0,1] coordinates
4. **Transform finger tip** — when a touch gesture is detected, transform the finger tip position through the homography to get the screen coordinate
5. **Execute click/move** at that screen coordinate

### Homography — The Math

A homography `H` maps points from one plane to another:

```
[x']     [h11 h12 h13]   [x]
[y'] = λ [h21 h22 h23] × [y]
[1 ]     [h31 h32 h33]   [1]
```

Given 4 known point correspondences (the 4 marker positions in camera → screen corners), OpenCV's `findHomography()` computes the 8 degrees of freedom in H.

### Partial Occlusion Recovery

During use, the user's hand covers markers. The system handles this:

| Visible Markers | Recovery Strategy |
|-----------------|-------------------|
| 4 | Direct detection (ideal) |
| 3 | Estimate missing corner using affine transform from 3 known points + cached positions |
| 2 | Estimate using similarity transform (translation + rotation + scale) from 2 known points |
| 1-0 | Grace period (5 frames) using cached homography, then invalidate |

### Temporal Smoothing (EMA)

```python
# Exponential Moving Average on corner positions
alpha = 0.75  # High alpha = responsive, low alpha = smooth
smoothed_corner = alpha * new_detection + (1 - alpha) * previous_smoothed
```

This prevents jitter in the homography mapping when markers are detected with slight variations frame-to-frame.

---

## 12. Paper Macro Pad — Physical Interface from CV

### Concept

A piece of A4 paper, printed with ArUco markers, that folds into a **triangular prism**. Each face of the prism has 8 buttons (4x2 grid) mapped to customizable actions. 3 faces = 24 total buttons.

```
           Flat Layout (A4 Paper):
┌─────────────────────┬─────────────────────┬─────────────────────┐
│   Set 1 (ID: 12)    │   Set 2 (ID: 13)    │   Set 3 (ID: 14)    │
│  [TL]  btn btn [TR]  │  [TL]  btn btn [TR]  │  [TL]  btn btn [TR]  │
│  [ML]  btn btn [MR]  │  [ML]  btn btn [MR]  │  [ML]  btn btn [MR]  │
│  [BL]  btn btn [BR]  │  [BL]  btn btn [BR]  │  [BL]  btn btn [BR]  │
│  [BL2]       [BR2]  │  [BL2]       [BR2]  │  [BL2]       [BR2]  │
└─────────────────────┴─────────────────────┴─────────────────────┘
                      ↓ Fold into prism ↓
                    ╱─────────╲
                   ╱  Face 1   ╲
                  ╱─────────────╲
                 │    Face 2     │
                 │               │
                  ╲─────────────╱
                   ╲  Face 3   ╱
                    ╲─────────╱
```

### 8-Marker Layout Per Face

```
[TL: set_id]  ·····  [TR: marker 4]
   [ML: 5]    ·····    [MR: 6]
   [BL: 7]    ·····    [BR: 8]
   [BL2: 9]            [BR2: 10]
```

- **TL marker ID** identifies which set is facing the camera (12, 13, or 14)
- **BL2/BR2** are fallback markers — when the user's hand covers BL/BR, the fallback markers maintain grid tracking
- Total: 8 markers, 4 corners + 2 midpoints + 2 fallbacks

### Detection Pipeline

1. **Three-pass detection** for robustness:
   - Pass 1: Standard grayscale
   - Pass 2: CLAHE (Contrast Limited Adaptive Histogram Equalization) enhanced
   - Pass 3: Gaussian blur to handle noise
2. **Set identification** by TL marker ID
3. **Corner recovery** when markers are occluded (parallelogram geometry, cached position blend)
4. **Grid subdivision** via bilinear interpolation → 8 button polygons
5. **Point-in-polygon** test: is the finger tip inside a button?

### Touch Activation Logic

```
HOVER: Finger is over a button but not touching
  → Visual feedback (highlight button)
  → 5-frame hover memory (button stays highlighted briefly after finger leaves)

TOUCH: Finger contacts the surface (detected by gesture model as "touch")
  → Cooldown check (0.7s global, per-button tracking)
  → Lookup button action from settings
  → Execute action (keyboard shortcut, app launch, text paste, etc.)
```

### Engineering Challenges Solved

1. **15+ physical prototypes** tested to optimize marker placement vs button density
2. **Hand occlusion tolerance** — works even when 3-4 of 8 markers are covered
3. **Camera angle robustness** — tested across webcam heights, tilts, and lighting
4. **Foldable design** — origami-inspired folding creates stable triangular prism

---

## 13. Screen Overlay Macro Pad

### Concept

A transparent overlay window displayed on screen with 12 buttons (4x3 grid) and its own ArUco markers (IDs 20-27). The camera detects the markers displayed on the screen itself.

```
┌──────────────────────────────┐
│  Overlay Window (on screen)  │
│  ┌────┬────┬────┬────┐      │
│  │ B0 │ B1 │ B2 │ B3 │      │
│  ├────┼────┼────┼────┤      │
│  │ B4 │ B5 │ B6 │ B7 │      │
│  ├────┼────┼────┼────┤      │
│  │ B8 │ B9 │B10 │B11 │      │
│  └────┴────┴────┴────┘      │
│  [ArUco markers at edges]    │
└──────────────────────────────┘
```

### How it differs from the paper macro pad

- No physical printing required
- Buttons are drawn on-screen with semi-transparent backgrounds
- The camera reads ArUco markers displayed on the monitor
- Same detection pipeline, different marker IDs (20-27 vs 4-14)

---

## 14. Action Execution System

### How Gestures Map to Actions

```yaml
# In handflow_setting.yaml:
Right_thumb_middle_swipe:
  actions:
    - type: shortcut      # Open command palette
      value: cmd+shift+p
      delay: 0.17
    - type: text           # Type command
      value: "run python file in terminal"
      delay: 0.17
    - type: shortcut      # Press enter
      value: enter
      delay: 0.17
```

This is a **multi-action sequence** — one gesture triggers 3 chained actions with delays between them.

### Supported Action Types (22 total)

| Category | Actions |
|----------|---------|
| **Mouse** | leftclick, rightclick, doubleclick |
| **Keyboard** | shortcut (e.g., "cmd+shift+s"), text (paste via clipboard) |
| **File** | Open file/app (platform-aware: `open` on macOS, `xdg-open` on Linux) |
| **Scroll** | scroll_up, scroll_down |
| **Zoom** | zoom_in (cmd+=), zoom_out (cmd+-) |
| **Media** | media_play, media_next, media_prev, volume_up, volume_down, volume_mute |
| **Window** | screenshot, minimize, maximize, desktop_left, desktop_right |

### Platform-Specific Optimization

On macOS, HandFlow uses **Quartz Core Graphics** for native mouse events:

```python
# Native macOS cursor movement (zero overhead)
event = Quartz.CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), 0)
Quartz.CGEventPost(kCGHIDEventTap, event)
```

This bypasses Python's overhead and provides the lowest possible latency for cursor tracking. The system falls back to PyAutoGUI on other platforms.

### Text Input Optimization

Instead of typing characters one by one (slow), HandFlow uses clipboard paste:

```python
# Save current clipboard
old_clipboard = pyperclip.paste()
# Set text to clipboard
pyperclip.copy(text_to_type)
# Paste (instant)
pyautogui.hotkey('command', 'v')
# Restore clipboard
pyperclip.copy(old_clipboard)
```

---

## 15. Mouse Control — Smooth Cursor Tracking

### The Challenge

Converting hand position (normalized 0-1 coordinates from MediaPipe) to smooth cursor movement requires handling:
- Noise in hand tracking (jitter)
- Different hand depths (closer = more sensitive)
- Natural hand tremor
- Different monitor sizes and multi-monitor setups

### Algorithm (in `MouseController._follow_loop`)

```
1. ORIGIN TRACKING
   Record starting hand position + cursor position
   All movement is relative to this origin

2. COMPUTE DELTA
   delta = current_hand_position - origin_position

3. ACTIVATION CHECK
   if |delta| < activation_threshold (0.06):
       don't start moving (prevents accidental activation)
   if already moving and |delta| < inner_deadzone (0.005):
       stop moving

4. DEADZONE PROCESSING
   inner_deadzone (0.005): No movement at all (absorbs tremor)
   outer_deadzone (0.014): 50% movement speed (transition zone)

5. DEPTH SENSITIVITY
   z_normalized = normalize(hand_z, [2e-7, 8e-7])
   depth_factor = (1 - z_normalized) ^ 1.2
   # Closer hand → higher sensitivity

6. NON-LINEAR SCALING
   speed = (|delta|^0.5) * 1.4 + 0.001
   # Square root gives progressive acceleration:
   # small movements are precise, large movements are fast

7. ADAPTIVE SMOOTHING
   if hand_moving_fast:
       alpha = 0.25  # Low smoothing → responsive
   else:
       alpha = 0.92  # High smoothing → stable
   smoothed = alpha * raw + (1-alpha) * previous

8. BUFFER SMOOTHING
   5-frame moving average for final position

9. MICRO-HYSTERESIS
   Ignore movements < 0.00025 normalized units
   (prevents sub-pixel jitter)
```

---

## 16. Signal Processing — OneEuro Filter

### What is the OneEuro Filter?

A **speed-adaptive low-pass filter** designed specifically for noisy interactive signals (like hand tracking). It was created by researchers at Inria for cursor smoothing.

### Core Idea

- When the signal is **moving slowly** → apply strong smoothing (reduce jitter)
- When the signal is **moving quickly** → apply weak smoothing (preserve responsiveness)

### How It Works

```
1. Compute the derivative (speed) of the signal:
   dx = (x_current - x_previous) / dt

2. Smooth the derivative with a fixed low-pass filter:
   dx_hat = LowPass(dx, d_cutoff)

3. Compute adaptive cutoff frequency:
   cutoff = min_cutoff + beta * |dx_hat|

   When stationary: cutoff ≈ min_cutoff (low → strong smoothing)
   When moving:     cutoff ≈ min_cutoff + beta * speed (high → weak smoothing)

4. Apply low-pass filter to the signal with adaptive cutoff:
   x_hat = LowPass(x, cutoff)
```

### Parameters in HandFlow

```python
min_cutoff = 1.4   # Base smoothness (higher = less smooth)
beta = 0.07        # Speed coefficient (higher = more responsive to fast movement)
d_cutoff = 1.0     # Derivative smoothness
```

### Why not just use a regular low-pass filter?

A fixed low-pass filter forces a choice:
- Low cutoff → smooth but laggy (bad for fast gestures)
- High cutoff → responsive but jittery (bad for hovering)

OneEuro adapts in real-time, giving you both.

---

## 17. Training Pipeline

### Data Collection

```bash
python scripts/collect_data.py
```

- Opens camera with gesture label selector
- Records hand landmark sequences as `.npy` files
- Directory structure: `data/raw/{hand}_mp_data/{gesture}/{sequence_id}/{frame}.npy`
- Each frame: 84 values (21 landmarks x 4)

### Data Processing

```bash
python scripts/dataset.py
```

1. Load raw `.npy` files per gesture
2. Run `FeatureEngineer.transform()` → 96 features per frame
3. Validate sequences (check NaN, outliers, length)
4. Save as compressed NPZ with config hash

### Training

```bash
python scripts/train.py --architecture tcn --epochs 100
```

**Pipeline:**
1. Load processed data for both hands, concatenate
2. Build model from config
3. Compute balanced class weights (handles class imbalance)
4. Create callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
5. Train with `AugmentedDataGenerator` (on-the-fly augmentation)
6. Log to W&B and TensorBoard
7. Save best model

### Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Adaptive learning rates, standard for small models |
| Learning rate | 0.0001 | Conservative — prevents overshooting |
| Batch size | 4 | Small batch = more gradient updates, better for small datasets |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=8) | Halve LR if val_accuracy plateaus for 8 epochs |
| Early stopping | patience=25 | Stop if no improvement for 25 epochs |
| Validation split | 15% | |
| Augmentation variants | 6 per sample | Effectively 7x dataset size |

### Resumable Training

```bash
python scripts/train.py --resume models/hand_action.keras --lr 0.00001 --epochs 50
```

Loads existing model, recompiles with fresh optimizer at new learning rate. Useful for fine-tuning after initial training.

---

## 18. Data Augmentation — Geometric Perturbation

### Why Augment?

Hand gesture data has natural variation:
- Different hand sizes
- Different camera distances
- Different camera angles
- Different lighting (affects depth estimation)
- Hand occlusion

Augmentation simulates these variations during training so the model generalizes better to unseen conditions.

### Augmentations Applied

| Augmentation | Prob | What it simulates | How it works |
|---|---|---|---|
| **Gaussian noise** | 40% | Sensor noise, tracking jitter | Add N(0, 0.003) noise to each coordinate. Motion-adaptive: less noise when hand is static |
| **Uniform scaling** | 20% | Hand distance from camera | Multiply all coordinates by [0.9, 1.1] |
| **2D rotation** | 15% | Camera tilt, wrist rotation | Rotate XY coordinates around wrist by [-8, 8]° |
| **Z-axis scaling** | 30% | Depth sensor calibration variation | Multiply Z coordinates by [0.85, 1.15] |
| **Z-axis shift** | 30% | Camera depth offset | Add [-0.08, 0.08] to all Z coordinates |
| **Z proportional** | 25% | Hand thickness variation | Scale fingertip Z proportionally to distance from wrist |
| **Z finger length** | 25% | Individual finger length variation | Per-finger Z scaling [0.9, 1.1] with progressive chain |
| **Z noise** | 40% | Depth-specific noise | Gaussian noise on Z only (std=0.004) |
| **Hand tilt** | 20% | Hand tilting toward/away from camera | Rotate in Y-Z plane around wrist by [-12, 12]° |
| **Landmark dropout** | 15% | Partial occlusion | Zero out entire landmarks or fingertips |

### Why so many Z-axis augmentations?

MediaPipe's depth (Z) estimation is the **least reliable** dimension:
- Monocular depth is inherently ambiguous
- Varies significantly with lighting and hand orientation
- Different cameras have different depth scales

By heavily augmenting Z, the model learns to not over-rely on depth information while still using it when it's informative.

---

## 19. Data Pipeline & Caching

### Caching Strategy

Processing raw `.npy` files through feature engineering is expensive. HandFlow caches the results:

```python
# When saving:
config_hash = MD5(sequence_length + input_dim + architecture)
np.savez_compressed("data/processed/data.npz", X=X, y=y, config_hash=config_hash)

# When loading:
stored_hash = data['config_hash']
current_hash = compute_hash(current_config)
if stored_hash != current_hash:
    print("Cache invalid — re-process data")
```

This ensures cached data is automatically invalidated when the model configuration changes.

### Data Validation

Before training, sequences are validated:
- **NaN check** — no NaN values in features
- **Outlier check** — no values beyond 5 standard deviations
- **Length check** — sequences match expected length

---

## 20. Evaluation & Visualization

### Metrics

```python
evaluator = ModelEvaluator(model, config)
results = evaluator.evaluate(X_test, y_true)
# Returns:
#   accuracy, weighted precision, weighted recall, weighted F1
#   confusion matrix, per-class classification report
```

### Misclassification Analysis

```python
errors = evaluator.get_misclassified_samples(X_test, y_true, paths)
# For each error:
#   true_label, predicted_label, confidence
#   source file path (for debugging data quality)
```

### Visualizations

1. **Gesture animation** — Matplotlib animation of 21-landmark hand skeleton with color-coded fingers, replaying a gesture sequence
2. **Feature plots** — Plotly interactive charts showing:
   - Inter-finger distances over time
   - Key landmark positions over time
   - Velocity features over time
   - Finger bending angles over time

---

## 21. Application Layer — GUI & Real-Time Loop

### Main Application (`HandFlowApp`)

Built with **CustomTkinter** (modern-looking Tkinter wrapper). Tabbed interface:

| Tab | Purpose |
|-----|---------|
| **Gesture Mapping** | Configure per-gesture, per-hand action sequences (up to 10 actions each) |
| **Macro Pad** | Manage up to 12 macropad button sets, configure button actions, generate PDF |
| **Calibration** | ArUco screen corner calibration with live preview |

### Detection Window

The real-time processing window. Key design decisions:

- **Display resolution**: 640x360 (efficient for UI updates)
- **MediaPipe resolution**: 320x180 (sufficient for hand detection, 4x fewer pixels)
- **Frame scheduling**: MediaPipe every frame, TCN every 2 frames, ArUco every 3 frames
- **macOS App Nap prevention**: Uses `NSActivityLatencyCritical` flag + `caffeinate` subprocess

### Keyboard Shortcuts in Detection Window

| Key | Action |
|-----|--------|
| H | Toggle horizontal flip |
| V | Toggle vertical flip |
| S | Swap hands (left↔right) |
| D | Toggle debug drawing |
| C | Toggle FPS cap |
| O | Toggle overlay debug |
| R | Start/stop recording |
| Q | Quit |

---

## 22. Performance Characteristics

### Benchmark Results

| Component | Time per Frame | Notes |
|-----------|---------------|-------|
| MediaPipe Hands | ~15-20ms | Dominant cost; runs on 320x180 |
| Feature Engineering | ~0.1ms | Pure NumPy operations |
| TFLite Inference | ~0.5ms | 907 KB model, single-threaded |
| ArUco Detection | ~2-3ms | Includes sub-pixel refinement |
| MacroPad Detection | ~3-5ms | Three-pass detection |
| **Full Pipeline** | **~25-35ms** | **~28-40 FPS** |

### Memory Footprint

| Component | Memory |
|-----------|--------|
| TFLite model | 907 KB |
| MediaPipe runtime | ~50-100 MB |
| Feature buffers | ~10 KB |
| Camera buffer | ~5 MB |
| **Total app** | **~200-300 MB** |

### FPS Management Strategy

```
Camera captures at: 30 FPS (typical webcam)
Target model FPS:   20 FPS (configured)
Actual processing:  25-30 FPS (typical laptop)

Strategy:
- Cap processing at 20 FPS for consistent model input
- Adaptive frame sampling: skip frames when camera is faster
- Linear interpolation: generate frames when camera is slower
- Result: model always sees consistent 20 FPS input regardless of hardware
```

---

## 23. Key Engineering Decisions & Trade-offs

### 1. TCN over Transformer

**Decision:** Use TCN as the primary architecture despite Transformers being more "modern."

**Rationale:** For 12-frame sequences on CPU, TCN is faster, smaller, and equally accurate. Transformer's O(n²) self-attention is wasteful when the sequence is only 12 timesteps long. TCN's dilated convolutions cover the full window with O(n) complexity.

### 2. Feature Engineering over End-to-End Learning

**Decision:** Hand-craft 96 features from 84 raw coordinates instead of letting the model learn features.

**Rationale:**
- Small dataset (hand-collected) doesn't support learning robust features
- FPS normalization requires explicit velocity computation
- Translation invariance is trivially achieved by wrist subtraction
- Joint angles and pinch dynamics provide strong gesture-discriminative signals
- Reduces input complexity → smaller model → faster inference

### 3. TFLite over ONNX Runtime

**Decision:** Use TFLite for inference instead of ONNX Runtime or PyTorch Mobile.

**Rationale:** TFLite has first-party support from Google (same as MediaPipe), is well-optimized for CPU inference, and has a smaller runtime footprint. ONNX would add another dependency without clear benefit.

### 4. Separate Handedness Tracking

**Decision:** Build custom handedness tracking instead of trusting MediaPipe.

**Rationale:** MediaPipe's handedness labels flip randomly, which would cause the wrong hand's gesture to trigger the wrong action. The custom tracker adds ~0.1ms overhead but provides stable labels essential for correct operation.

### 5. OneEuro Filter over Kalman Filter

**Decision:** Use OneEuro filter for finger-tip smoothing instead of a Kalman filter.

**Rationale:** OneEuro is simpler to tune (3 parameters vs. Kalman's process/measurement noise matrices), specifically designed for interactive cursor tracking, and has the adaptive smoothing property that perfectly matches this use case.

### 6. Paper Macro Pad Design

**Decision:** Use a foldable triangular prism rather than a flat sheet.

**Rationale:**
- Flat sheet: all 24 buttons visible but most are occluded by the hand during use
- Triangular prism: 8 buttons per face, only the active face needs to be detected
- The prism naturally angles toward the camera, improving marker detection
- Physically intuitive: "flip to switch" between button sets

### 7. Cached Finger Position for Clicks

**Decision:** Use finger position from 6 frames ago for click events, not the current position.

**Rationale:** At the moment of touch, the finger is decelerating and position jitters. The position from 6 frames before (~300ms) is more stable because the finger was still approaching in a smooth trajectory.

---

## 24. Glossary of Key Terms

| Term | Definition |
|------|-----------|
| **ArUco Marker** | A binary square fiducial marker with a unique ID, used for camera-based tracking and pose estimation |
| **Batch Normalization** | A technique that normalizes layer inputs to stabilize and accelerate training |
| **Causal Convolution** | A convolution that only looks at current and past timesteps (no future information) |
| **Centroid** | The center point of a set of coordinates, used for spatial tracking |
| **Confusion Matrix** | A table showing correct vs. incorrect classifications for each gesture class |
| **Dilation** | In dilated convolution, the spacing between kernel elements — dilation=2 means the kernel skips every other timestep |
| **EMA** | Exponential Moving Average — a smoothing technique where recent values have more weight |
| **Feature Engineering** | Manually computing discriminative features from raw data instead of learning them end-to-end |
| **Flatbuffer** | A serialization format used by TFLite for efficient model storage and loading |
| **FPS Normalization** | Scaling velocity/acceleration features by frame rate so the model is frame-rate invariant |
| **Grace Period** | A brief delay before invalidating a detection, allowing temporary occlusions to be bridged |
| **Homography** | A 3x3 perspective transformation matrix mapping points between two planes |
| **Landmark** | A specific anatomical point on the hand (e.g., index finger tip, wrist) |
| **Majority Voting** | Taking the most common prediction over a window of frames to smooth classification |
| **OneEuro Filter** | A speed-adaptive low-pass filter that smooths slow movements while preserving fast ones |
| **Quantization** | Reducing the numerical precision of model weights (e.g., 32-bit float → 8-bit integer) |
| **Receptive Field** | The span of input timesteps that influence a particular output neuron |
| **Residual Connection** | A skip connection that adds the input directly to the output of a block, enabling gradient flow |
| **Softmax** | A function that converts raw model outputs into a probability distribution (sums to 1) |
| **TCN** | Temporal Convolutional Network — a CNN architecture designed for sequence data using dilated causal convolutions |
| **TFLite** | TensorFlow Lite — Google's framework for running ML models on edge devices |
| **Temporal** | Relating to time — "temporal features" capture how the hand moves over time, not just its static position |

---

*This document covers the full HandFlow system as of February 2025. For code-level details, refer to the source files in `src/handflow/`.*
