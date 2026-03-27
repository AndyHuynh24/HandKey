# HandFlow

## One camera. Any surface. Total control.

HandFlow turns your webcam into the only input device you need. Point it at your hands, and every surface becomes a controller.

---

### The Problem

You're stuck with the same input devices from 1964. Mouse. Keyboard. Trackpad. They're fine — but they force you into one way of working. What if your desk, a sheet of paper, or thin air could be just as powerful?

### The Solution

HandFlow uses real-time hand gesture recognition to give you a new layer of computer control — no wearables, no special hardware, just a camera you already have.

**Your hands become the controller:**
- Point and touch → move the cursor
- Pinch → click
- Swipe → scroll, switch desktops
- Close fist → trigger shortcuts, launch apps

**Any printed page becomes a macro pad:**
- Print a single sheet of paper with ArUco markers
- Place it on your desk
- Touch the buttons on the paper — HandFlow detects your finger position and fires the mapped action
- Each "paper pad" is fully customizable: shortcuts, app launches, media controls, text macros
- Fold it. Pocket it. Print a new one in seconds.

**Screen overlay mode:**
- No printer? HandFlow displays virtual buttons directly on your screen
- Same touch detection, zero paper needed

---

### How It Works

1. A webcam captures your hands at 20+ FPS
2. MediaPipe extracts 21 hand landmarks per hand
3. A custom TCN (Temporal Convolutional Network) classifies gestures in real-time
4. ArUco marker detection maps finger positions to physical or on-screen buttons
5. Actions execute instantly — mouse control, keyboard shortcuts, app launches, multi-step macros

**Built with:** Python · TensorFlow · MediaPipe · OpenCV · NiceGUI

---

### Why This Matters

- **Zero additional hardware** — works with any webcam, including built-in laptop cameras
- **Instant customization** — remap gestures and macropad buttons in the UI, print a new pad in seconds
- **8,200+ training samples** across 12 gesture classes, trained on both hands
- **Sub-frame latency** — gesture detection runs in under 5ms per inference

---

### Demo Flow

1. Open HandFlow → show the clean NiceGUI configuration interface
2. Wave hands → real-time gesture detection with live preview
3. Place a printed paper macropad on the desk → touch buttons to trigger actions
4. Switch macro pad sets by swapping printed sheets
5. Toggle screen overlay mode → virtual buttons appear on screen
6. Customize a button live → save → use immediately

---

### One-liner

**HandFlow turns a $5 webcam and a sheet of paper into the most customizable input device ever made.**
