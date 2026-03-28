"""Knuckle MacroPad — turns palm-up hand finger segments into touchable buttons."""

import time
import cv2
import numpy as np
from typing import Optional, Tuple, Dict

from handflow.utils import Setting, get_logger
from handflow.actions import ActionExecutor


class KnuckleMacroPad:
    """
    When palm faces up, the segments between finger landmarks become buttons.

    Index finger:  btn 0 (5→6), btn 1 (6→7), btn 2 (7→8)
    Middle finger: btn 3 (9→10), btn 4 (10→11), btn 5 (11→12)

    Each button is a rotated rectangle oriented along the finger segment,
    with width and height = 95% of the distance between the two landmarks.
    """

    # (button_idx, landmark_start, landmark_end)
    BUTTONS = [
        (0, 5, 6),    # Index MCP→PIP
        (1, 6, 7),    # Index PIP→DIP
        (2, 7, 8),    # Index DIP→TIP
        (3, 9, 10),   # Middle MCP→PIP
        (4, 10, 11),  # Middle PIP→DIP
        (5, 11, 12),  # Middle DIP→TIP
    ]

    SIZE_RATIO = 0.95
    ACTIVATION_COOLDOWN = 0.4  # seconds

    def __init__(self, setting: Setting, executor: ActionExecutor):
        self.logger = get_logger("handflow.knuckle_macropad")
        self.setting = setting
        self.executor = executor

        self._left_active = False
        self._right_active = False
        self._last_activation_times = {i: 0.0 for i in range(6)}
        self._frame_count = 0

        self._left_buttons = []
        self._right_buttons = []
        self._hovered_left = None
        self._hovered_right = None

        self.logger.info("[KnuckleMacroPad] Initialized — 6 buttons per hand (index + middle)")

    def _get_xy(self, kp, landmark_id):
        """Get (x, y) from flattened keypoints array (21*4)."""
        base = landmark_id * 4
        return kp[base], kp[base + 1]

    def is_palm_up(self, kp) -> bool:
        """Check if palm faces up (knuckles visible to camera).
        Accounts for camera flip settings since they change coordinate space.
        """
        wrist_x, wrist_y = self._get_xy(kp, 0)
        index_x, index_y = self._get_xy(kp, 5)
        pinky_x, pinky_y = self._get_xy(kp, 17)

        v1_x = index_x - wrist_x
        v1_y = index_y - wrist_y
        v2_x = pinky_x - wrist_x
        v2_y = pinky_y - wrist_y

        cross_z = v1_x * v2_y - v1_y * v2_x

        # Flip settings change the coordinate handedness
        flip_h = self.setting.camera.flip_horizontal
        flip_v = self.setting.camera.flip_vertical

        # Each flip reverses the cross product sign
        if flip_h:
            cross_z = -cross_z
        if flip_v:
            cross_z = -cross_z

        return cross_z > 0

    def _compute_buttons(self, kp):
        """Compute rotated rectangle buttons from keypoints.
        Returns list of (idx, center_x, center_y, corners) for each button.
        corners is a (4, 2) array of the rectangle vertices in normalized coords.
        """
        buttons = []
        for idx, lm_a, lm_b in self.BUTTONS:
            ax, ay = self._get_xy(kp, lm_a)
            bx, by = self._get_xy(kp, lm_b)

            # Center
            cx = (ax + bx) / 2
            cy = (ay + by) / 2

            # Direction vector and length
            dx = bx - ax
            dy = by - ay
            length = np.sqrt(dx * dx + dy * dy)
            if length < 1e-6:
                continue

            # Unit vectors along and perpendicular to segment
            ux = dx / length  # along finger
            uy = dy / length
            px = -uy  # perpendicular
            py = ux

            # Half sizes (95% of segment length for both width and height)
            half_h = length * self.SIZE_RATIO / 2  # along finger
            half_w = length * self.SIZE_RATIO / 2  # perpendicular

            # 4 corners of rotated rectangle
            corners = np.array([
                [cx - px * half_w - ux * half_h, cy - py * half_w - uy * half_h],
                [cx + px * half_w - ux * half_h, cy + py * half_w - uy * half_h],
                [cx + px * half_w + ux * half_h, cy + py * half_w + uy * half_h],
                [cx - px * half_w + ux * half_h, cy - py * half_w + uy * half_h],
            ])

            buttons.append((idx, cx, cy, corners))

        return buttons

    def _point_in_rotated_rect(self, px, py, corners):
        """Check if point (px, py) is inside a rotated rectangle defined by 4 corners.
        Works regardless of winding order — all cross products must have the same sign.
        """
        n = len(corners)
        signs = []
        for i in range(n):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % n]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            signs.append(cross)
        return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)

    def update(
        self,
        left_kp,
        right_kp,
        left_index_tip: Optional[Tuple[float, float]],
        right_index_tip: Optional[Tuple[float, float]],
        left_gesture: str,
        right_gesture: str,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Update knuckle macropad for both hands.

        Args:
            left_kp: left hand keypoints (84,) or None
            right_kp: right hand keypoints (84,) or None
            left_index_tip: (x, y) normalized left index tip
            right_index_tip: (x, y) normalized right index tip
            left_gesture: left hand gesture string
            right_gesture: right hand gesture string

        Returns:
            (activated_on_left_hand, activated_on_right_hand)
            Each is a button index (0-5) or None
        """
        self._frame_count += 1
        now = time.time()

        # Left hand keypoints are x-flipped by the hand tracker for canonicalization.
        # Un-flip them so button positions match the actual screen coordinates.
        left_kp_screen = None
        if left_kp is not None and np.any(left_kp):
            left_kp_screen = left_kp.copy()
            for i in range(21):
                left_kp_screen[i * 4] = 1.0 - left_kp_screen[i * 4]  # flip x back

        # Check palm orientation (use screen-space coords)
        self._left_active = (left_kp_screen is not None and
                             self.is_palm_up(left_kp_screen))
        self._right_active = (right_kp is not None and np.any(right_kp) and
                              self.is_palm_up(right_kp))

        # Compute button rectangles in screen space
        self._left_buttons = self._compute_buttons(left_kp_screen) if self._left_active else []
        self._right_buttons = self._compute_buttons(right_kp) if self._right_active else []

        activated_left = None
        activated_right = None
        self._hovered_left = None
        self._hovered_right = None

        touch_gestures = ("touch", "touch_hover", "touch_hold")

        # Right hand interacts with left hand's knuckle buttons
        if self._left_active and right_index_tip is not None:
            tip_x, tip_y = right_index_tip
            for idx, cx, cy, corners in self._left_buttons:
                if self._point_in_rotated_rect(tip_x, tip_y, corners):
                    self._hovered_left = idx
                    # Activate on touch gesture
                    if right_gesture in touch_gestures:
                        if now - self._last_activation_times[idx] > self.ACTIVATION_COOLDOWN:
                            self._last_activation_times[idx] = now
                            activated_left = idx
                            self._execute_button(idx)
                    break

        # Left hand interacts with right hand's knuckle buttons
        if self._right_active and left_index_tip is not None:
            tip_x, tip_y = left_index_tip
            for idx, cx, cy, corners in self._right_buttons:
                if self._point_in_rotated_rect(tip_x, tip_y, corners):
                    self._hovered_right = idx
                    if left_gesture in touch_gestures:
                        if now - self._last_activation_times[idx] > self.ACTIVATION_COOLDOWN:
                            self._last_activation_times[idx] = now
                            activated_right = idx
                            self._execute_button(idx)
                    break

        return activated_left, activated_right

    def _execute_button(self, idx: int):
        """Execute the action mapped to a knuckle button."""
        knuckle_buttons = getattr(self.setting, 'knuckle_macropad_buttons', {})
        if not knuckle_buttons:
            return

        from handflow.utils.setting import MacroPadButton
        btn_config = knuckle_buttons.get(idx, MacroPadButton())
        actions = btn_config.get_actions()
        if not actions:
            return

        label = btn_config.label or f"Knuckle {idx + 1}"
        self.logger.info(f"[KnuckleMacroPad] Activated: {label} (btn {idx})")

        import threading
        def run_actions():
            for action in actions:
                if action.delay > 0:
                    time.sleep(action.delay)
                self.executor.execute(action.type, action.value)
        threading.Thread(target=run_actions, daemon=True).start()

    def draw_debug(self, image: np.ndarray, w: int, h: int) -> np.ndarray:
        """Draw knuckle button rectangles on debug view."""
        for buttons, base_color, hovered_idx in [
            (self._left_buttons, (0, 200, 255), self._hovered_left),
            (self._right_buttons, (255, 200, 0), self._hovered_right),
        ]:
            for idx, cx, cy, corners in buttons:
                pts = (corners * [w, h]).astype(np.int32)

                if idx == hovered_idx:
                    # Hovered: filled with transparency + thick border
                    overlay = image.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
                    cv2.polylines(image, [pts], True, (0, 255, 0), 3)
                else:
                    cv2.polylines(image, [pts], True, base_color, 2)

                # Button number at center
                px, py = int(cx * w), int(cy * h)
                cv2.putText(image, str(idx + 1), (px - 5, py + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return image

    def is_active(self) -> bool:
        return self._left_active or self._right_active
