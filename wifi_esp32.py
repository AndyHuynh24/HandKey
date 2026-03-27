"""High-FPS WiFi video stream from ESP32-CAM (OV2640 / OV5640)."""

import cv2
import numpy as np
import time
import urllib.request
import threading
import argparse

# ── Default Config ──────────────────────────────────────
ESP32_IP = "192.168.1.100"


def configure_camera(ip: str, sensor: str = "auto"):
    """Configure ESP32-CAM for max FPS over WiFi.

    Key settings:
    - Lower resolution = higher FPS (QVGA 320x240 or CIF 400x296)
    - JPEG quality 10-12 = small frames = faster transfer
    - framesize: 4=CIF(400x296), 5=VGA(640x480), 0=QQVGA(160x120), 3=QVGA(320x240), 6=SVGA, 8=XGA
    """
    base = f"http://{ip}/control"
    settings = [
        ("framesize", "5"),   # VGA 640x480 — good balance of quality and speed
        ("quality", "10"),    # JPEG quality (4-63, lower = better quality but larger)
    ]

    # OV5640-specific: disable auto-focus, enable high-speed mode
    if sensor == "ov5640":
        settings += [
            ("af", "0"),           # Disable auto-focus (saves processing time)
            ("raw_gma", "1"),      # Enable gamma correction
            ("lenc", "1"),         # Lens correction
        ]

    for param, val in settings:
        url = f"{base}?var={param}&val={val}"
        try:
            urllib.request.urlopen(url, timeout=2)
        except Exception:
            pass  # Some params may not exist on all sensors

    print(f"[Config] Camera configured for max FPS (VGA, quality=10, sensor={sensor})")


def set_resolution(ip: str, framesize: int):
    """Change resolution on the fly.

    Framesize values:
      0=QQVGA(160x120)  3=QVGA(320x240)  4=CIF(400x296)
      5=VGA(640x480)     6=SVGA(800x600)  8=XGA(1024x768)
      9=SXGA(1280x1024) 10=UXGA(1600x1200)

    OV5640 also supports:
      12=FHD(1920x1080) 13=QXGA(2048x1536)
    """
    url = f"http://{ip}/control?var=framesize&val={framesize}"
    try:
        urllib.request.urlopen(url, timeout=2)
    except Exception:
        pass


def stream(ip: str, sensor: str = "auto"):
    """Low-latency MJPEG stream with threaded reader for max FPS."""

    stream_url = f"http://{ip}:81/stream"
    configure_camera(ip, sensor)
    time.sleep(0.3)

    print(f"Connecting to {stream_url} ...")

    # Threaded reader: continuously pulls bytes so we never fall behind
    latest_frame = [None]
    lock = threading.Lock()
    running = [True]

    def reader_thread():
        while running[0]:
            try:
                resp = urllib.request.urlopen(stream_url, timeout=10)
                buf = b""

                while running[0]:
                    chunk = resp.read(8192)
                    if not chunk:
                        break

                    buf += chunk

                    # Find JPEG boundaries
                    while True:
                        soi = buf.find(b"\xff\xd8")
                        eoi = buf.find(b"\xff\xd9")

                        if soi == -1 or eoi == -1 or eoi <= soi:
                            break

                        jpeg = buf[soi : eoi + 2]
                        buf = buf[eoi + 2 :]

                        # Decode immediately
                        arr = np.frombuffer(jpeg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with lock:
                                latest_frame[0] = frame

                    # Prevent buffer from growing unbounded
                    if len(buf) > 200_000:
                        last_soi = buf.rfind(b"\xff\xd8")
                        if last_soi > 0:
                            buf = buf[last_soi:]

                resp.close()
            except Exception as e:
                if running[0]:
                    print(f"[Stream] Connection lost ({e}), reconnecting...")
                    time.sleep(1)

    t = threading.Thread(target=reader_thread, daemon=True)
    t.start()

    # Wait for first frame
    print("Waiting for frames...")
    timeout_start = time.time()
    while latest_frame[0] is None:
        if time.time() - timeout_start > 10:
            print("Timeout: no frames received. Check IP and WiFi.")
            running[0] = False
            return
        time.sleep(0.05)

    print("Streaming. Keys: q=quit, 1-6=resolution, +=quality up, -=quality down")

    frame_count = 0
    fps_time = time.time()
    fps = 0.0
    quality = 10

    while True:
        with lock:
            frame = latest_frame[0]

        if frame is not None:
            frame_count += 1
            now = time.time()
            elapsed = now - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_time = now

            display = frame.copy()
            h, w = display.shape[:2]

            # FPS + resolution overlay
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"{w}x{h}  Q:{quality}", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

            cv2.imshow("ESP32-CAM WiFi", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            set_resolution(ip, 0)   # QQVGA 160x120
            print("[Res] QQVGA 160x120")
        elif key == ord("2"):
            set_resolution(ip, 3)   # QVGA 320x240
            print("[Res] QVGA 320x240")
        elif key == ord("3"):
            set_resolution(ip, 5)   # VGA 640x480
            print("[Res] VGA 640x480")
        elif key == ord("4"):
            set_resolution(ip, 6)   # SVGA 800x600
            print("[Res] SVGA 800x600")
        elif key == ord("5"):
            set_resolution(ip, 8)   # XGA 1024x768
            print("[Res] XGA 1024x768")
        elif key == ord("6"):
            set_resolution(ip, 10)  # UXGA 1600x1200
            print("[Res] UXGA 1600x1200")
        elif key == ord("+") or key == ord("="):
            quality = max(4, quality - 2)
            url = f"http://{ip}/control?var=quality&val={quality}"
            try: urllib.request.urlopen(url, timeout=2)
            except: pass
            print(f"[Quality] {quality} (lower=better)")
        elif key == ord("-"):
            quality = min(63, quality + 2)
            url = f"http://{ip}/control?var=quality&val={quality}"
            try: urllib.request.urlopen(url, timeout=2)
            except: pass
            print(f"[Quality] {quality} (lower=better)")

    running[0] = False
    cv2.destroyAllWindows()
    print(f"Final FPS: {fps:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESP32-CAM WiFi Stream (OV2640/OV5640)")
    parser.add_argument("--ip", default=ESP32_IP, help="ESP32 IP address")
    parser.add_argument("--sensor", choices=["auto", "ov2640", "ov5640"], default="auto",
                        help="Camera sensor type (default: auto)")
    args = parser.parse_args()

    stream(args.ip, args.sensor)
