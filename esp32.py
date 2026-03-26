import serial
import cv2
import numpy as np
import time

SERIAL_PORT = "/dev/cu.usbmodem101"
BAUD_RATE = 2000000

FRAME_START = b'\xff\xaa\xbb\xcc'
FRAME_END   = b'\xcc\xbb\xaa\xff'

def main():
    print("Connecting...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(4)
    ser.reset_input_buffer()
    print("Ready. Press 'q' to quit.")

    frame_count = 0
    fps_time = time.time()
    fps = 0

    while True:
        buf = ser.read(1024)
        while True:
            pos = buf.find(FRAME_START)
            if pos >= 0:
                buf = buf[pos + 4:]
                break
            new = ser.read(1024)
            if not new:
                continue
            buf = buf[-3:] + new

        while len(buf) < 4:
            buf += ser.read(4 - len(buf))
        frame_size = int.from_bytes(buf[:4], 'little')
        buf = buf[4:]

        if frame_size <= 0 or frame_size > 80000:
            ser.reset_input_buffer()
            continue

        while len(buf) < frame_size + 4:
            remaining = frame_size + 4 - len(buf)
            buf += ser.read(remaining)

        jpeg_data = buf[:frame_size]
        end_marker = buf[frame_size:frame_size + 4]
        buf = buf[frame_size + 4:]

        if end_marker != FRAME_END:
            ser.reset_input_buffer()
            continue

        img = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

        if frame is not None:
           
            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_time = time.time()

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("ESP32 Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()