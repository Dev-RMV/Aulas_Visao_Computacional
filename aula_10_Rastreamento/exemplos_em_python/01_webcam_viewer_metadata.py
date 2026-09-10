import argparse
import time

import cv2

from utils import open_capture, resize_keep_aspect, overlay_text


def main():
    parser = argparse.ArgumentParser(description="Visualizador simples com metadados de vídeo.")
    parser.add_argument("--source", default="0", help="0 para webcam ou caminho de vídeo")
    parser.add_argument("--width", type=int, default=800)
    args = parser.parse_args()

    cap = open_capture(args.source)
    frame_count = 0
    t0 = time.perf_counter()

    while True:
        tic = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_keep_aspect(frame, args.width)
        frame_count += 1
        elapsed = time.perf_counter() - t0
        fps = frame_count / max(elapsed, 1e-6)
        proc_ms = (time.perf_counter() - tic) * 1000

        lines = [
            f"shape: {frame.shape}",
            f"frame: {frame_count}",
            f"FPS medio: {fps:.1f}",
            f"tempo do loop: {proc_ms:.2f} ms",
        ]
        cv2.imshow("Fonte de video + metadados", overlay_text(frame, lines))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
