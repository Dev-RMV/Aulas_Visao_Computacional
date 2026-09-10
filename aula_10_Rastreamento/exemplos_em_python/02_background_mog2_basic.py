import argparse
import time
from collections import deque

import cv2

from utils import open_capture, resize_keep_aspect, clean_foreground_mask, contours_from_mask, annotate_contours, overlay_text


def main():
    parser = argparse.ArgumentParser(description="Background subtraction com MOG2.")
    parser.add_argument("--source", default="data/synthetic_motion.mp4")
    parser.add_argument("--history", type=int, default=500)
    parser.add_argument("--var-threshold", type=float, default=16)
    parser.add_argument("--detect-shadows", action="store_true")
    parser.add_argument("--min-area", type=float, default=500)
    parser.add_argument("--width", type=int, default=800)
    args = parser.parse_args()

    cap = open_capture(args.source)
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.var_threshold,
        detectShadows=args.detect_shadows,
    )
    times = deque(maxlen=120)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_keep_aspect(frame, args.width)
        frame_idx += 1

        tic = time.perf_counter()
        mask = subtractor.apply(frame)
        clean = clean_foreground_mask(mask)
        contours = contours_from_mask(clean, min_area=args.min_area)
        elapsed_ms = (time.perf_counter() - tic) * 1000
        times.append(elapsed_ms)

        annotated = annotate_contours(frame, contours)
        avg_ms = sum(times) / len(times)
        annotated = overlay_text(annotated, [
            "MOG2",
            f"frame: {frame_idx}",
            f"objetos: {len(contours)}",
            f"tempo medio: {avg_ms:.2f} ms",
        ])

        print(f"MOG2 | frame={frame_idx:04d} | objetos={len(contours)} | tempo={elapsed_ms:.2f} ms | media={avg_ms:.2f} ms")
        cv2.imshow("MOG2 - mascara foreground", clean)
        cv2.imshow("MOG2 - frame anotado", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
