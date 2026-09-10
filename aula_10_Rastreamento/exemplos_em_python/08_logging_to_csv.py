import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2

from utils import open_capture, resize_keep_aspect, clean_foreground_mask, contours_from_mask, annotate_contours, overlay_text


def main():
    parser = argparse.ArgumentParser(description="Registra métricas de foreground em CSV.")
    parser.add_argument("--source", default="data/synthetic_motion.mp4")
    parser.add_argument("--method", choices=["mog2", "knn"], default="mog2")
    parser.add_argument("--output", default="output/background_metrics.csv")
    parser.add_argument("--min-area", type=float, default=500)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    cap = open_capture(args.source)
    if args.method == "mog2":
        subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    else:
        subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = deque(maxlen=120)
    frame_idx = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "method", "objects", "loop_ms", "avg_loop_ms"])

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
            loop_ms = (time.perf_counter() - tic) * 1000
            times.append(loop_ms)
            avg_ms = sum(times) / len(times)
            writer.writerow([frame_idx, args.method, len(contours), loop_ms, avg_ms])

            annotated = annotate_contours(frame, contours)
            annotated = overlay_text(annotated, [
                f"metodo: {args.method}",
                f"frame: {frame_idx}",
                f"objetos: {len(contours)}",
                f"media: {avg_ms:.2f} ms",
            ])
            cv2.imshow("Mascara", clean)
            cv2.imshow("Frame anotado", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"CSV salvo em: {out_path}")


if __name__ == "__main__":
    main()
