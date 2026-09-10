import argparse
import csv
import time
from pathlib import Path

import cv2

from utils import open_capture, resize_keep_aspect, clean_foreground_mask, contours_from_mask


def run_sweep(source, method, thresholds, frames_limit, min_area, width):
    rows = []
    for threshold in thresholds:
        cap = open_capture(source)
        if method == "mog2":
            subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=threshold, detectShadows=True)
        else:
            subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=threshold, detectShadows=True)

        frame_idx = 0
        total_ms = 0
        total_objects = 0
        while frame_idx < frames_limit:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_keep_aspect(frame, width)
            tic = time.perf_counter()
            mask = subtractor.apply(frame)
            clean = clean_foreground_mask(mask)
            contours = contours_from_mask(clean, min_area=min_area)
            total_ms += (time.perf_counter() - tic) * 1000
            total_objects += len(contours)
            frame_idx += 1

        cap.release()
        avg_ms = total_ms / max(frame_idx, 1)
        avg_objects = total_objects / max(frame_idx, 1)
        rows.append({
            "method": method,
            "threshold": threshold,
            "frames": frame_idx,
            "avg_ms": avg_ms,
            "avg_objects": avg_objects,
        })
        print(f"{method} threshold={threshold} frames={frame_idx} avg_ms={avg_ms:.2f} avg_objects={avg_objects:.2f}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Varredura simples de parâmetros MOG2/KNN.")
    parser.add_argument("--source", default="data/synthetic_motion.mp4")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--min-area", type=float, default=500)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--output", default="output/background_parameter_sweep.csv")
    args = parser.parse_args()

    thresholds_mog2 = [8, 12, 16, 24, 32]
    thresholds_knn = [100, 200, 400, 800, 1200]
    rows = []
    rows.extend(run_sweep(args.source, "mog2", thresholds_mog2, args.frames, args.min_area, args.width))
    rows.extend(run_sweep(args.source, "knn", thresholds_knn, args.frames, args.min_area, args.width))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "threshold", "frames", "avg_ms", "avg_objects"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resultados salvos em: {out}")


if __name__ == "__main__":
    main()
