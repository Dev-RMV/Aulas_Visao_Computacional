import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import open_capture, resize_keep_aspect, clean_foreground_mask, contours_from_mask, annotate_contours, overlay_text


def parse_roi(values):
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("ROI deve ser x y w h")
    return tuple(map(int, values))


def initialize_camshift(frame, roi):
    x, y, w, h = roi
    track_window = (x, y, w, h)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_hsv = hsv[y:y + h, x:x + w]
    mask = cv2.inRange(roi_hsv, np.array((0, 40, 30)), np.array((180, 255, 255)))
    hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return track_window, hist


def create_kalman(q=0.03, r=8.0):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    # Q: ruído de processo. Maior Q permite reagir a acelerações e manobras, mas reduz suavização.
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * q
    # R: ruído da medição. Maior R faz o filtro confiar menos no centroide do CamShift.
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
    # P: incerteza do estado inicial. Maior P gera correções iniciais mais fortes.
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
    return kf


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo: MOG2, KNN, CamShift e Kalman.")
    parser.add_argument("--source", default="data/synthetic_motion.mp4")
    parser.add_argument("--roi", nargs="*", type=int, help="ROI opcional: x y w h")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--min-area", type=float, default=500)
    parser.add_argument("--q", type=float, default=0.03)
    parser.add_argument("--r", type=float, default=8.0)
    parser.add_argument("--output", default="output/full_pipeline_trajectory.csv")
    args = parser.parse_args()

    cap = open_capture(args.source)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Fonte sem frames.")
    first = resize_keep_aspect(first, args.width)

    roi = parse_roi(args.roi)
    if roi is None:
        roi = cv2.selectROI("Selecione ROI para CamShift", first, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Selecione ROI para CamShift")

    track_window, hist = initialize_camshift(first, roi)
    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
    kf = create_kalman(args.q, args.r)
    x, y, w, h = roi
    kf.statePost = np.array([[x + w / 2], [y + h / 2], [0], [0]], dtype=np.float32)

    times_mog2 = deque(maxlen=120)
    times_knn = deque(maxlen=120)
    times_track = deque(maxlen=120)
    measured_points = []
    predicted_points = []
    frame_idx = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = out_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "mog2_objects", "knn_objects", "measured_x", "measured_y", "kalman_x", "kalman_y", "track_ms"])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_keep_aspect(frame, args.width)
        frame_idx += 1

        tic = time.perf_counter()
        mask_mog2 = clean_foreground_mask(mog2.apply(frame))
        cont_mog2 = contours_from_mask(mask_mog2, args.min_area)
        times_mog2.append((time.perf_counter() - tic) * 1000)

        tic = time.perf_counter()
        mask_knn = clean_foreground_mask(knn.apply(frame))
        cont_knn = contours_from_mask(mask_knn, args.min_area)
        times_knn.append((time.perf_counter() - tic) * 1000)

        tic = time.perf_counter()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backproj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        rotated_rect, track_window = cv2.CamShift(backproj, track_window, term_criteria)
        measured_x, measured_y = rotated_rect[0]
        pred = kf.predict()
        pred_x, pred_y = float(pred[0]), float(pred[1])
        kf.correct(np.array([[np.float32(measured_x)], [np.float32(measured_y)]]))
        track_ms = (time.perf_counter() - tic) * 1000
        times_track.append(track_ms)

        measured_points.append((measured_x, measured_y))
        predicted_points.append((pred_x, pred_y))
        writer.writerow([frame_idx, len(cont_mog2), len(cont_knn), measured_x, measured_y, pred_x, pred_y, track_ms])

        ann_mog2 = annotate_contours(frame, cont_mog2)
        ann_knn = annotate_contours(frame, cont_knn, color=(255, 128, 0))
        tracking = frame.copy()
        cv2.ellipse(tracking, rotated_rect, (0, 255, 0), 2)
        cv2.circle(tracking, (int(measured_x), int(measured_y)), 6, (0, 0, 255), -1)
        cv2.circle(tracking, (int(pred_x), int(pred_y)), 6, (255, 0, 0), -1)

        ann_mog2 = overlay_text(ann_mog2, [f"MOG2 objetos: {len(cont_mog2)}", f"media: {sum(times_mog2)/len(times_mog2):.2f} ms"])
        ann_knn = overlay_text(ann_knn, [f"KNN objetos: {len(cont_knn)}", f"media: {sum(times_knn)/len(times_knn):.2f} ms"])
        tracking = overlay_text(tracking, ["CamShift + Kalman", "vermelho=medido | azul=predito", f"media: {sum(times_track)/len(times_track):.2f} ms"])

        cv2.imshow("MOG2 mascara", mask_mog2)
        cv2.imshow("KNN mascara", mask_knn)
        cv2.imshow("MOG2 anotado", ann_mog2)
        cv2.imshow("KNN anotado", ann_knn)
        cv2.imshow("CamShift + Kalman", tracking)

        print(
            f"frame={frame_idx:04d} | MOG2={len(cont_mog2)} | KNN={len(cont_knn)} | "
            f"med=({measured_x:.1f},{measured_y:.1f}) | kalman=({pred_x:.1f},{pred_y:.1f}) | track_ms={track_ms:.2f}"
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    if measured_points:
        measured = np.array(measured_points)
        predicted = np.array(predicted_points)
        plt.figure(figsize=(8, 5))
        plt.plot(measured[:, 0], measured[:, 1], "r.-", label="CamShift medido")
        plt.plot(predicted[:, 0], predicted[:, 1], "b.-", label="Kalman predito")
        plt.gca().invert_yaxis()
        plt.xlabel("x [pixels]")
        plt.ylabel("y [pixels]")
        plt.title("Pipeline completo: trajetórias")
        plt.grid(True)
        plt.legend()
        plot_path = out_path.with_suffix(".png")
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        print(f"CSV salvo em: {out_path}")
        print(f"Gráfico salvo em: {plot_path}")
        plt.show()


if __name__ == "__main__":
    main()
