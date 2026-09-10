import argparse
import time
from collections import deque

import cv2
import numpy as np

from utils import open_capture, resize_keep_aspect, overlay_text


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


def main():
    parser = argparse.ArgumentParser(description="CamShift com ROI inicial.")
    parser.add_argument("--source", default="data/synthetic_motion.mp4")
    parser.add_argument("--roi", nargs="*", type=int, help="ROI opcional: x y w h")
    parser.add_argument("--width", type=int, default=800)
    args = parser.parse_args()

    cap = open_capture(args.source)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Fonte sem frames.")
    frame = resize_keep_aspect(frame, args.width)

    roi = parse_roi(args.roi)
    if roi is None:
        roi = cv2.selectROI("Selecione a ROI do objeto", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Selecione a ROI do objeto")

    track_window, hist = initialize_camshift(frame, roi)
    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    times = deque(maxlen=120)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_keep_aspect(frame, args.width)
        frame_idx += 1

        tic = time.perf_counter()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backproj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        rotated_rect, track_window = cv2.CamShift(backproj, track_window, term_criteria)
        elapsed_ms = (time.perf_counter() - tic) * 1000
        times.append(elapsed_ms)

        out = frame.copy()
        cv2.ellipse(out, rotated_rect, (0, 255, 0), 2)
        cx, cy = map(int, rotated_rect[0])
        cv2.circle(out, (cx, cy), 4, (0, 255, 255), -1)
        avg_ms = sum(times) / len(times)
        out = overlay_text(out, [
            "CamShift",
            f"centro medido: ({cx}, {cy})",
            f"tempo medio: {avg_ms:.2f} ms",
            "r: reiniciar ROI | q: sair",
        ])

        cv2.imshow("Backprojection HSV", backproj)
        cv2.imshow("CamShift - elipse de rastreamento", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            roi = cv2.selectROI("Selecione nova ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Selecione nova ROI")
            track_window, hist = initialize_camshift(frame, roi)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
