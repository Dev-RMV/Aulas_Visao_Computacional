"""Exemplo 12 — localização visual no vídeo usando SIFT + FLANN + RANSAC."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from utils import RESOURCES, create_feature, ensure_outputs, extract, flann_for, lowe_filter


parser = argparse.ArgumentParser()
parser.add_argument("--source", default=str(RESOURCES / "video_localizacao.mp4"), help="0 ou caminho de vídeo")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--max-frames", type=int, default=0)
parser.add_argument("--output-video", default=str(ensure_outputs() / "12_localizacao_visual.mp4"))
args = parser.parse_args()

reference = cv2.imread(str(RESOURCES / "cena_referencia.png"))
kp_ref, desc_ref, _ = extract("SIFT", reference, repeats=1)
sift = create_feature("SIFT")
source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir {args.source}")

writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
corners = np.float32([[0, 0], [reference.shape[1] - 1, 0],
                      [reference.shape[1] - 1, reference.shape[0] - 1],
                      [0, reference.shape[0] - 1]]).reshape(-1, 1, 2)

times, inlier_values = [], []
frame_index = 0
while True:
    ok, frame = cap.read()
    if not ok or (args.max_frames and frame_index >= args.max_frames):
        break
    start = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = sift.detectAndCompute(gray, None)
    inliers = 0
    if desc_frame is not None:
        good = lowe_filter(flann_for("SIFT").knnMatch(desc_ref, desc_frame, k=2), 0.75)
        if len(good) >= 8:
            pa = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pb = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
            if homography is not None and mask is not None:
                inliers = int(mask.sum())
                if inliers >= 12:
                    projected = cv2.perspectiveTransform(corners, homography)
                    cv2.polylines(frame, [np.int32(projected)], True, (38, 226, 167), 5, cv2.LINE_AA)
    latency = (time.perf_counter() - start) * 1000
    times.append(latency); inlier_values.append(inliers)
    cv2.putText(frame, f"inliers={inliers} | {latency:.1f} ms", (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (38, 226, 167), 2, cv2.LINE_AA)
    writer.write(frame)
    if not args.headless:
        cv2.imshow("Localização visual", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    frame_index += 1

cap.release(); writer.release(); cv2.destroyAllWindows()
values = np.asarray(times)
print(f"Frames processados: {len(values)}")
print(f"Latência média: {values.mean():.2f} ms/frame")
print(f"Inliers médios: {np.mean(inlier_values):.1f}")
print(f"Frames localizados (>=12 inliers): {sum(v >= 12 for v in inlier_values)}/{len(inlier_values)}")
print(f"Vídeo: {args.output_video}")
