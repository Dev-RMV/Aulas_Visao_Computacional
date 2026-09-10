from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from utils import RECURSOS, SAIDAS


parser = argparse.ArgumentParser(description="Item B avançado: trackbars, múltiplas ROIs e FPS.")
parser.add_argument("--fonte", default=str(RECURSOS / "video_alvo.mp4"))
parser.add_argument("--sem-janelas", action="store_true")
parser.add_argument("--limite-frames", type=int, default=0)
args = parser.parse_args()

fonte = int(args.fonte) if args.fonte.isdigit() else args.fonte
cap = cv2.VideoCapture(fonte)
if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir {args.fonte}")

if not args.sem_janelas:
    cv2.namedWindow("Avancado")
    for nome, valor, maximo in [("H min", 35, 179), ("H max", 85, 179), ("S min", 90, 255), ("V min", 70, 255)]:
        cv2.createTrackbar(nome, "Avancado", valor, maximo, lambda _x: None)

contador = 0
t0 = time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if args.sem_janelas:
        hmin, hmax, smin, vmin = 35, 85, 90, 70
    else:
        hmin = cv2.getTrackbarPos("H min", "Avancado")
        hmax = cv2.getTrackbarPos("H max", "Avancado")
        smin = cv2.getTrackbarPos("S min", "Avancado")
        vmin = cv2.getTrackbarPos("V min", "Avancado")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([hmin, smin, vmin]), np.array([hmax, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 255)
    visual = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    area_total_bbox = 0
    rois = 0
    for c in sorted(contornos, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < 500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        area_total_bbox += w * h
        rois += 1
        cv2.rectangle(visual, (x, y), (x + w, y + h), (255, 255, 255), 2)

    proporcao = area_total_bbox / float(frame.shape[0] * frame.shape[1])
    fps = (contador + 1) / max(time.perf_counter() - t0, 1e-9)
    cv2.putText(visual, f"ROIs={rois} area={proporcao:.2%} FPS={fps:.1f}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    print(f"frame={contador:04d} rois={rois} proporcao_total={proporcao:.6f} fps={fps:.1f}")

    if contador == 75:
        cv2.imwrite(str(SAIDAS / "10_item_b_avancado.png"), visual)
    contador += 1
    if not args.sem_janelas:
        cv2.imshow("Avancado", visual)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    if args.limite_frames and contador >= args.limite_frames:
        break

cap.release()
cv2.destroyAllWindows()

