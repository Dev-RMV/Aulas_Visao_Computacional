"""Exemplo 9 — compara resolução e salto de frames para um alvo embarcado."""

from __future__ import annotations

import time

import cv2
import face_recognition
import numpy as np

from utils import RESOURCES


EXPERIMENTS = [
    ("qualidade", 1.00, 1),
    ("equilibrado", 0.50, 1),
    ("economico", 0.50, 2),
    ("minimo", 0.25, 2),
]


def benchmark(scale: float, every: int, max_frames: int = 80) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(RESOURCES / "video_teste.mp4"))
    times, processed, read = [], 0, 0
    while read < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if read % every == 0:
            start = time.perf_counter()
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model="hog")
            face_recognition.face_encodings(rgb, locations)
            times.append((time.perf_counter() - start) * 1000)
            processed += 1
        read += 1
    cap.release()
    mean_ms = float(np.mean(times))
    effective_fps = 1000 / mean_ms * every
    return mean_ms, effective_fps, processed


print("perfil       escala  a_cada  latência(ms)  FPS efetivo aproximado")
for name, scale, every in EXPERIMENTS:
    mean_ms, fps, count = benchmark(scale, every)
    print(f"{name:<12} {scale:<6.2f} {every:<7} {mean_ms:>10.2f} {fps:>21.1f}  n={count}")

print("\nInterpretação: reduzir a imagem diminui pixels; pular frames reduz chamadas de inferência.")
print("Em controle robótico, verifique também a idade máxima aceitável da última identidade.")
