"""Exemplo 01 — cria um par planar com perspectiva, escala e iluminação distintas."""

from __future__ import annotations

import json

import cv2
import numpy as np

from utils import RESOURCES


WIDTH, HEIGHT = 960, 720


def finder_pattern(image, x: int, y: int, size: int = 78) -> None:
    cv2.rectangle(image, (x, y), (x + size, y + size), (15, 15, 15), -1)
    cv2.rectangle(image, (x + 10, y + 10), (x + size - 10, y + size - 10), (245, 245, 245), -1)
    cv2.rectangle(image, (x + 24, y + 24), (x + size - 24, y + size - 24), (15, 15, 15), -1)


def make_reference() -> np.ndarray:
    rng = np.random.default_rng(42)
    image = np.full((HEIGHT, WIDTH, 3), (238, 242, 245), np.uint8)
    cv2.rectangle(image, (45, 40), (915, 680), (24, 30, 36), 5)
    cv2.rectangle(image, (66, 61), (894, 659), (250, 250, 247), -1)

    # Tabuleiro: muitos cantos em duas direções.
    cell = 44
    for row in range(6):
        for col in range(8):
            color = (25, 25, 25) if (row + col) % 2 == 0 else (242, 242, 242)
            x, y = 100 + col * cell, 170 + row * cell
            cv2.rectangle(image, (x, y), (x + cell, y + cell), color, -1)
    cv2.rectangle(image, (98, 168), (100 + 8 * cell + 2, 170 + 6 * cell + 2), (0, 112, 215), 4)

    # Regiões assimétricas evitam ambiguidades globais.
    finder_pattern(image, 760, 105)
    finder_pattern(image, 760, 505)
    finder_pattern(image, 535, 315, 68)
    cv2.circle(image, (595, 190), 63, (38, 226, 167), -1)
    cv2.circle(image, (595, 190), 39, (21, 27, 33), 5)
    cv2.line(image, (560, 190), (630, 190), (21, 27, 33), 4)
    cv2.line(image, (595, 155), (595, 225), (21, 27, 33), 4)

    triangle = np.array([[690, 310], [820, 360], [720, 455]], np.int32)
    cv2.fillConvexPoly(image, triangle, (61, 141, 255))
    cv2.polylines(image, [triangle], True, (15, 20, 28), 5)
    cv2.putText(image, "ROBOTICS LAB 07", (245, 128), cv2.FONT_HERSHEY_DUPLEX, 1.55, (20, 25, 31), 3, cv2.LINE_AA)
    cv2.putText(image, "SIFT  ORB  AKAZE", (250, 625), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (20, 25, 31), 3, cv2.LINE_AA)
    cv2.putText(image, "LOCAL FEATURES", (515, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (90, 95, 102), 2, cv2.LINE_AA)

    # Constelação determinística de cruzes e círculos pequenos.
    for i, (x, y) in enumerate(rng.integers([85, 80], [880, 645], size=(95, 2))):
        if 90 < x < 470 and 155 < y < 455:
            continue
        color = (30, 30, 30) if i % 3 else (180, 55, 48)
        if i % 2:
            cv2.drawMarker(image, (int(x), int(y)), color, cv2.MARKER_CROSS, 10, 2)
        else:
            cv2.circle(image, (int(x), int(y)), 4, color, -1)
    return image


def transform(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.float32([[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]])
    target = np.float32([[92, 74], [875, 25], [925, 648], [44, 698]])
    homography = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(reference, homography, (WIDTH, HEIGHT), borderValue=(35, 40, 46))

    # Alteração fotométrica: gradiente de iluminação, contraste e ruído.
    gradient = np.linspace(0.72, 1.03, WIDTH, dtype=np.float32)[None, :, None]
    adjusted = np.clip(warped.astype(np.float32) * gradient + 12, 0, 255)
    noise = np.random.default_rng(7).normal(0, 2.5, adjusted.shape).astype(np.float32)
    adjusted = np.clip(adjusted + noise, 0, 255).astype(np.uint8)
    adjusted = cv2.GaussianBlur(adjusted, (3, 3), 0.45)
    cv2.rectangle(adjusted, (735, 560), (900, 655), (48, 52, 58), -1)  # pequena oclusão
    cv2.putText(adjusted, "OCLUSAO", (758, 615), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 185, 190), 2, cv2.LINE_AA)
    return adjusted, homography


def make_video(reference: np.ndarray) -> None:
    writer = cv2.VideoWriter(str(RESOURCES / "video_localizacao.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Falha ao criar vídeo")
    src = np.float32([[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]])
    for frame_index in range(120):
        t = frame_index / 119.0
        dx, dy = 28 * np.sin(t * 2 * np.pi), 20 * np.cos(t * 2 * np.pi)
        dst = np.float32([
            [78 + dx, 60 + dy], [875 - dx, 34 + dy * 0.4],
            [920 - dx * 0.3, 650 - dy], [48 + dx * 0.4, 690 - dy * 0.5],
        ])
        h = cv2.getPerspectiveTransform(src, dst)
        frame = cv2.warpPerspective(reference, h, (WIDTH, HEIGHT), borderValue=(30, 36, 42))
        alpha = 0.82 + 0.15 * np.sin(t * 4 * np.pi)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=10)
        writer.write(frame)
    writer.release()


def main() -> None:
    RESOURCES.mkdir(parents=True, exist_ok=True)
    reference = make_reference()
    transformed, homography = transform(reference)
    cv2.imwrite(str(RESOURCES / "cena_referencia.png"), reference)
    cv2.imwrite(str(RESOURCES / "cena_transformada.png"), transformed)
    pair = np.hstack([reference, transformed])
    cv2.imwrite(str(RESOURCES / "par_imagens.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 94])
    with (RESOURCES / "homografia_real.json").open("w", encoding="utf-8") as f:
        json.dump({"H_referencia_para_transformada": homography.tolist()}, f, indent=2)
    make_video(reference)
    print(f"Recursos criados em: {RESOURCES}")
    print("Transformações: perspectiva + escala + rotação aparente + iluminação + ruído + oclusão.")


if __name__ == "__main__":
    main()
