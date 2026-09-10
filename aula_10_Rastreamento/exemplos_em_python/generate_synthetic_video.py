import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Gera vídeo sintético para testes de rastreamento.")
    parser.add_argument("--output", default="data/synthetic_motion.mp4")
    parser.add_argument("--frames", type=int, default=260)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))

    rng = np.random.default_rng(7)

    for t in range(args.frames):
        frame = np.full((args.height, args.width, 3), 35, dtype=np.uint8)

        # Fundo com textura leve para simular ruído real de câmera.
        noise = rng.normal(0, 7, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Linhas fixas no fundo: ajudam a perceber a diferença entre fundo e movimento.
        for x in range(0, args.width, 80):
            cv2.line(frame, (x, 0), (x, args.height), (55, 55, 55), 1)
        for y in range(0, args.height, 60):
            cv2.line(frame, (0, y), (args.width, y), (55, 55, 55), 1)

        # Objeto principal: círculo vermelho, ideal para CamShift em HSV.
        cx = 60 + int(2.0 * t)
        cy = 145 + int(45 * np.sin(t / 25))
        cv2.circle(frame, (cx % args.width, cy), 28, (35, 35, 220), -1)
        cv2.circle(frame, (cx % args.width, cy), 28, (255, 255, 255), 2)

        # Segundo objeto: retângulo azul para testar contagem por background subtraction.
        rx = args.width - 90 - int(1.25 * t)
        ry = 230 + int(25 * np.cos(t / 20))
        cv2.rectangle(frame, (rx % args.width, ry), ((rx + 70) % args.width, ry + 50), (210, 80, 30), -1)

        # Oclusão parcial temporária.
        if 110 < t < 145:
            cv2.rectangle(frame, (260, 95), (340, 220), (20, 20, 20), -1)

        cv2.putText(frame, f"frame {t}", (12, args.height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)
        writer.write(frame)

    writer.release()
    print(f"Vídeo sintético salvo em: {out_path}")


if __name__ == "__main__":
    main()
