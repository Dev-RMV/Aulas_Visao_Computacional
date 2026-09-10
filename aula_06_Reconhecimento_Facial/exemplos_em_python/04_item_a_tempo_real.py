"""Item A completo — Haar em vídeo/webcam, bounding box e ROI 48×48.

Trade-off testado no curso:
- 'sensivel' (scaleFactor=1.05, minNeighbors=3): pirâmide mais densa e menor
  exigência de consenso. Tende a aumentar a taxa de detecção, mas também pode
  aceitar mais falsos positivos e custa mais CPU.
- 'especifica' (scaleFactor=1.20, minNeighbors=6): busca mais espaçada e exige
  mais retângulos vizinhos. Tende a rejeitar falsos positivos, mas pode perder
  rostos pequenos, inclinados ou com iluminação difícil.

Em robôs/drones, escolha os parâmetros a partir do custo do erro: para evitar
colisão/acompanhar uma pessoa pode-se priorizar sensibilidade; para liberar uma
porta, um falso aceite é mais grave e a decisão deve ser mais específica.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from utils import RESOURCES, ensure_outputs, load_ground_truth, load_haar, open_source, put_label


CONFIGS = {
    "sensivel": dict(scaleFactor=1.05, minNeighbors=3, minSize=(35, 35)),
    "especifica": dict(scaleFactor=1.20, minNeighbors=6, minSize=(50, 50)),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(RESOURCES / "video_teste.mp4"), help="0 ou caminho de vídeo")
    p.add_argument("--config", choices=CONFIGS, default="sensivel")
    p.add_argument("--headless", action="store_true", help="não abre janela; útil em servidor")
    p.add_argument("--max-frames", type=int, default=0, help="0 processa até o fim")
    p.add_argument("--save-every", type=int, default=1, help="1 salva cada ROI detectada")
    p.add_argument("--output-video", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detector = load_haar()
    cap = open_source(args.source)
    truth = load_ground_truth() if not args.source.isdigit() else {}
    capture_dir = ensure_outputs() / "capturas_48x48" / args.config
    capture_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    frame_index = saved = detected_total = expected_total = matched_total = false_positive_total = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_index >= args.max_frames):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = detector.detectMultiScale(gray, **CONFIGS[args.config])
        detected_total += len(faces)

        expected = truth.get(frame_index)
        if expected is not None:
            expected_total += expected
            matched_total += min(len(faces), expected)
            false_positive_total += max(0, len(faces) - expected)

        for face_index, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (38, 226, 167), 2)
            put_label(frame, f"face {face_index + 1}", (x, max(28, y)))
            if frame_index % max(1, args.save_every) == 0:
                roi_48 = cv2.resize(gray[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA)
                saved += 1
                cv2.imwrite(str(capture_dir / f"face_{saved:05d}.png"), roi_48)

        if expected_total:
            rate = matched_total / expected_total * 100
            status = f"{args.config} | detecao~ {rate:5.1f}% | FP visiveis~ {false_positive_total}"
        else:
            status = f"{args.config} | detectados: {len(faces)}"
        put_label(frame, status, (15, 30))

        if args.output_video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(frame)
        if not args.headless:
            cv2.imshow("Item A - Haar", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    rate = matched_total / expected_total * 100 if expected_total else float("nan")
    print(f"Configuração: {args.config} {CONFIGS[args.config]}")
    print(f"Frames processados: {frame_index}")
    print(f"ROI 48x48 salvas: {saved} em {capture_dir}")
    print(f"Falsos positivos visíveis estimados: {false_positive_total}")
    print(f"Taxa de detecção estimada: {rate:.1f}%")


if __name__ == "__main__":
    main()
