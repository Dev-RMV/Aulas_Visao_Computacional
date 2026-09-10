"""Item B completo — reconhecimento em vídeo/webcam e latência por frame.

Cuidado ético: biometria não é apenas uma variável técnica. Em robôs e drones,
o reconhecimento pode permitir vigilância e rastreamento sem consentimento;
erros e diferenciais de desempenho podem atingir grupos de forma desigual.
Uma implantação responsável exige finalidade legítima e explícita, minimização
e proteção dos dados, prazo de retenção, auditoria, teste por subgrupos e revisão
humana. Uma correspondência biométrica não deve disparar punição autônoma.
"""

from __future__ import annotations

import argparse
import pickle
import time

import cv2
import face_recognition
import numpy as np

from utils import RESOURCES, ensure_outputs, open_source, put_label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(RESOURCES / "video_teste.mp4"))
    p.add_argument("--tolerance", type=float, default=0.50)
    p.add_argument("--scale", type=float, default=0.50, help="0.5 reduz largura e altura à metade")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--output-video", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    database_path = ensure_outputs() / "encodings.pkl"
    if not database_path.exists():
        raise RuntimeError("Execute primeiro: python 06_cadastrar_identidades.py")
    with database_path.open("rb") as f:
        database = pickle.load(f)

    cap = open_source(args.source)
    writer = None
    latencies_ms: list[float] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_idx >= args.max_frames):
            break
        start = time.perf_counter()

        small = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        vectors = face_recognition.face_encodings(rgb, locations)
        predictions = []
        for vector in vectors:
            distances = face_recognition.face_distance(database["encodings"], vector)
            best = int(np.argmin(distances))
            name = database["names"][best] if distances[best] <= args.tolerance else "Desconhecido"
            predictions.append((name, float(distances[best])))

        latency = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency)
        inv = 1.0 / args.scale
        for (top, right, bottom, left), (name, distance) in zip(locations, predictions):
            top, right, bottom, left = [int(v * inv) for v in (top, right, bottom, left)]
            color = (38, 226, 167) if name != "Desconhecido" else (84, 91, 231)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            put_label(frame, f"{name} d={distance:.3f}", (left, max(28, top)), color)
        put_label(frame, f"inferência {latency:.1f} ms | média {np.mean(latencies_ms):.1f} ms", (15, 30))

        if args.output_video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(frame)
        if not args.headless:
            cv2.imshow("Item B - reconhecimento facial", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    values = np.asarray(latencies_ms)
    print(f"Frames processados: {len(values)}")
    print(f"Latência média de inferência: {values.mean():.2f} ms/frame")
    print(f"Latência p95: {np.percentile(values, 95):.2f} ms/frame")
    print(f"FPS teórico da inferência: {1000 / values.mean():.1f}")


if __name__ == "__main__":
    main()
