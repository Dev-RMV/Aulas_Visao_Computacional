from __future__ import annotations

import argparse
import time

import cv2

from utils import RECURSOS, SAIDAS, destacar_roi, segmentar_verde


parser = argparse.ArgumentParser(description="Item B completo em vídeo ou webcam.")
parser.add_argument("--fonte", default=str(RECURSOS / "video_alvo.mp4"), help="Arquivo de vídeo ou índice da câmera, por exemplo 0")
parser.add_argument("--sem-janelas", action="store_true")
parser.add_argument("--limite-frames", type=int, default=0, help="0 processa até o fim")
args = parser.parse_args()

fonte = int(args.fonte) if args.fonte.isdigit() else args.fonte
cap = cv2.VideoCapture(fonte)
if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir a fonte: {args.fonte}")

fps_entrada = cap.get(cv2.CAP_PROP_FPS) or 30.0
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    str(SAIDAS / "09_item_b_processado.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_entrada,
    (largura, altura),
)

contador = 0
inicio = time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # (1) range HSV, (2) erode + dilate
    _, mascara_limpa = segmentar_verde(frame)
    # (3) bounding box + máscara colorida semitransparente
    visual, bbox, proporcao = destacar_roi(frame, mascara_limpa)
    # (4) proporção da área da ROI em relação ao frame total
    print(f"frame={contador:04d} proporcao_roi={proporcao:.6f} ({proporcao:.2%})")

    writer.write(visual)
    if contador == 45:
        cv2.imwrite(str(SAIDAS / "09_item_b_frame_referencia.png"), visual)
        cv2.imwrite(str(SAIDAS / "09_item_b_mascara_referencia.png"), mascara_limpa)

    contador += 1
    if not args.sem_janelas:
        cv2.imshow("Item B - ROI em tempo real", visual)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    if args.limite_frames and contador >= args.limite_frames:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
duracao = max(time.perf_counter() - inicio, 1e-9)
print(f"Resumo: {contador} frames em {duracao:.2f}s; desempenho médio={contador/duracao:.1f} FPS")

