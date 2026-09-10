import argparse

import cv2

from utils import RECURSOS, SAIDAS, destacar_roi, segmentar_verde


parser = argparse.ArgumentParser(description="Extrai ROI, bounding box e overlay semitransparente.")
parser.add_argument("--sem-janelas", action="store_true")
args = parser.parse_args()

imagem = cv2.imread(str(RECURSOS / "alvo_hsv.png"))
if imagem is None:
    raise FileNotFoundError("Execute primeiro: python gerar_recursos.py")
_, limpa = segmentar_verde(imagem)
saida, bbox, proporcao = destacar_roi(imagem, limpa)
if bbox is None:
    print("Nenhuma ROI válida encontrada.")
else:
    x, y, w, h = bbox
    roi = imagem[y : y + h, x : x + w]
    cv2.imwrite(str(SAIDAS / "08_roi_recortada.png"), roi)
    print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
    print(f"Proporção da área da ROI em relação ao frame: {proporcao:.4f} ({proporcao:.2%})")
cv2.imwrite(str(SAIDAS / "08_roi_overlay.png"), saida)

if not args.sem_janelas:
    cv2.imshow("ROI destacada", saida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

