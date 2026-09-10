import argparse

import cv2
import numpy as np

from utils import (
    SAIDAS,
    calcular_disparidade_sgbm,
    carregar_par,
    colorir_jet,
    normalizar_disparidade,
    pixel_extremo_robusto,
)


parser = argparse.ArgumentParser(description="Item A completo: disparidade e profundidade relativa.")
parser.add_argument("--sem-janelas", action="store_true")
args = parser.parse_args()

esquerda, direita = carregar_par()
disparidade = calcular_disparidade_sgbm(esquerda, direita)
proximidade, profundidade_rel, validos = normalizar_disparidade(disparidade)

x_perto, y_perto = pixel_extremo_robusto(disparidade, validos, percentil=90, tolerancia=1.5)
x_longe, y_longe = pixel_extremo_robusto(disparidade, validos, percentil=10, tolerancia=0.75)

print("Convenção: proximidade 0=longe e 1=perto; profundidade relativa 0=perto e 1=longe")
print("Extremos robustos: percentis 10 e 90 em componentes conexos, rejeitando outliers isolados.")
print(
    f"MAIS PRÓXIMO: (x={x_perto}, y={y_perto}), "
    f"disparidade={disparidade[y_perto, x_perto]:.2f}, "
    f"proximidade={proximidade[y_perto, x_perto]:.3f}, "
    f"profundidade_rel={profundidade_rel[y_perto, x_perto]:.3f}"
)
print(
    f"MAIS DISTANTE: (x={x_longe}, y={y_longe}), "
    f"disparidade={disparidade[y_longe, x_longe]:.2f}, "
    f"proximidade={proximidade[y_longe, x_longe]:.3f}, "
    f"profundidade_rel={profundidade_rel[y_longe, x_longe]:.3f}"
)

mapa = colorir_jet(proximidade)
mapa[~validos] = 0
cv2.drawMarker(mapa, (int(x_perto), int(y_perto)), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
cv2.drawMarker(mapa, (int(x_longe), int(y_longe)), (0, 0, 0), cv2.MARKER_CROSS, 24, 2)
cv2.imwrite(str(SAIDAS / "04_item_a_mapa_jet.png"), mapa)
cv2.imwrite(str(SAIDAS / "04_profundidade_relativa.png"), (profundidade_rel * 255).astype(np.uint8))

# Em um drone, a disparidade maior sugere obstáculo mais próximo e pode alimentar
# a evitação reativa. Porém, vibração, motion blur, rolling shutter, baixa textura,
# mudança de iluminação e atraso de processamento degradam o mapa. Antes de obter
# distância métrica Z=f*B/d, o par precisa ser calibrado e retificado; na prática,
# também convém filtrar a profundidade no tempo e rejeitar pixels inválidos.

if not args.sem_janelas:
    cv2.imshow("Item A - mapa de disparidade colorido", mapa)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
