from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECURSOS = ROOT / "recursos"
SAIDAS = ROOT / "saidas"
SAIDAS.mkdir(exist_ok=True)


def carregar_par() -> tuple[np.ndarray, np.ndarray]:
    esquerda = cv2.imread(str(RECURSOS / "estereo_esquerda.png"))
    direita = cv2.imread(str(RECURSOS / "estereo_direita.png"))
    if esquerda is None or direita is None:
        raise FileNotFoundError("Execute primeiro: python gerar_recursos.py")
    if esquerda.shape != direita.shape:
        raise ValueError("As imagens do par estéreo devem ter a mesma resolução.")
    return esquerda, direita


def criar_sgbm(num_disparidades: int = 96, bloco: int = 5) -> cv2.StereoSGBM:
    if num_disparidades % 16 != 0:
        raise ValueError("num_disparidades deve ser múltiplo de 16.")
    canais = 1
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparidades,
        blockSize=bloco,
        P1=8 * canais * bloco**2,
        P2=32 * canais * bloco**2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def calcular_disparidade_sgbm(esquerda: np.ndarray, direita: np.ndarray) -> np.ndarray:
    cinza_e = cv2.cvtColor(esquerda, cv2.COLOR_BGR2GRAY)
    cinza_d = cv2.cvtColor(direita, cv2.COLOR_BGR2GRAY)
    return criar_sgbm().compute(cinza_e, cinza_d).astype(np.float32) / 16.0


def normalizar_disparidade(
    disparidade: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Remove inválidos, bordas sem correspondência e valores localmente incoerentes.
    mediana_local = cv2.medianBlur(disparidade, 5)
    validos = (
        np.isfinite(disparidade)
        & (disparidade > 0.5)
        & (np.abs(disparidade - mediana_local) < 1.5)
    )
    h, w = disparidade.shape
    validos[:10, :] = False
    validos[h - 10 :, :] = False
    validos[:, : min(100, w // 4)] = False
    validos[:, w - 10 :] = False
    if not np.any(validos):
        raise RuntimeError("Nenhuma disparidade válida foi encontrada.")

    d_validos = disparidade[validos]
    # Percentis robustos evitam que um único falso casamento defina toda a escala.
    d_min, d_max = np.percentile(d_validos, [10, 90]).astype(float)
    proximidade = np.zeros_like(disparidade, dtype=np.float32)
    d_limitada = np.clip(d_validos, d_min, d_max)
    proximidade[validos] = (d_limitada - d_min) / max(d_max - d_min, 1e-6)

    inversa = np.zeros_like(disparidade, dtype=np.float32)
    inversa[validos] = 1.0 / np.maximum(d_limitada, 1e-6)
    inv_validos = inversa[validos]
    i_min, i_max = float(inv_validos.min()), float(inv_validos.max())
    profundidade_rel = np.zeros_like(disparidade, dtype=np.float32)
    profundidade_rel[validos] = (inv_validos - i_min) / max(i_max - i_min, 1e-6)
    return proximidade, profundidade_rel, validos


def pixel_extremo_robusto(
    disparidade: np.ndarray, validos: np.ndarray, percentil: float, tolerancia: float
) -> tuple[int, int]:
    """Retorna (x,y) no maior componente próximo do percentil solicitado."""
    alvo = float(np.percentile(disparidade[validos], percentil))
    faixa = (validos & (np.abs(disparidade - alvo) <= tolerancia)).astype(np.uint8)
    n, rotulos, stats, centroides = cv2.connectedComponentsWithStats(faixa)
    if n <= 1:
        y, x = np.unravel_index(np.argmin(np.where(validos, np.abs(disparidade - alvo), np.inf)), disparidade.shape)
        return int(x), int(y)
    componente = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centroides[componente]
    ys, xs = np.where(rotulos == componente)
    indice = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return int(xs[indice]), int(ys[indice])


def colorir_jet(mapa_01: np.ndarray) -> np.ndarray:
    oito_bits = np.clip(mapa_01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(oito_bits, cv2.COLORMAP_JET)


def maior_contorno(mask: np.ndarray, area_minima: float = 400.0):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [c for c in contornos if cv2.contourArea(c) >= area_minima]
    return max(contornos, key=cv2.contourArea) if contornos else None


def segmentar_verde(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara_bruta = cv2.inRange(hsv, np.array([35, 90, 70]), np.array([85, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    erodida = cv2.erode(mascara_bruta, kernel, iterations=1)
    limpa = cv2.dilate(erodida, kernel, iterations=2)
    return mascara_bruta, limpa


def destacar_roi(frame: np.ndarray, mask: np.ndarray, area_minima: float = 400.0):
    saida = frame.copy()
    contorno = maior_contorno(mask, area_minima)
    if contorno is None:
        return saida, None, 0.0

    x, y, w, h = cv2.boundingRect(contorno)
    overlay = saida.copy()
    overlay[mask > 0] = (0, 255, 255)
    saida = cv2.addWeighted(overlay, 0.38, saida, 0.62, 0)
    cv2.rectangle(saida, (x, y), (x + w, y + h), (255, 255, 255), 2)
    proporcao_bbox = (w * h) / float(frame.shape[0] * frame.shape[1])
    cv2.putText(
        saida,
        f"ROI/frame: {proporcao_bbox:.2%}",
        (x, max(24, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return saida, (x, y, w, h), proporcao_bbox
