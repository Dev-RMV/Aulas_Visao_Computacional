from __future__ import annotations

import csv
from pathlib import Path

import cv2


PROJECT = Path(__file__).resolve().parents[1]
RESOURCES = PROJECT / "recursos"
OUTPUTS = PROJECT / "saidas"


def load_haar() -> cv2.CascadeClassifier:
    """Carrega a cópia local do Haar; usa a instalação do OpenCV como reserva."""
    local = RESOURCES / "haarcascade_frontalface_default.xml"
    path = local if local.exists() else Path(cv2.data.haarcascades) / local.name
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise RuntimeError(f"Falha ao carregar o classificador Haar: {path}")
    return cascade


def open_source(source: str) -> cv2.VideoCapture:
    """Aceita '0' para webcam ou um caminho de vídeo."""
    parsed = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(parsed)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {source}")
    return cap


def load_ground_truth() -> dict[int, int]:
    path = RESOURCES / "gabarito_video.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {int(row["frame"]): int(row["rostos_esperados"]) for row in csv.DictReader(f)}


def put_label(frame, text: str, origin: tuple[int, int], color=(38, 226, 167)) -> None:
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y + 4), (20, 25, 31), -1)
    cv2.putText(frame, text, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)


def ensure_outputs() -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS
