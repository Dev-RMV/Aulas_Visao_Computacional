"""Exemplo 3 — extrair a ROI facial, normalizar para 48×48 e salvar numerada."""

import cv2

from utils import RESOURCES, ensure_outputs, load_haar


frame = cv2.imread(str(RESOURCES / "foto_grupo.jpg"))
if frame is None:
    raise RuntimeError("Imagem de grupo não encontrada")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = load_haar().detectMultiScale(gray, 1.08, 4, minSize=(45, 45))
folder = ensure_outputs() / "capturas_48x48"
folder.mkdir(parents=True, exist_ok=True)

for number, (x, y, w, h) in enumerate(faces, start=1):
    roi = gray[y:y + h, x:x + w]
    roi_48 = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
    filename = folder / f"face_{number:03d}.png"
    cv2.imwrite(str(filename), roi_48)
    print(f"{filename.name}: shape={roi_48.shape}, min={roi_48.min()}, max={roi_48.max()}")

print(f"Total salvo: {len(faces)} ROI(s) em {folder}")
