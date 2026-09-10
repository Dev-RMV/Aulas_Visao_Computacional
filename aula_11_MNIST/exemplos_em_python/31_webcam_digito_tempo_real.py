"""
Exemplo 31 - Inferência por webcam com OpenCV.

Objetivo:
- Demonstrar uso em captura real, aproximando o exercício de percepção robótica.

Controles:
- pressione ESPAÇO para capturar o ROI central e classificar;
- pressione ESC para sair.
"""

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from mnist_utils import build_cnn, load_mnist

model_path = Path("resultados/cnn_mnist.keras")
if model_path.exists():
    model = tf.keras.models.load_model(model_path)
else:
    (x_train, y_train), _ = load_mnist(flatten=False)
    model = build_cnn()
    model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
    model_path.parent.mkdir(exist_ok=True)
    model.save(model_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Não foi possível abrir a câmera.")

prediction_text = "Aperte ESPAÇO"
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    size = min(h, w) // 2
    x1 = w // 2 - size // 2
    y1 = h // 2 - size // 2
    x2 = x1 + size
    y2 = y1 + size

    display = frame.copy()
    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(display, prediction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam - digito", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == 32:
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            binary = 255 - binary
        resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
        x = resized.astype("float32").reshape(1, 28, 28, 1) / 255.0
        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        prediction_text = f"Predito: {pred} ({probs[pred]:.1%})"

cap.release()
cv2.destroyAllWindows()
