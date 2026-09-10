from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 05 - Detecção facial pela webcam.
Pressione Q para sair.
"""
import cv2, time
from utils.face_utils import detect_faces_haar, draw_label
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Nao foi possivel abrir a webcam.")
frames, t0 = 0, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok: break
    faces = detect_faces_haar(frame)
    for box in faces:
        draw_label(frame, box, "Rosto")
    frames += 1
    fps = frames / max(1e-6, (time.perf_counter() - t0))
    cv2.putText(frame, f"FPS medio: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.imshow("Haar Cascade - Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
print(f"FPS medio: {fps:.2f}")
