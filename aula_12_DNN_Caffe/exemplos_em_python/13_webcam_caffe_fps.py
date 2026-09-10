from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 13 - Pipeline em tempo real: Haar + Caffe + FPS medio.
Pressione Q para sair.
"""
import cv2, time
from utils.face_utils import detect_faces_haar, crop_face, load_caffe_age_gender, predict_age_gender, draw_label
age_net, gender_net = load_caffe_age_gender()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Nao foi possivel abrir a webcam.")
frames, t0 = 0, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok: break
    for box in detect_faces_haar(frame):
        pred = predict_age_gender(crop_face(frame, box), age_net, gender_net)
        draw_label(frame, box, f"{pred.gender} | {pred.age}")
    frames += 1
    fps = frames / max(1e-6, time.perf_counter() - t0)
    cv2.putText(frame, f"FPS medio: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.imshow("Idade e genero - OpenCV DNN", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
print(f"FPS medio: {fps:.2f}")
