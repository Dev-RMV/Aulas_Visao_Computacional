from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 32 - Pipeline final do Item A: imagem/video/webcam + FPS + rotulos.
Para webcam: python exemplos/32_pipeline_final_item_a.py --fonte 0
Para video:  python exemplos/32_pipeline_final_item_a.py --fonte data/video.mp4
"""
import argparse, cv2, time
from utils.face_utils import detect_faces_haar, crop_face, load_caffe_age_gender, predict_age_gender, draw_label
parser = argparse.ArgumentParser(); parser.add_argument("--fonte", default="0")
args = parser.parse_args()
source = int(args.fonte) if args.fonte.isdigit() else args.fonte
age_net, gender_net = load_caffe_age_gender()
cap = cv2.VideoCapture(source)
frames, t0 = 0, time.perf_counter()
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    for box in detect_faces_haar(frame):
        pred = predict_age_gender(crop_face(frame, box), age_net, gender_net)
        draw_label(frame, box, f"{pred.gender} | {pred.age}")
    frames += 1
    fps = frames / max(1e-6, time.perf_counter() - t0)
    cv2.putText(frame, f"FPS medio: {fps:.1f}", (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.imshow("Item A - Pipeline final", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
print(f"FPS medio final: {fps:.2f}")
