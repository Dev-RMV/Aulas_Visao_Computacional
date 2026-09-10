from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 14 - Rodar pipeline em um arquivo de video.
Uso: python exemplos/14_video_arquivo_caffe_fps.py --video data/video.mp4
"""
import argparse, cv2, time
from utils.face_utils import detect_faces_haar, crop_face, load_caffe_age_gender, predict_age_gender, draw_label
parser = argparse.ArgumentParser(); parser.add_argument("--video", required=True)
args = parser.parse_args()
age_net, gender_net = load_caffe_age_gender()
cap = cv2.VideoCapture(args.video)
frames, t0 = 0, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok: break
    for box in detect_faces_haar(frame):
        pred = predict_age_gender(crop_face(frame, box), age_net, gender_net)
        draw_label(frame, box, f"{pred.gender} | {pred.age}")
    frames += 1
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
fps = frames / max(1e-6, time.perf_counter() - t0)
print(f"Frames processados: {frames}")
print(f"FPS medio: {fps:.2f}")
