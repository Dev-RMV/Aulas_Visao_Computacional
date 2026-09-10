# Aula prática: rastreamento clássico de objetos em movimento

Este pacote acompanha a aula em PowerPoint sobre rastreamento clássico usando OpenCV.
Ele cobre:

- Background subtraction com MOG2 e KNN;
- contornos, limpeza morfológica e contagem de objetos;
- CamShift com seleção de ROI;
- Filtro de Kalman com medição, predição e suavização;
- registro de métricas em CSV e gráfico 2D de trajetórias.

## Instalação

```bash
pip install -r requirements.txt
```

## Fonte de vídeo

Todos os exemplos aceitam `--source`:

```bash
python 04_compare_mog2_knn.py --source 0
python 04_compare_mog2_knn.py --source data/synthetic_motion.mp4
```

Use `0` para webcam. Também foi incluído um vídeo sintético em `data/synthetic_motion.mp4`.

## Ordem sugerida

1. `00_check_environment.py`
2. `01_webcam_viewer_metadata.py`
3. `generate_synthetic_video.py`
4. `02_background_mog2_basic.py`
5. `03_background_knn_basic.py`
6. `04_compare_mog2_knn.py`
7. `05_camshift_roi.py`
8. `06_camshift_kalman.py`
9. `07_parameter_sweep_background.py`
10. `08_logging_to_csv.py`
11. `09_plot_trajectories_csv.py`
12. `10_full_lab_pipeline.py`

## Teclas comuns

- `q`: sair
- `p`: pausar/despausar em alguns exemplos
- `r`: reiniciar ROI em exemplos com CamShift

## Observação sobre ROI no vídeo sintético

Para rastrear automaticamente o círculo vermelho do vídeo sintético, use:

```bash
python 06_camshift_kalman.py --source data/synthetic_motion.mp4 --roi 70 120 80 80
```

A ROI é informada como: `x y w h`.
