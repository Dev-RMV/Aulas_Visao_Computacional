# Guia do professor

Sugestão de condução em quatro momentos:

Aula 1 — Fundamentos do detector
- Use os exemplos 01 a 10.
- Mostre que detecção não é apenas desenhar caixas: envolve confiança, classe, NMS, FPS e latência.

Aula 2 — Modelos reais
- Use os exemplos 11 a 20.
- Demonstre YOLOv8 via Ultralytics e SSD/YOLO via OpenCV DNN quando os arquivos estiverem disponíveis.
- Peça aos alunos para preencherem uma tabela com FPS, latência, tamanho em disco e observações visuais.

Aula 3 — Rastreamento
- Use os exemplos 21 a 30.
- Mostre por que detecção frame a frame não garante identidade persistente.
- Trabalhe IoU, associação entre frames, trilhas e contagem de entradas/saídas.

Aula 4 — Análise e responsabilidade
- Use os exemplos 31 a 35 e o notebook.
- Discuta limitações técnicas, erros, ID switches, privacidade, viés, sinalização e minimização de dados.

Perguntas úteis:
1. Qual a diferença entre detectar e rastrear?
2. Por que aplicar Non-Maximum Suppression?
3. Um FPS maior sempre é melhor?
4. O que acontece quando dois objetos se cruzam?
5. Por que ID switches são problemáticos em contagem?
6. Em quais cenários SSD ainda pode ser interessante?
7. O que deve ser medido no próprio ambiente de operação?
8. Quais cuidados éticos são necessários ao usar drones em áreas urbanas?
