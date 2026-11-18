# Modelo

O arquivo `best_v3_merge_V2.onnx` (~99 MB) **não fica no código do repositório**. Ele está publicado na aba **Releases**.

| Item | Valor |
|---|---|
| Formato | ONNX (opset 20) |
| Treinado com | Ultralytics 8.4.79 |
| Tarefa | Detecção (caixas) |
| Classes | `0: Tree` |
| Entrada | `images` — `[1, 3, 640, 640]`, float32, RGB normalizado 0–1 |
| Saída | `output0` — `[1, 5, 8400]` (cx, cy, w, h, confiança), sem NMS embutido |
| Arquitetura | YOLOv8m (pré-treinado no COCO, fine-tuning com 407 imagens próprias) |
| Parâmetros | ~25,9 milhões |
| Resolução de uso | 3,5 cm/pixel |
| Licença | AGPL-3.0 |

No Deepness, use **Detector type = `YOLO_Ultralytics`**, **Resolution = 3,5 cm/px**, Confidence 0,20 e IoU 0,30.

O modelo **Tree-Tops Detection** usado na comparação não é redistribuído aqui. Ele pode ser baixado no Model Zoo do Deepness.
