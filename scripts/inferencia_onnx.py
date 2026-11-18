"""Teste rápido do modelo ONNX num recorte (PNG/JPG), sem QGIS.

Uso:
    python scripts/inferencia_onnx.py --modelo best_v3_merge_V2.onnx \
        --imagem tile.png --saida resultado.png --conf 0.25 --iou 0.45

A imagem é redimensionada para 640x640 com bordas (letterbox), como no treino.
Para ortofotos inteiras, use o plugin Deepness no QGIS.
"""
import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

TAMANHO = 640


def letterbox(img: Image.Image, tamanho: int = TAMANHO):
    w, h = img.size
    escala = min(tamanho / w, tamanho / h)
    nw, nh = round(w * escala), round(h * escala)
    fundo = Image.new("RGB", (tamanho, tamanho), (114, 114, 114))
    dx, dy = (tamanho - nw) // 2, (tamanho - nh) // 2
    fundo.paste(img.resize((nw, nh), Image.BILINEAR), (dx, dy))
    return fundo, escala, dx, dy


def nms(caixas: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    x1, y1, x2, y2 = caixas.T
    areas = (x2 - x1) * (y2 - y1)
    ordem = scores.argsort()[::-1]
    manter = []
    while ordem.size:
        i = ordem[0]
        manter.append(int(i))
        xx1 = np.maximum(x1[i], x1[ordem[1:]])
        yy1 = np.maximum(y1[i], y1[ordem[1:]])
        xx2 = np.minimum(x2[i], x2[ordem[1:]])
        yy2 = np.minimum(y2[i], y2[ordem[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        sobre = inter / (areas[i] + areas[ordem[1:]] - inter + 1e-9)
        ordem = ordem[1:][sobre < iou]
    return manter


def detectar(modelo: str, imagem: str, conf: float, iou: float):
    original = Image.open(imagem).convert("RGB")
    entrada, escala, dx, dy = letterbox(original)
    x = np.asarray(entrada, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

    sessao = ort.InferenceSession(modelo, providers=["CPUExecutionProvider"])
    saida = sessao.run(None, {sessao.get_inputs()[0].name: x})[0][0]  # (5, 8400)
    cx, cy, w, h, score = saida

    filtro = score >= conf
    cx, cy, w, h, score = cx[filtro], cy[filtro], w[filtro], h[filtro], score[filtro]
    caixas = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    if len(caixas):
        idx = nms(caixas, score, iou)
        caixas, score = caixas[idx], score[idx]

    # volta para as coordenadas da imagem original
    caixas[:, [0, 2]] = (caixas[:, [0, 2]] - dx) / escala
    caixas[:, [1, 3]] = (caixas[:, [1, 3]] - dy) / escala
    caixas = caixas.clip(0, [original.width, original.height] * 2)
    return original, caixas, score


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--modelo", required=True)
    p.add_argument("--imagem", required=True)
    p.add_argument("--saida", default="resultado.png")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    a = p.parse_args()

    img, caixas, scores = detectar(a.modelo, a.imagem, a.conf, a.iou)
    desenho = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), s in zip(caixas, scores):
        desenho.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        desenho.text((x1 + 2, y1 + 2), f"{s:.2f}", fill=(0, 255, 0))
    img.save(a.saida)
    print(f"{len(caixas)} árvores detectadas -> {a.saida}")


if __name__ == "__main__":
    main()
