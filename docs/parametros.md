# Parâmetros testados (explicação simples)

## Reamostragem e formato do raster

| Etapa | Ferramenta no QGIS | Por quê |
|---|---|---|
| 3,5 cm → 10 cm/pixel | `Raster → Projeções → Deformar (reprojetar)` | Colocar a imagem na escala em que DeepForest e Tree-Tops foram treinados |
| `TILED=YES` | `Raster → Conversão → Converter (formato)` → Opções de criação adicionais | Permite ler a imagem em blocos, sem carregar tudo na memória |

Equivalente em linha de comando:

```bash
gdalwarp -tr 0.1 0.1 -r average -co TILED=YES entrada.tif saida_10cm.tif
```

## DeepForest (GeoAI)

| Parâmetro | O que faz | Observação |
|---|---|---|
| **Mode** | *Single Image* processa tudo de uma vez; *Large Tile* divide em pedaços | Imagens grandes exigem *Large Tile* |
| **Patch Size** | Tamanho de cada pedaço analisado | Patch × resolução deve dar cerca de 40 m (400 px a 10 cm) |
| **Patch Overlap** | Quanto um pedaço se sobrepõe ao vizinho | Evita cortar árvores na borda, mas aumenta duplicatas |
| **IoU Threshold** | Quando duas caixas viram uma | Alto demais funde árvores vizinhas: 0,25 derrubou de 71 para 36 detecções |
| **Batch Size** | Quantos pedaços por vez | 1 em CPU |
| **Score Threshold** | Confiança mínima para aceitar uma detecção | Baixo detecta mais e erra mais; alto detecta menos e erra menos |

## Deepness

| Parâmetro | Tree-Tops (pré-treinado) | Modelo próprio |
|---|---|---|
| Detector type | `YOLO_v9` | `YOLO_Ultralytics` |
| Resolution [cm/px] | 10 | 3,5 |
| Tile size | 640 | 640 |
| Tiles overlap | 15% | 15% |
| Input layer | recorte reamostrado em 10 cm | ortofoto original (3,5 cm) |
| Confidence | 0,20 | 0,20 |
| IoU threshold | 0,30 | 0,30 |

### Tipos de detector no Deepness

- `YOLO_v5_or_v7_default`: exportações dos repositórios clássicos YOLOv5/v7
- `YOLO_v6` / `YOLO_v9`: formatos dessas arquiteturas (o Tree-Tops usa `YOLO_v9`)
- `YOLO_Ultralytics`: modelos treinados e exportados pelo pacote `ultralytics` (caixas)
- `YOLO_Ultralytics_segmentation`: mesma biblioteca, com máscaras
- `YOLO_Ultralytics_obb`: caixas rotacionadas
