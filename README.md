# Medical Image Classification — CNN Architectures

Trabajo Practico Semana 6 — Arquitecturas CNN: Comparacion, Batch Normalization y Transfer Learning

## Descripcion

Implementacion, entrenamiento y analisis de arquitecturas clasicas de redes neuronales convolucionales para clasificacion de imagenes microscopicas de celulas sanguineas. Se implementaron LeNet-5 y VGG-11 desde cero, y se aplico Transfer Learning con VGG-16 preentrenada en ImageNet.

**Dataset:** Blood Cell Image Dataset — 4 clases (EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL), ~12,500 imagenes RGB de 320x240px.

## Requisitos

- Python 3.11
- PyTorch 2.6.0 + CUDA 12.4
- Ver `requirements.txt` para dependencias completas

## Instalacion

```bash
conda create -n ml-upc-dl python=3.11 -y
conda activate ml-upc-dl
pip install -r requirements.txt
```

## Dataset

```bash
bash data/download_data.sh
```

Requiere cuenta en Kaggle y API key configurada en `~/.kaggle/kaggle.json`.

## Ejecucion

Correr los notebooks en orden:

1. `notebooks/01_eda.ipynb` — Analisis exploratorio del dataset
2. `notebooks/02_lenet_vgg.ipynb` — Implementacion y entrenamiento de LeNet5 y VGG11
3. `notebooks/03_transfer.ipynb` — Transfer Learning con VGG16

## Resultados

| Modelo | Parametros | Tiempo/epoca | Mejor Accuracy |
|--------|-----------|--------------|---------------|
| LeNet5 | 337,976 | ~12s | 64.6% |
| LeNet5 BN | 338,020 | ~12s | 58.9% |
| VGG11 | 8,609,988 | ~13s | 85.1% |
| VGG11 BN | 8,612,740 | ~13s | 81.7% |
| VGG16 Feature Extraction | 134,276,932 | ~27s | 40.5% |
| VGG16 Fine-tuning parcial | 134,276,932 | ~32s | 85.0% |
| VGG16 Fine-tuning total | 134,276,932 | ~35s | 80.7% |

## Hallazgos principales

- VGG11 supero a LeNet5 significativamente con su 85.1% vs el 64.6% de este ultimo. Asi se confirma que arquitecturas mas profundas capturan mejor las caracteristicas de las celulas.
- Batch Normalization no mejoro el rendimiento en este experimento — ambas variantes con BN obtuvieron menor accuracy que los otros modelos sin BN. Puede deberse a la sensibilidad al learning rate.
- Transfer Learning con Fine-tuning parcial igualo al mejor modelo desde cero con un 85% en menos epocas de entrenamiento. Asi se demostro la utilidad de los pesos preentrenados en ImageNet.
- Feature Extraction resulto insuficiente con su 40.5%, confirmando que las features de ImageNet no son directamente aplicables a imagenes microscopicas sin ajuste.