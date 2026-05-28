# NoyoSafe – Sistema de Detección Automática de Accidentes de Tránsito

**Autores:** Joseph Flores, Mateo Ortega  
**Asignatura:** Inteligencia Artificial 1 – Ingeniería de Software, UDLA  
**Fecha:** Abril 2026

---

## Descripción

NoyoSafe detecta accidentes de tránsito en tiempo real procesando video de cámaras de vigilancia. Combina **YOLOv8** para detección de vehículos, **ByteTrack** para seguimiento de trayectorias, y lógica geométrica/vectorial para identificar colisiones por superposición de bounding boxes y cambios bruscos de velocidad.

---

## Estructura del proyecto

```
IA_noyosafe/
├── src/
│   ├── detector.py      # Pipeline principal: detección, seguimiento y alerta
│   ├── logger.py        # Registro de eventos de accidente en JSON
│   ├── evaluator.py     # Métricas: Precision, Recall, F1, TPR, FPR
│   ├── benchmark.py     # Latencia CPU vs GPU (Experimento 3)
│   └── filters.py       # Filtros de condiciones adversas (Experimento 4)
├── data/
│   └── videos/          # Videos de prueba (.mp4)
├── outputs/
│   ├── detections/      # Videos procesados con anotaciones
│   ├── logs/            # JSON de eventos detectados
│   ├── benchmark/       # Reportes de benchmark
│   └── reports/         # Reportes de evaluación
└── requirements.txt
```

---

## Instalación

### 1. Crear entorno virtual

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 2. Instalar dependencias

**Solo CPU:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Con GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Comandos de ejecución

> Todos los comandos se ejecutan desde la carpeta `src/`.
> ```bash
> cd src
> ```

---

### Detección básica (detector.py)

**Procesar un video y guardar resultado:**
```bash
python detector.py ../data/videos/test.mp4
```

**Solo ver en pantalla, sin guardar:**
```bash
python detector.py ../data/videos/test.mp4
# presionar Q para detener
```

**Con filtro de condición adversa (Experimento 4):**
```bash
python detector.py ../data/videos/test.mp4 --filter rain_day
python detector.py ../data/videos/test.mp4 --filter night
python detector.py ../data/videos/test.mp4 --filter heavy_fog
python detector.py ../data/videos/test.mp4 --filter low_light
python detector.py ../data/videos/test.mp4 --filter rain
python detector.py ../data/videos/test.mp4 --filter fog
```

El video procesado se guarda en:
```
outputs/detections/<nombre_del_video>.mp4
```
El log de eventos se guarda en:
```
outputs/logs/accidents_YYYYMMDD_HHMMSS.json
```

---

### Desde Python

**Detección simple:**
```python
from detector import NoyoSafeDetector

detector = NoyoSafeDetector()
detector.process_video(
    video_path="data/videos/test.mp4",
    output_path="outputs/detections/test.mp4",
    show=True,
)
```

**Forzar CPU o GPU:**
```python
detector = NoyoSafeDetector(device="cpu")
detector = NoyoSafeDetector(device="cuda")
```

**Con filtro de condiciones adversas:**
```python
from detector import NoyoSafeDetector
from filters import PRESETS, apply_rain, apply_low_light

detector = NoyoSafeDetector()

# Usando preset
detector.process_video("data/videos/test.mp4", preprocess=PRESETS["night"])

# Filtro manual
detector.process_video("data/videos/test.mp4", preprocess=apply_rain)

# Composición: lluvia + baja luz
detector.process_video(
    "data/videos/test.mp4",
    preprocess=lambda f: apply_rain(apply_low_light(f, factor=0.4), intensity=0.6),
)
```

**Iterar todos los filtros (Experimento 4 completo):**
```python
from detector import NoyoSafeDetector
from filters import PRESETS

for nombre, filtro in PRESETS.items():
    det = NoyoSafeDetector()
    det.process_video(
        video_path="data/videos/test.mp4",
        output_path=f"outputs/detections/test_{nombre}.mp4",
        show=False,
        preprocess=filtro,
    )
    print(f"Filtro: {nombre} | Eventos: {det.accident_events}")
```

---

### Benchmark CPU vs GPU (Experimento 3)

**Línea de comandos:**
```bash
python benchmark.py ../data/videos/test.mp4
python benchmark.py ../data/videos/test.mp4 yolov8s.pt   # modelo más pesado
```

**Desde Python:**
```python
from benchmark import run_benchmark

report = run_benchmark(
    video_path="data/videos/test.mp4",
    model_path="yolov8n.pt",
    output_dir="outputs/benchmark",
)
print(f"Speedup GPU: {report.get('gpu_speedup_x', 'N/A')}x")
```

El reporte se guarda en:
```
outputs/benchmark/benchmark_YYYYMMDD_HHMMSS.json
```

---

### Evaluación de métricas (Experimentos 1 y 2)

**Formato del archivo de ground truth para accidentes (`gt_accidentes.json`):**
```json
{
  "total_frames": 1200,
  "accident_segments": [
    { "start": 340, "end": 410, "description": "colision trasera" },
    { "start": 890, "end": 930, "description": "colision lateral" }
  ]
}
```

**Formato del archivo de ground truth para vehículos (`gt_vehiculos.json`):**
```json
{
  "frames": [
    { "frame_id": 1, "boxes": [[120, 80, 300, 210], [400, 150, 600, 320]] },
    { "frame_id": 2, "boxes": [[125, 82, 305, 215]] }
  ]
}
```

**Experimento 2 – Detección de accidentes (TPR / FPR):**
```python
from evaluator import Evaluator

ev = Evaluator()
results = ev.evaluate_accidents(
    gt_file="data/gt_accidentes.json",
    log_file="outputs/logs/accidents_20260415_120000.json",
)
ev.print_report(results)
ev.save_report(results)
```

**Experimento 1 – Detección de vehículos (Precision / Recall / F1):**
```python
from evaluator import Evaluator, extract_vehicle_predictions

# Paso 1: extraer predicciones del video
preds = extract_vehicle_predictions("data/videos/test.mp4", device="cpu")

# Paso 2: comparar contra ground truth
ev = Evaluator()
results = ev.evaluate_vehicles(
    gt_file="data/gt_vehiculos.json",
    predictions=preds,
    iou_threshold=0.5,
)
ev.print_report(results)
ev.save_report(results)
```

Los reportes se guardan en:
```
outputs/reports/eval_<experimento>_YYYYMMDD_HHMMSS.json
```

---

## Parámetros configurables

Editar al inicio de `src/detector.py`:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `MODEL_PATH` | `yolov8n.pt` | Modelo YOLO (n/s/m/l/x) |
| `CONFIDENCE_THRESHOLD` | `0.4` | Confianza mínima de detección |
| `COLLISION_IOU_THRESHOLD` | `0.05` | IoU mínimo para colisión |
| `SPEED_CHANGE_THRESHOLD` | `5` | Cambio de velocidad mínimo (px/frame) |
| `ALERT_DURATION_FRAMES` | `60` | Duración visual de la alerta (frames) |

---

## Filtros disponibles

| Nombre | Efecto |
|---|---|
| `low_light` | Brilla × 0.3 |
| `rain` | Lluvia media |
| `fog` | Niebla media |
| `night` | Baja luz + niebla leve |
| `rain_day` | Lluvia intensa + brillo 75% |
| `heavy_fog` | Niebla densa |

---

## Modelos YOLO disponibles

| Modelo | Velocidad | Precisión |
|---|---|---|
| `yolov8n.pt` | Más rápido | Menor |
| `yolov8s.pt` | Rápido | Media |
| `yolov8m.pt` | Medio | Alta |
| `yolov8l.pt` | Lento | Muy alta |
| `yolov8x.pt` | Más lento | Máxima |

Se descargan automáticamente de Ultralytics la primera vez que se usan.
