"""
NoyoSafe – Módulo de benchmark de latencia (Experimento 3)

Mide el tiempo de procesamiento frame a frame en CPU y GPU (si disponible)
y calcula la latencia de detección desde que ocurre un accidente hasta
que se genera la alerta.

Uso:
    python benchmark.py <ruta_video> [modelo]

    Ejemplo:
        python benchmark.py data/videos/test.mp4 yolov8n.pt
"""

import cv2
import json
import os
import sys
import time

import numpy as np

VEHICLE_CLASSES     = [2, 3, 5, 7]
CONFIDENCE_THRESHOLD = 0.4
COLLISION_IOU       = 0.05
SPEED_THRESHOLD     = 5
ALERT_WINDOW        = 60      # frames que dura la alerta


# ─────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────

def _iou(b1: list, b2: list) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def _has_collision(vehicles: dict, speed_hist: dict) -> bool:
    ids   = list(vehicles.keys())
    boxes = [vehicles[i]["box"] for i in ids]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if _iou(boxes[i], boxes[j]) < COLLISION_IOU:
                continue
            for tid in (ids[i], ids[j]):
                sh = list(speed_hist.get(tid, []))
                if len(sh) >= 3 and abs(sh[-1] - np.mean(sh[:-1])) > SPEED_THRESHOLD:
                    return True
    return False


# ─────────────────────────────────────────────
# Medición en un dispositivo
# ─────────────────────────────────────────────

def _run_on_device(video_path: str, model_path: str, device: str) -> dict:
    """
    Procesa el video completo en el dispositivo indicado.
    Devuelve estadísticas de tiempo y latencias de detección.
    """
    from collections import defaultdict, deque
    from ultralytics import YOLO

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {video_path}")

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"[Benchmark] {device.upper()} | {total} frames | {fps_vid} FPS")

    frame_times: list[float] = []
    speed_hist  = defaultdict(lambda: deque(maxlen=10))
    track_hist  = defaultdict(lambda: deque(maxlen=30))

    in_collision  = False
    collision_start: int | None = None
    alert_latencies: list[float] = []   # segundos entre inicio colisión y alerta

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        t0 = time.perf_counter()
        results = model.track(
            frame,
            persist=True,
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            tracker="bytetrack.yaml",
            verbose=False,
            device=device,
        )
        t1 = time.perf_counter()
        frame_times.append((t1 - t0) * 1000)

        vehicles: dict = {}
        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box, tid in zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.id.cpu().numpy().astype(int),
            ):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                prev = track_hist[tid][-1] if len(track_hist[tid]) >= 1 else (cx, cy)
                speed = float(np.sqrt((cx - prev[0])**2 + (cy - prev[1])**2))
                track_hist[tid].append((cx, cy))
                speed_hist[tid].append(speed)
                vehicles[tid] = {"box": [x1, y1, x2, y2], "speed": speed}

        collision_now = len(vehicles) >= 2 and _has_collision(vehicles, speed_hist)

        if collision_now and not in_collision:
            collision_start = frame_idx
        if collision_now:
            in_collision = True
        else:
            if in_collision and collision_start is not None:
                latency_frames = frame_idx - collision_start
                alert_latencies.append(latency_frames / fps_vid)
            in_collision = False
            collision_start = None

    cap.release()

    arr = np.array(frame_times)
    result = {
        "device":           device,
        "frames_processed": frame_idx,
        "avg_ms":           round(float(np.mean(arr)),             2),
        "median_ms":        round(float(np.median(arr)),           2),
        "min_ms":           round(float(np.min(arr)),              2),
        "max_ms":           round(float(np.max(arr)),              2),
        "p95_ms":           round(float(np.percentile(arr, 95)),   2),
        "avg_fps":          round(1000.0 / float(np.mean(arr)),    1),
    }
    if alert_latencies:
        result["alert_latency_avg_s"] = round(float(np.mean(alert_latencies)),   3)
        result["alert_latency_min_s"] = round(float(np.min(alert_latencies)),    3)
        result["alert_latency_max_s"] = round(float(np.max(alert_latencies)),    3)

    return result


# ─────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────

def run_benchmark(
    video_path: str,
    model_path: str = "yolov8n.pt",
    output_dir: str = "outputs/benchmark",
) -> dict:
    """
    Corre el benchmark en todos los dispositivos disponibles y guarda el reporte JSON.
    Devuelve el dict del reporte.
    """
    devices = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            devices.append("cuda")
            print(f"[Benchmark] GPU detectada: {torch.cuda.get_device_name(0)}")
        else:
            print("[Benchmark] GPU no disponible — solo CPU.")
    except ImportError:
        pass

    results: dict = {}
    for dev in devices:
        print(f"\n[Benchmark] === Corriendo en {dev.upper()} ===")
        results[dev] = _run_on_device(video_path, model_path, dev)

    report: dict = {
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video":      os.path.basename(video_path),
        "model":      model_path,
        "results":    results,
    }

    if "cpu" in results and "cuda" in results:
        speedup = results["cpu"]["avg_ms"] / results["cuda"]["avg_ms"]
        report["gpu_speedup_x"] = round(speedup, 2)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(
        output_dir, f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _print_report(report)
    print(f"[Benchmark] Reporte guardado: {report_path}")
    return report


def _print_report(report: dict):
    print("\n" + "=" * 58)
    print("  BENCHMARK NoyoSafe — Experimento 3 (Latencia)")
    print("=" * 58)
    for dev, res in report["results"].items():
        print(f"\n  Dispositivo      : {dev.upper()}")
        print(f"  Frames           : {res['frames_processed']}")
        print(f"  Prom ms/frame    : {res['avg_ms']} ms")
        print(f"  Mediana ms/frame : {res['median_ms']} ms")
        print(f"  P95 ms/frame     : {res['p95_ms']} ms")
        print(f"  FPS promedio     : {res['avg_fps']}")
        if "alert_latency_avg_s" in res:
            print(f"  Latencia alerta  : {res['alert_latency_avg_s']} s (prom) "
                  f"| {res['alert_latency_min_s']} s (min) "
                  f"| {res['alert_latency_max_s']} s (max)")
    if "gpu_speedup_x" in report:
        print(f"\n  Aceleracion GPU  : {report['gpu_speedup_x']}x")
    print("=" * 58 + "\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:  python benchmark.py <ruta_video> [modelo]")
        print("Ej:   python benchmark.py ../data/videos/test.mp4 yolov8n.pt")
        sys.exit(1)

    vpath = sys.argv[1]
    mpath = sys.argv[2] if len(sys.argv) > 2 else "yolov8n.pt"
    run_benchmark(vpath, mpath)
