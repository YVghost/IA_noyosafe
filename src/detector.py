"""
NoyoSafe - Sistema de Detección Automática de Accidentes de Tránsito
Módulo principal: detección, seguimiento y alerta
Autores: Joseph Flores, Mateo Ortega
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import os
import sys

# Permite importar módulos hermanos cuando se ejecuta como script (python detector.py)
sys.path.insert(0, os.path.dirname(__file__))
from logger import AccidentLogger

# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────

MODEL_PATH           = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
COLLISION_IOU_THRESHOLD = 0.05   # subido de 0.05 → reduce falsos positivos
SPEED_CHANGE_THRESHOLD  = 5      # píxeles por frame
ALERT_DURATION_FRAMES   = 60

# Clases COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Colores BGR
COLOR_BOX       = (0, 255, 0)
COLOR_COLLISION = (0, 0, 255)
COLOR_ALERT_BG  = (0, 0, 200)
COLOR_ALERT_TX  = (255, 255, 255)
COLOR_TRACK     = (255, 165, 0)


# ─────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────

class NoyoSafeDetector:
    def __init__(self, model_path: str = MODEL_PATH, device: str | None = None):
        print("[NoyoSafe] Cargando modelo YOLOv8...")
        self.model = YOLO(model_path)

        try:
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = device or "cpu"
        print(f"[NoyoSafe] Modelo cargado | Dispositivo: {self.device.upper()}")

        # deque con maxlen evita pop(0) manual (O(1) vs O(n))
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.speed_history = defaultdict(lambda: deque(maxlen=10))

        self.alert_frames_remaining = 0
        self._in_collision = False       # para contar eventos únicos, no frames

        self.total_frames    = 0
        self.accident_events = 0         # eventos distintos (antes era "frames")
        self.detection_times: list = []

        self.logger = AccidentLogger()

    # ── Geometría ──────────────────────────────────────────────────────────

    def compute_iou(self, box1: list, box2: list) -> float:
        x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def compute_speed(self, track_id: int, cx: int, cy: int) -> float:
        history = self.track_history[track_id]
        if len(history) < 2:
            return 0.0
        prev_cx, prev_cy = history[-2]
        return float(np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2))

    # ── Detección de colisiones ────────────────────────────────────────────

    def detect_collision(self, vehicles: dict) -> tuple:
        """
        Devuelve (set de IDs en colisión, IoU máximo observado en el frame).
        Condición: IoU >= umbral AND cambio brusco de velocidad en al menos un vehículo.
        """
        colliding_ids = set()
        max_iou = 0.0

        ids   = list(vehicles.keys())
        boxes = [vehicles[i]["box"] for i in ids]

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                iou = self.compute_iou(boxes[i], boxes[j])
                if iou > max_iou:
                    max_iou = iou

                speed_change = False
                for tid in (ids[i], ids[j]):
                    sh = list(self.speed_history[tid])  # deque no soporta slicing
                    if len(sh) >= 3:
                        delta = abs(sh[-1] - np.mean(sh[:-1]))
                        if delta > SPEED_CHANGE_THRESHOLD:
                            speed_change = True
                            break

                if iou >= COLLISION_IOU_THRESHOLD and speed_change:
                    colliding_ids.add(ids[i])
                    colliding_ids.add(ids[j])

        return colliding_ids, max_iou

    # ── Visualización ──────────────────────────────────────────────────────

    def draw_alert(self, frame: np.ndarray, message: str = "!! ACCIDENTE DETECTADO") -> np.ndarray:
        _, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 70), COLOR_ALERT_BG, -1)
        cv2.putText(frame, message, (20, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, COLOR_ALERT_TX, 2, cv2.LINE_AA)
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w - 150, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ALERT_TX, 2, cv2.LINE_AA)
        return frame

    # ── Pipeline principal ─────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_path=None,
        show: bool = True,
        preprocess=None,
    ):
        """
        Procesa un video frame a frame.
        - video_path  : ruta al video de entrada
        - output_path : ruta de salida (None = no guardar)
        - show        : mostrar ventana en tiempo real
        - preprocess  : callable(frame) -> frame, para Experimento 4 (ver filters.py)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir el video: {video_path}")
            return

        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[NoyoSafe] Video: {width}x{height} | {fps} FPS | {total} frames")

        self.logger.set_video(video_path, fps)

        writer = None
        if output_path:
            out_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(out_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                # mp4v falla en algunos sistemas Windows; intentar con XVID + .avi
                alt_path = os.path.splitext(output_path)[0] + ".avi"
                fourcc   = cv2.VideoWriter_fourcc(*"XVID")
                writer   = cv2.VideoWriter(alt_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    print(f"[NoyoSafe] mp4v no disponible, guardando como AVI: {alt_path}")
                    output_path = alt_path
                else:
                    print(f"[WARN] No se pudo abrir el VideoWriter. El video NO se guardará.")
                    writer = None
            else:
                print(f"[NoyoSafe] Guardando video en: {output_path}")

        frame_idx          = 0
        self._in_collision = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx         += 1
            self.total_frames += 1
            t_start = time.time()

            if preprocess is not None:
                frame = preprocess(frame)

            # ── DETECCIÓN Y SEGUIMIENTO ────────────────────────
            results = self.model.track(
                frame,
                persist=True,
                classes=VEHICLE_CLASSES,
                conf=CONFIDENCE_THRESHOLD,
                tracker="bytetrack.yaml",
                verbose=False,
                device=self.device,
            )

            vehicles = {}

            if results[0].boxes is not None and results[0].boxes.id is not None:
                for box, track_id, _ in zip(
                    results[0].boxes.xyxy.cpu().numpy(),
                    results[0].boxes.id.cpu().numpy().astype(int),
                    results[0].boxes.conf.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    self.track_history[track_id].append((cx, cy))
                    speed = self.compute_speed(track_id, cx, cy)
                    self.speed_history[track_id].append(speed)

                    vehicles[track_id] = {"box": [x1, y1, x2, y2], "speed": speed}

            # ── DETECCIÓN DE COLISIONES ────────────────────────
            colliding_ids = set()
            max_iou = 0.0
            if len(vehicles) >= 2:
                colliding_ids, max_iou = self.detect_collision(vehicles)

            if colliding_ids:
                self.alert_frames_remaining = ALERT_DURATION_FRAMES
                if not self._in_collision:          # nuevo evento, no frame repetido
                    self.accident_events += 1
                    self.logger.log_event(frame_idx, colliding_ids, max_iou)
                self._in_collision = True
            else:
                self._in_collision = False

            # ── DIBUJAR SOBRE EL FRAME ─────────────────────────
            for track_id, data in vehicles.items():
                x1, y1, x2, y2 = data["box"]
                color = COLOR_COLLISION if track_id in colliding_ids else COLOR_BOX

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id} v:{data['speed']:.0f}px"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

                history = self.track_history[track_id]
                for k in range(1, len(history)):
                    cv2.line(frame, history[k - 1], history[k], COLOR_TRACK, 2)

            if self.alert_frames_remaining > 0:
                frame = self.draw_alert(frame)
                self.alert_frames_remaining -= 1

            t_end   = time.time()
            proc_ms = (t_end - t_start) * 1000
            self.detection_times.append(proc_ms)

            info = f"Frame {frame_idx}/{total} | {proc_ms:.1f}ms | Vehiculos: {len(vehicles)}"
            cv2.putText(frame, info, (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            if writer:
                writer.write(frame)

            if show:
                cv2.imshow("NoyoSafe", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[NoyoSafe] Detenido por el usuario.")
                    break

        # ── CIERRE ─────────────────────────────────────────────
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        if self.logger.events:
            self.logger.save()

        self.print_summary(frame_idx)

    def print_summary(self, total_frames_processed: int):
        avg_time = float(np.mean(self.detection_times)) if self.detection_times else 0.0
        avg_fps  = 1000.0 / avg_time if avg_time > 0 else 0.0

        print("\n" + "=" * 50)
        print("      RESUMEN NoyoSafe")
        print("=" * 50)
        print(f"  Frames procesados  : {total_frames_processed}")
        print(f"  Eventos detectados : {self.accident_events}")
        print(f"  Tiempo promedio    : {avg_time:.1f} ms/frame")
        print(f"  FPS promedio       : {avg_fps:.1f}")
        print(f"  Dispositivo        : {self.device.upper()}")
        print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python detector.py <ruta_video> [--filter <nombre>]")
        print("     filtros: night, rain_day, heavy_fog, low_light, rain, fog")
        print("Ej:  python detector.py ../data/videos/test.mp4")
        print("Ej:  python detector.py ../data/videos/test.mp4 --filter rain_day")
        sys.exit(1)

    video_input   = sys.argv[1]
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_output  = os.path.join(
        _project_root, "outputs", "detections", os.path.basename(video_input)
    )
    print(f"[NoyoSafe] Salida: {video_output}")

    preprocess_fn = None
    if "--filter" in sys.argv:
        idx = sys.argv.index("--filter")
        if idx + 1 < len(sys.argv):
            from filters import PRESETS
            filter_name   = sys.argv[idx + 1]
            preprocess_fn = PRESETS.get(filter_name)
            if preprocess_fn is None:
                print(f"[WARN] Filtro '{filter_name}' desconocido. Se ignora.")
            else:
                print(f"[NoyoSafe] Filtro activo: {filter_name}")

    detector = NoyoSafeDetector()
    detector.process_video(
        video_path=video_input,
        output_path=video_output,
        show=True,
        preprocess=preprocess_fn,
    )
