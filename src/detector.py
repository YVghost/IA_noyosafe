"""
NoyoSafe - Sistema de Detección Automática de Accidentes de Tránsito
Módulo principal: detección, seguimiento y alerta
Autores: Joseph Flores, Mateo Ortega
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import os

# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────

# Modelo YOLO a usar (se descarga automáticamente la primera vez)
MODEL_PATH = "yolov8n.pt"

# Umbral de confianza: YOLO solo reporta detecciones con confianza >= este valor
CONFIDENCE_THRESHOLD = 0.4

# IoU mínimo para considerar que dos vehículos colisionaron
# IoU = qué porcentaje del área de los dos rectángulos se superpone
COLLISION_IOU_THRESHOLD = 0.05

# Cambio mínimo de velocidad para considerar impacto brusco (píxeles por frame)
SPEED_CHANGE_THRESHOLD = 5

# Cuántos frames debe mantenerse la alerta visible después de detectar el accidente
ALERT_DURATION_FRAMES = 60

# Clases de vehículos según YOLO (COCO dataset)
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Colores (BGR para OpenCV)
COLOR_BOX       = (0, 255, 0)       # verde - bounding box normal
COLOR_COLLISION = (0, 0, 255)       # rojo  - vehículo en colisión
COLOR_ALERT_BG  = (0, 0, 200)       # rojo oscuro - fondo alerta
COLOR_ALERT_TX  = (255, 255, 255)   # blanco - texto alerta
COLOR_TRACK     = (255, 165, 0)     # naranja - línea de trayectoria


# ─────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────

class NoyoSafeDetector:
    def __init__(self, model_path=MODEL_PATH):
        print("[NoyoSafe] Cargando modelo YOLOv8...")
        self.model = YOLO(model_path)
        print("[NoyoSafe] Modelo cargado correctamente.")

        # Historial de posiciones por ID de vehículo
        # { track_id: [(cx, cy), (cx, cy), ...] }
        self.track_history = defaultdict(list)

        # Historial de velocidades por ID
        # { track_id: [speed, speed, ...] }
        self.speed_history = defaultdict(list)

        # Contador de frames en alerta activa
        self.alert_frames_remaining = 0

        # Estadísticas para el informe
        self.total_frames = 0
        self.accident_frames = 0
        self.detection_times = []   # tiempos de procesamiento por frame

    def compute_iou(self, box1, box2):
        """
        Calcula el IoU (Intersection over Union) entre dos bounding boxes.
        Cada box es [x1, y1, x2, y2].
        IoU = área de intersección / área de unión
        Resultado entre 0 (sin superposición) y 1 (idénticos)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def compute_speed(self, track_id, cx, cy):
        """
        Calcula la velocidad actual de un vehículo comparando su posición
        actual con la anterior. Devuelve la magnitud del vector de movimiento
        en píxeles por frame.
        """
        history = self.track_history[track_id]
        if len(history) < 2:
            return 0.0
        prev_cx, prev_cy = history[-2]
        speed = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
        return speed

    def detect_collision(self, vehicles):
        """
        Detecta colisiones entre vehículos.
        Condición de colisión: IoU >= umbral Y cambio brusco de velocidad.
        Devuelve lista de IDs de vehículos involucrados en colisión.
        """
        colliding_ids = set()

        ids   = list(vehicles.keys())
        boxes = [vehicles[i]["box"]   for i in ids]
        speeds= [vehicles[i]["speed"] for i in ids]

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                iou = self.compute_iou(boxes[i], boxes[j])

                # Detectar cambio brusco de velocidad en alguno de los dos
                speed_change = False
                sh_i = self.speed_history[ids[i]]
                sh_j = self.speed_history[ids[j]]

                if len(sh_i) >= 3:
                    delta = abs(sh_i[-1] - np.mean(sh_i[:-1]))
                    if delta > SPEED_CHANGE_THRESHOLD:
                        speed_change = True

                if len(sh_j) >= 3:
                    delta = abs(sh_j[-1] - np.mean(sh_j[:-1]))
                    if delta > SPEED_CHANGE_THRESHOLD:
                        speed_change = True

                if iou >= COLLISION_IOU_THRESHOLD and speed_change:
                    colliding_ids.add(ids[i])
                    colliding_ids.add(ids[j])

        return colliding_ids

    def draw_alert(self, frame, message="ACCIDENTE DETECTADO"):
        """
        Dibuja la alerta visual sobre el frame.
        Barra roja en la parte superior con texto de alerta.
        """
        h, w = frame.shape[:2]

        # Barra de alerta
        cv2.rectangle(frame, (0, 0), (w, 70), COLOR_ALERT_BG, -1)

        # Texto principal
        cv2.putText(frame, message, (20, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4,
                    COLOR_ALERT_TX, 2, cv2.LINE_AA)

        # Timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w - 150, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    COLOR_ALERT_TX, 2, cv2.LINE_AA)

        return frame

    def process_video(self, video_path, output_path=None, show=True):
        """
        Procesa un video completo frame por frame.
        - video_path: ruta al video de entrada
        - output_path: ruta donde guardar el video procesado (None = no guardar)
        - show: mostrar ventana en tiempo real
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir el video: {video_path}")
            return

        # Propiedades del video
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[NoyoSafe] Video: {width}x{height} | {fps} FPS | {total} frames")

        # Configurar escritor de video de salida
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            self.total_frames += 1
            t_start = time.time()

            # ── DETECCIÓN Y SEGUIMIENTO CON YOLO ──────────────
            # persist=True activa el tracker interno de Ultralytics (ByteTrack)
            results = self.model.track(
                frame,
                persist=True,
                classes=VEHICLE_CLASSES,
                conf=CONFIDENCE_THRESHOLD,
                tracker="bytetrack.yaml",
                verbose=False
            )

            vehicles = {}   # { track_id: {"box": [...], "speed": float} }

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Actualizar historial de posición
                    self.track_history[track_id].append((cx, cy))
                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id].pop(0)

                    # Calcular velocidad
                    speed = self.compute_speed(track_id, cx, cy)
                    self.speed_history[track_id].append(speed)
                    if len(self.speed_history[track_id]) > 10:
                        self.speed_history[track_id].pop(0)

                    vehicles[track_id] = {
                        "box": [x1, y1, x2, y2],
                        "speed": speed
                    }

            # ── DETECCIÓN DE COLISIONES ────────────────────────
            colliding_ids = set()
            if len(vehicles) >= 2:
                colliding_ids = self.detect_collision(vehicles)

            if colliding_ids:
                self.alert_frames_remaining = ALERT_DURATION_FRAMES
                self.accident_frames += 1

            # ── DIBUJAR SOBRE EL FRAME ─────────────────────────
            for track_id, data in vehicles.items():
                x1, y1, x2, y2 = data["box"]
                color = COLOR_COLLISION if track_id in colliding_ids else COLOR_BOX

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Etiqueta con ID y velocidad
                label = f"ID:{track_id} v:{data['speed']:.0f}px"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            color, 2, cv2.LINE_AA)

                # Línea de trayectoria
                history = self.track_history[track_id]
                for k in range(1, len(history)):
                    cv2.line(frame, history[k-1], history[k],
                             COLOR_TRACK, 2)

            # Alerta visual
            if self.alert_frames_remaining > 0:
                frame = self.draw_alert(frame)
                self.alert_frames_remaining -= 1

            # Info en esquina inferior
            t_end = time.time()
            proc_ms = (t_end - t_start) * 1000
            self.detection_times.append(proc_ms)

            info = f"Frame {frame_idx}/{total} | {proc_ms:.1f}ms | Vehiculos: {len(vehicles)}"
            cv2.putText(frame, info, (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 200), 1, cv2.LINE_AA)

            # Guardar frame
            if writer:
                writer.write(frame)

            # Mostrar en pantalla
            if show:
                cv2.imshow("NoyoSafe", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[NoyoSafe] Detenido por el usuario.")
                    break

        # ── RESUMEN FINAL ──────────────────────────────────────
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        self.print_summary(frame_idx)

    def print_summary(self, total_frames_processed):
        """Imprime resumen de métricas al terminar el video."""
        avg_time = np.mean(self.detection_times) if self.detection_times else 0
        avg_fps  = 1000 / avg_time if avg_time > 0 else 0

        print("\n" + "="*50)
        print("      RESUMEN NoyoSafe")
        print("="*50)
        print(f"  Frames procesados : {total_frames_processed}")
        print(f"  Frames con alerta : {self.accident_frames}")
        print(f"  Tiempo promedio   : {avg_time:.1f} ms/frame")
        print(f"  FPS promedio      : {avg_fps:.1f}")
        print("="*50 + "\n")


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Uso: python detector.py <ruta_video>
    # Ejemplo: python detector.py data/videos/accidente1.mp4
    if len(sys.argv) < 2:
        print("Uso: python detector.py <ruta_al_video>")
        print("Ejemplo: python detector.py data/videos/test.mp4")
        sys.exit(1)

    video_input  = sys.argv[1]
    video_output = os.path.join("outputs", "detections",
                                os.path.basename(video_input))

    detector = NoyoSafeDetector()
    detector.process_video(
        video_path=video_input,
        output_path=video_output,
        show=True
    )