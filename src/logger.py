"""
NoyoSafe – Módulo de logging de eventos
Registra cada accidente detectado en un archivo JSON con marca de tiempo.
"""

import json
import os
import time


class AccidentLogger:
    def __init__(self, output_dir="outputs/logs"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.events: list[dict] = []
        self._video_source = ""
        self._fps = 1

    def set_video(self, video_path: str, fps: int):
        self._video_source = os.path.basename(video_path)
        self._fps = fps if fps > 0 else 1

    def log_event(self, frame_idx: int, vehicle_ids: set, max_iou: float) -> dict:
        """Registra un nuevo evento de colisión."""
        event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame": frame_idx,
            "time_in_video_s": round(frame_idx / self._fps, 2),
            "vehicle_ids": sorted(int(i) for i in vehicle_ids),
            "max_iou": round(float(max_iou), 4),
            "video_source": self._video_source,
        }
        self.events.append(event)
        print(f"[Logger] Evento #{len(self.events)} registrado — frame {frame_idx} "
              f"| IDs: {event['vehicle_ids']} | IoU: {event['max_iou']}")
        return event

    def save(self, filename: str | None = None) -> str:
        """Guarda todos los eventos en un archivo JSON y devuelve la ruta."""
        if filename is None:
            filename = f"accidents_{time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        payload = {
            "total_events": len(self.events),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "video_source": self._video_source,
            "events": self.events,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[Logger] Log guardado: {filepath}")
        return filepath

    def load(self, filepath: str):
        """Carga un log JSON existente (útil para el evaluador)."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        self.events = data.get("events", [])
        self._video_source = data.get("video_source", "")
        return data
