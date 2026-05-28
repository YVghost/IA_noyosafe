"""
NoyoSafe – Módulo de evaluación de métricas

Experimento 1 – Detección de vehículos (Precision / Recall / F1)
    Compara detecciones de YOLO contra bounding boxes de ground truth.
    Formato GT: {"frames": [{"frame_id": 1, "boxes": [[x1,y1,x2,y2], ...]}, ...]}

Experimento 2 – Detección de accidentes (TPR / FPR / Precision / Recall / F1)
    Compara eventos del AccidentLogger contra segmentos etiquetados.
    Formato GT: {
        "total_frames": 1200,
        "accident_segments": [{"start": 100, "end": 200, "description": "..."}, ...]
    }

Uso rápido:
    evaluator = Evaluator()
    results = evaluator.evaluate_accidents("gt.json", "log.json")
    evaluator.print_report(results)
    evaluator.save_report(results)
"""

import json
import os
import time


def _iou(box1: list, box2: list) -> float:
    """IoU entre dos boxes [x1, y1, x2, y2]."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class Evaluator:

    # ── Experimento 2: detección de accidentes ──────────────────────────────

    def evaluate_accidents(
        self,
        gt_file: str,
        log_file: str,
        alert_window: int = 60,
    ) -> dict:
        """
        Evalúa la detección de accidentes a nivel de frame.

        gt_file      : JSON con 'total_frames' y 'accident_segments'.
        log_file     : JSON del AccidentLogger con 'events'.
        alert_window : cuántos frames dura cada alerta (debe coincidir
                       con ALERT_DURATION_FRAMES del detector).

        Devuelve dict con TP, FP, FN, TN, Precision, Recall, F1, TPR, FPR.
        """
        with open(gt_file, encoding="utf-8") as f:
            gt = json.load(f)
        with open(log_file, encoding="utf-8") as f:
            log = json.load(f)

        total_frames: int = gt["total_frames"]

        # Frames con accidente real
        gt_frames: set[int] = set()
        for seg in gt.get("accident_segments", []):
            gt_frames.update(range(int(seg["start"]), int(seg["end"]) + 1))

        # Frames cubiertos por las alertas del detector
        pred_frames: set[int] = set()
        for event in log.get("events", []):
            start = int(event["frame"])
            pred_frames.update(range(start, min(start + alert_window, total_frames + 1)))

        all_frames = set(range(1, total_frames + 1))

        tp = len(gt_frames & pred_frames)
        fp = len(pred_frames - gt_frames)
        fn = len(gt_frames - pred_frames)
        tn = len(all_frames - gt_frames - pred_frames)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "experiment": "2_accident_detection",
            "gt_file":    os.path.basename(gt_file),
            "log_file":   os.path.basename(log_file),
            "total_frames":     total_frames,
            "gt_accident_frames":   len(gt_frames),
            "pred_alert_frames":    len(pred_frames),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "tpr":       round(tpr,       4),
            "fpr":       round(fpr,       4),
        }

    # ── Experimento 1: detección de vehículos ───────────────────────────────

    def evaluate_vehicles(
        self,
        gt_file: str,
        predictions: dict,
        iou_threshold: float = 0.5,
    ) -> dict:
        """
        Evalúa la detección de vehículos frame a frame usando IoU matching.

        gt_file     : JSON con {"frames": [{"frame_id": int,
                                            "boxes": [[x1,y1,x2,y2], ...]}, ...]}
        predictions : {frame_id: [[x1,y1,x2,y2], ...]}
                      (obtenido corriendo el módulo detector en modo evaluación)
        iou_threshold: IoU mínimo para contar como TP (por defecto PASCAL VOC: 0.5)
        """
        with open(gt_file, encoding="utf-8") as f:
            gt_data = json.load(f)

        tp = fp = fn = 0

        for frame_info in gt_data.get("frames", []):
            fid       = int(frame_info["frame_id"])
            gt_boxes  = frame_info.get("boxes", [])
            pred_boxes = predictions.get(fid, [])

            matched_gt: set[int] = set()
            for pb in pred_boxes:
                best_iou = 0.0
                best_idx = -1
                for idx, gb in enumerate(gt_boxes):
                    if idx in matched_gt:
                        continue
                    score = _iou(pb, gb)
                    if score > best_iou:
                        best_iou = score
                        best_idx = idx
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_idx)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        return {
            "experiment":     "1_vehicle_detection",
            "gt_file":        os.path.basename(gt_file),
            "iou_threshold":  iou_threshold,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        }

    # ── Utilidades ──────────────────────────────────────────────────────────

    def print_report(self, results: dict):
        exp = results.get("experiment", "?")
        print("\n" + "=" * 52)
        print(f"  EVALUACION — {exp}")
        print("=" * 52)
        skip = {"experiment", "gt_file", "log_file"}
        for k, v in results.items():
            if k not in skip:
                print(f"  {k:<26}: {v}")
        print("=" * 52 + "\n")

    def save_report(self, results: dict, output_dir: str = "outputs/reports") -> str:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"eval_{results.get('experiment', 'report')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Evaluator] Reporte guardado: {filepath}")
        return filepath


# ─────────────────────────────────────────────
# Función de ayuda: extrae predicciones de vehículos de un video
# usando el modelo YOLO para alimentar evaluate_vehicles()
# ─────────────────────────────────────────────

def extract_vehicle_predictions(
    video_path: str,
    model_path: str = "yolov8n.pt",
    confidence: float = 0.4,
    vehicle_classes: list | None = None,
    device: str = "cpu",
) -> dict:
    """
    Corre YOLOv8 sobre cada frame de un video y devuelve las detecciones
    en el formato que espera Evaluator.evaluate_vehicles().

    Retorna: {frame_id: [[x1, y1, x2, y2], ...], ...}
    """
    import cv2
    from ultralytics import YOLO

    if vehicle_classes is None:
        vehicle_classes = [2, 3, 5, 7]

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    predictions: dict = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.predict(
            frame,
            classes=vehicle_classes,
            conf=confidence,
            verbose=False,
            device=device,
        )

        boxes = []
        if results[0].boxes is not None:
            for b in results[0].boxes.xyxy.cpu().numpy():
                boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
        predictions[frame_idx] = boxes

    cap.release()
    return predictions
