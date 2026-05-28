"""
NoyoSafe – Módulo de filtros de condiciones adversas (Experimento 4)
Simula baja iluminación, lluvia y niebla sobre frames individuales.

Uso:
    from filters import apply_low_light, apply_rain, apply_fog
    frame_degradado = apply_rain(frame, intensity=0.6)

También se puede componer:
    frame_degradado = apply_fog(apply_low_light(frame, factor=0.4), intensity=0.3)
"""

import cv2
import numpy as np


def apply_low_light(frame: np.ndarray, factor: float = 0.3) -> np.ndarray:
    """
    Simula baja iluminación multiplicando el brillo por 'factor'.
    factor=1.0 → sin cambio | factor=0.1 → muy oscuro
    """
    factor = float(np.clip(factor, 0.05, 1.0))
    return (frame.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def apply_rain(frame: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Simula lluvia dibujando trazos diagonales semitransparentes.
    intensity: 0.0 (sin lluvia) → 1.0 (lluvia intensa)
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = frame.shape[:2]
    rain_layer = np.zeros_like(frame, dtype=np.uint8)

    n_drops = int(intensity * 700)
    rng = np.random.default_rng(seed=42)  # seed fijo para reproducibilidad

    starts_x = rng.integers(0, w, n_drops)
    starts_y = rng.integers(0, h, n_drops)
    lengths  = rng.integers(8, 20, n_drops)
    offsets  = rng.integers(-3, 4, n_drops)

    for i in range(n_drops):
        x1, y1 = int(starts_x[i]), int(starts_y[i])
        x2 = int(np.clip(x1 + offsets[i], 0, w - 1))
        y2 = int(np.clip(y1 + lengths[i], 0, h - 1))
        cv2.line(rain_layer, (x1, y1), (x2, y2), (180, 185, 200), 1)

    return cv2.addWeighted(frame, 0.85, rain_layer, 0.70, 0)


def apply_fog(frame: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Simula niebla/neblina fusionando el frame con una capa gris clara.
    intensity: 0.0 (sin niebla) → 1.0 (niebla densa)
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    fog_layer = np.full_like(frame, 210, dtype=np.uint8)
    alpha = intensity * 0.80
    return cv2.addWeighted(frame, 1.0 - alpha, fog_layer, alpha, 0)


# ─────────────────────────────────────────────
# Presets listos para usar en process_video(preprocess=...)
# ─────────────────────────────────────────────

def preset_night(frame: np.ndarray) -> np.ndarray:
    """Noche: baja luz + niebla leve."""
    return apply_fog(apply_low_light(frame, factor=0.25), intensity=0.15)


def preset_rain_day(frame: np.ndarray) -> np.ndarray:
    """Lluvia diurna: lluvia intensa + brillo ligeramente reducido."""
    return apply_rain(apply_low_light(frame, factor=0.75), intensity=0.70)


def preset_heavy_fog(frame: np.ndarray) -> np.ndarray:
    """Niebla densa."""
    return apply_fog(frame, intensity=0.65)


PRESETS = {
    "night":      preset_night,
    "rain_day":   preset_rain_day,
    "heavy_fog":  preset_heavy_fog,
    "low_light":  lambda f: apply_low_light(f, factor=0.3),
    "rain":       lambda f: apply_rain(f, intensity=0.5),
    "fog":        lambda f: apply_fog(f, intensity=0.5),
}
