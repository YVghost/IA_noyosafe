## Conclusión (generada a partir de las métricas calculadas arriba)

| Métrica    | Modelo normal | Modelo entrenado | Diferencia |
|---|---|---|---|
| Precisión  | 0.485 | 0.670 | +0.185 |
| TPR (Recall) | 0.242 | 0.985 | +0.742 |
| FPR        | 0.187 | 0.352 | +0.165 |
| F1-score   | 0.323 | 0.798 | +0.474, +147% |
| mAP50      | N/A (no aplica a una regla heurística) | 0.886 | — |
| Latencia   | 10.6 ms/img | 10.1 ms/img | — |
| Cumple criterio de éxito del informe (TPR>70%, FPR<20%) | No | No | — |

La heurística de IoU + velocidad de `src/detector.py` está diseñada para video con seguimiento temporal
(ByteTrack); sobre imágenes sueltas pierde la señal de cambio de velocidad y solo puede usar el
solapamiento geométrico entre vehículos, lo que limita su TPR en este set de test. El modelo entrenado
reconoce el patrón visual de un accidente directamente en una sola imagen, sin depender de tracking
previo, y fue además expuesto durante el entrenamiento a condiciones adversas simuladas (Sección 6,
mejora §5.2 del informe) — el Experimento 4 repetido en la Sección 13 muestra cuánto ayuda eso frente
a lluvia, niebla y baja luz. Las curvas PR/ROC de la Sección 11 (mejora §5.4 del informe) muestran el
trade-off completo, no solo el punto de operación por defecto.
