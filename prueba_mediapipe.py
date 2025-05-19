import cv2
import mediapipe as mp
import os

# === Parámetros ===
INPUT_IMAGE = "feret_png/00002/00002_931230_fb.png"           # Ruta a tu imagen original
OUTPUT_IMAGE = "rostro_recortado.png" # Ruta de salida
IMAGE_SIZE = (160, 160)               # Tamaño del recorte
MIN_CONFIDENCE = 0.9                  # Umbral de confianza
MARGEN = 0                          # Margen alrededor del rostro (20%)

# Inicializar MediaPipe
mp_face_detection = mp.solutions.face_detection

# Cargar imagen
imagen = cv2.imread(INPUT_IMAGE)
if imagen is None:
    print(f"[✗] No se pudo abrir la imagen: {INPUT_IMAGE}")
    exit()

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=MIN_CONFIDENCE) as face_detection:
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = face_detection.process(imagen_rgb)

    if resultados.detections:
        # Tomar la detección con mayor score
        deteccion = max(resultados.detections, key=lambda d: d.score[0])
        confianza = deteccion.score[0]

        if confianza < MIN_CONFIDENCE:
            print(f"[!] Rostro detectado pero con confianza baja ({confianza:.2f}). Imagen descartada.")
        else:
            bbox = deteccion.location_data.relative_bounding_box
            h, w, _ = imagen.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)

            dx = int(w_box * MARGEN)
            dy = int(h_box * MARGEN)
            x1 = max(x - dx, 0)
            y1 = max(y - dy, 0)
            x2 = min(x + w_box + dx, w)
            y2 = min(y + h_box + dy, h)

            rostro = imagen[y1:y2, x1:x2]
            rostro_redimensionado = cv2.resize(rostro, IMAGE_SIZE)
            cv2.imwrite(OUTPUT_IMAGE, rostro_redimensionado)
            print(f"[✓] Rostro guardado en: {OUTPUT_IMAGE} (confianza: {confianza:.2f})")
    else:
        print(f"[!] No se detectó ningún rostro en la imagen.")
