import os
import cv2
import mediapipe as mp

# === Configuración general ===
INPUT_DIR = "feret_png"        # Ruta con imágenes PNG completas
OUTPUT_DIR = "feret_faces_png" # Ruta de salida con rostros recortados
IMAGE_SIZE = (160, 160)        # Tamaño final del rostro
MIN_CONFIDENCE = 0.96           # Umbral de confianza (puedes cambiarlo)

# Inicializar el detector de rostros de MediaPipe
mp_face_detection = mp.solutions.face_detection

def recortar_y_guardar_rostro(input_path, output_path, min_conf=0.8, margen=0):
    imagen = cv2.imread(input_path)
    if imagen is None:
        print(f"[✗] No se pudo abrir: {input_path}")
        return False

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_conf) as face_detection:
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultados = face_detection.process(imagen_rgb)

        if resultados.detections:
            # Tomar la detección más confiable
            deteccion = max(resultados.detections, key=lambda d: d.score[0])
            confianza = deteccion.score[0]

            if confianza < min_conf:
                print(f"[!] Confianza baja ({confianza:.2f}) en {input_path}")
                return False

            bbox = deteccion.location_data.relative_bounding_box
            h, w, _ = imagen.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)

            # Añadir margen al recorte
            dx = int(w_box * margen)
            dy = int(h_box * margen)
            x1 = max(x - dx, 0)
            y1 = max(y - dy, 0)
            x2 = min(x + w_box + dx, w)
            y2 = min(y + h_box + dy, h)

            rostro = imagen[y1:y2, x1:x2]
            rostro_redimensionado = cv2.resize(rostro, IMAGE_SIZE)
            cv2.imwrite(output_path, rostro_redimensionado)
            print(f"[✓] Rostro guardado: {output_path} (confianza={confianza:.2f})")
            return True
        else:
            print(f"[!] No se detectó rostro en {input_path}")
            return False

# === Procesar toda la base de datos ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

for usuario in os.listdir(INPUT_DIR):
    user_input_dir = os.path.join(INPUT_DIR, usuario)
    if not os.path.isdir(user_input_dir):
        continue

    user_output_dir = os.path.join(OUTPUT_DIR, usuario)
    os.makedirs(user_output_dir, exist_ok=True)

    for file in os.listdir(user_input_dir):
        if file.endswith(".png"):
            input_path = os.path.join(user_input_dir, file)
            output_path = os.path.join(user_output_dir, file)
            recortar_y_guardar_rostro(input_path, output_path, min_conf=MIN_CONFIDENCE)
