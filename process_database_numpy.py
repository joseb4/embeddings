import os
import numpy as np
from PIL import Image

from tflite_runtime.interpreter import Interpreter

# === CONFIGURACIÓN ===
TFLITE_MODEL_PATH = "facenet128.tflite"
INPUT_DIR = "feret_faces_png"
OUTPUT_DIR = "embeddings128_float_feret"

# === Cargar modelo TFLite ===
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Preprocesamiento ===
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((160, 160))  # Por si acaso
    img_np = np.asarray(img).astype(np.float32)
    img_np = (img_np - 127.5) / 128.0
    img_np = np.expand_dims(img_np, axis=0)
    return img_np


# === Procesamiento ===
user_id = 1  

for user_folder in sorted(os.listdir(INPUT_DIR)):
    user_path = os.path.join(INPUT_DIR, user_folder)
    if not os.path.isdir(user_path):
        continue

    # Asignar nombre numérico a la carpeta
    user_code = f"{user_id:03d}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_path = os.path.join(OUTPUT_DIR, f"{user_code}.npy")

    embeddings_list = []

    for image_name in sorted(os.listdir(user_path)):
        if image_name.lower().endswith(".png"):
            image_path = os.path.join(user_path, image_name)
            img_input = preprocess_image(image_path)

            # Inferencia
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            embedding = interpreter.get_tensor(output_details[0]['index'])[0]

            embeddings_list.append(embedding)
            #print(f"[✓] Embedding extraído de {image_path}")

    # Convertir lista a array y guardar como .npy
    if embeddings_list:
        embeddings_array = np.stack(embeddings_list)  # shape = (N, D)
        np.save(output_file_path, embeddings_array)
        print(f"[💾] Embeddings del usuario {user_code} guardados en {output_file_path}")

    user_id += 1
