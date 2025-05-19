import os
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# === CONFIGURACIÓN ===
TFLITE_MODEL_PATH = "facenet128.tflite"
INPUT_DIR = "feret_faces_png"
OUTPUT_DIR = "embeddings128_4bits_feret_02"
THRESHOLD = 0.1 # EN EL CÓDIGO DE PAULA LO TIENE A 0.1 -------------------------------------------------------------------------------------------------------------------------------
THRESHOLD1 = 0.12
THRESHOLD2 = 0.04

# === Binarización de los embeddings ===
def binarize_embedding(embedding, th):
    emb_binario = []
    for emb in embedding:
        if emb <= -th:
            emb_binario.extend([0, 0, 0])
        elif -th < emb <= 0:
            emb_binario.extend([0, 0, 1])
        elif 0 < emb <= th:
            emb_binario.extend([0, 1, 1])
        else:
            emb_binario.extend([1, 1, 1])
    return np.array(emb_binario, dtype=np.uint8)

# Binarización de los embeddings
def binarize_embedding_4bits(embedding, th1, th2):
    emb_binario = []
    for emb in embedding:
        if emb <= -th1:
            emb_binario.extend([0, 0, 0, 0])
        elif -th1 < emb <= -th2:
            emb_binario.extend([0, 0, 0, 1])
        elif -th2 < emb <= th2:
            emb_binario.extend([0, 0, 1, 1])
        elif th2 < emb <= th1:
            emb_binario.extend([0, 1, 1, 1])
        else:
            emb_binario.extend([1, 1, 1, 1])
    
    return np.array(emb_binario, dtype=np.uint8)

# === Preprocesamiento ===
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((160, 160))  # Por si acaso
    img_np = np.asarray(img).astype(np.float32)
    img_np = (img_np - 127.5) / 128.0
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

# === Cargar modelo TFLite con tflite_runtime ===
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Procesamiento ===
user_id = 1  

for user_folder in sorted(os.listdir(INPUT_DIR)):
    user_path = os.path.join(INPUT_DIR, user_folder)
    if not os.path.isdir(user_path):
        continue

    # Asignar nombre numérico a la carpeta
    user_code = f"{user_id:03d}"
    output_user_dir = os.path.join(OUTPUT_DIR, user_code)
    os.makedirs(output_user_dir, exist_ok=True)

    # Contador de imagenes dentro del usuario
    img_count = 1

    for image_name in sorted(os.listdir(user_path)):
        if image_name.lower().endswith(".png"):
            image_path = os.path.join(user_path, image_name)
            img_input = preprocess_image(image_path)

            # Inferencia
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            embedding = interpreter.get_tensor(output_details[0]['index'])[0]

            # Binarización
            # bin_embedding = binarize_embedding(embedding, THRESHOLD)
            bin_embedding = binarize_embedding_4bits(embedding, THRESHOLD1, THRESHOLD2)
            bin_str = "".join(map(str, bin_embedding))  
            
            filename = f"{img_count:03d}.txt"
            output_file = os.path.join(output_user_dir, filename)
            with open(output_file, "w") as f:
                f.write(bin_str + "\n")

            print(f"[✓] {output_file} guardado")
            img_count += 1

    user_id += 1  
