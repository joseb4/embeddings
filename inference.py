import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# Ruta al modelo TFLite
model_path = "facenet512.tflite"

# Cargar el modelo
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extraer tamaño esperado (ej. [1, 160, 160, 3])
input_shape = input_details[0]['shape']
img_height, img_width = input_shape[1], input_shape[2]

# 🔁 Función para preprocesar una imagen
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_width, img_height))

    img_np = np.asarray(img).astype(np.float32)
    img_np = (img_np - 127.5) / 128.0  # Normalización para FaceNet

    img_np = np.expand_dims(img_np, axis=0)  # Shape: (1, H, W, 3)
    return img_np

# Binarización de los embeddings
def binarize_embedding_3bits(embedding, th):
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


# Cargar y procesar imagen
image_data = preprocess_image("aaron.png")

# Pasar imagen al modelo
interpreter.set_tensor(input_details[0]['index'], image_data)
interpreter.invoke()

# Obtener salida (vector 512D)
embedding = interpreter.get_tensor(output_details[0]['index'])  # Shape: (1, 512)
bin_embbeding = binarize_embedding_4bits(embedding[0], 0.6, 0.2)
# print(f"Vector de embedding (512 dimensiones):\n{embedding[0]}")

print(f"Longitud de los embeddings: {max(list(embedding[0]))}")
