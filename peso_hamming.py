import os
import numpy as np

def hamming_weight(bin_str):

    return sum(1 for bit in bin_str if bit == '1')

INPUT_DIR = "FERET_BBDD"
# INPUT_DIR = "embeddings512"
pesos_hamming = [] # Se guardan los pesos hamming de todos los embeddings para luego calcular cual es la media de peso

# Recorrer todas las carpetas (usuarios)
for user_folder in sorted(os.listdir(INPUT_DIR)):
    user_path = os.path.join(INPUT_DIR, user_folder)
    
    if not os.path.isdir(user_path):
        continue  # Saltar si no es una carpeta

    # Leer todos los archivos .txt dentro del usuario
    for file_name in sorted(os.listdir(user_path)):
        if file_name.lower().endswith(".txt"):
            file_path = os.path.join(user_path, file_name)

            with open(file_path, "r") as f:
                binary_embedding = f.read().strip()
            
            pesos_hamming.append(hamming_weight(binary_embedding))

media = sum(pesos_hamming) / len(pesos_hamming)
print(f"La media de los pesos hamming con vectores de 128 es: {media}")