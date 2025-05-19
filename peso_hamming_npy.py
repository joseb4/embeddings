import os
import numpy as np

def hamming_weight(array_binario):
    """
    Calcula el peso de Hamming (número de 1s) de un array 1D binario.
    """
    return np.sum(array_binario)

INPUT_DIR = "embeddings512_4bits_bin_feret_02"  # Cambia esto a tu ruta real
pesos_hamming = []

# Recorrer todos los archivos .npy del directorio
for file_name in sorted(os.listdir(INPUT_DIR)):
    if file_name.lower().endswith(".npy"):
        file_path = os.path.join(INPUT_DIR, file_name)
        
        # Cargar el archivo .npy
        try:
            embeddings = np.load(file_path)  # embeddings.shape = (N, M)
        except Exception as e:
            print(f"⚠️ Error al cargar {file_name}: {e}")
            continue
        
        # Calcular peso de Hamming por muestra (fila)
        for fila in embeddings:
            pesos_hamming.append(hamming_weight(fila))

# Calcular media
if pesos_hamming:
    media = sum(pesos_hamming) / len(pesos_hamming)
    print(f"La media de los pesos Hamming es: {media:.2f}")
else:
    print("⚠️ No se encontraron datos válidos para calcular la media.")
