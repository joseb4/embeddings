import os
import numpy as np

# 📁 Directorio con los archivos .npy
INPUT_DIR = "embeddings128_float_LFW"  

# Inicializamos valores extremos
valor_minimo_global = float("inf")
valor_maximo_global = float("-inf")

# Recorremos todos los archivos .npy
for file_name in sorted(os.listdir(INPUT_DIR)):
    if file_name.lower().endswith(".npy"):
        file_path = os.path.join(INPUT_DIR, file_name)
        
        try:
            matriz = np.load(file_path) 
        except Exception as e:
            print(f"❌ Error al cargar {file_name}: {e}")
            continue

        # Comprobar si la matriz contiene floats
        if not np.issubdtype(matriz.dtype, np.floating):
            print(f"⚠️ {file_name} no contiene floats. Saltando.")
            continue

        # Actualizar mínimos y máximos globales
        valor_min = np.min(matriz)
        valor_max = np.max(matriz)

        if valor_min < valor_minimo_global:
            valor_minimo_global = valor_min
        if valor_max > valor_maximo_global:
            valor_maximo_global = valor_max

# Mostrar resultados finales
if valor_minimo_global == float("inf"):
    print("⚠️ No se encontraron datos válidos.")
else:
    print(f"✅ Valor mínimo encontrado: {valor_minimo_global}")
    print(f"✅ Valor máximo encontrado: {valor_maximo_global}")
