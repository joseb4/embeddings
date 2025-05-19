import numpy as np

# Cargar el archivo
data = np.load('embeddings512_4bits_bin_feret_02/003.npy')

# Mostrar el contenido
print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("Contenido:")
print(data)
