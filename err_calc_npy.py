import os
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# --------------------------------
# 🔧 VARIABLES CONFIGURABLES
# --------------------------------
DATASET_DIR = "embeddings128_3bits_LFW_nuevosumbrales"
EMBEDDING_LENGTH = 128*3       
NUM_IDENTIDADES = 4000             # Número de identidades a muestrear para pares genuinos
NUM_IMPOSTOR_PAIRS = 100000        # Número de pares impostores aleatorios
# THRESHOLDS = list(range(1, EMBEDDING_LENGTH + 1)) 
THRESHOLDS = list(range(1, 450)) 
# --------------------------------
# ⚙️ FUNCIÓN PARA CALCULAR DISTANCIA DE HAMMING
# --------------------------------
def hamming(arr1, arr2):
    """
    Calcula la distancia de Hamming entre dos arrays 1D binarios (0/1).
    """
    # Usando NumPy para comparar (más rápido que zip en listas grandes).
    return np.sum(arr1 != arr2)

# --------------------------------
# 🔃 FUNCIÓN PARA CARGAR EMBEDDINGS DESDE FICHEROS .NPY
# --------------------------------
def cargar_embeddings(dataset_dir, embedding_length):
    """
    Carga todos los ficheros .npy en un diccionario.
      - Clave: nombre del fichero (o un ID extraído del nombre).
      - Valor: array numpy 2D con las muestras [n_samples, embedding_length].
    """
    data = {}
    files = [f for f in os.listdir(dataset_dir) if f.endswith(".npy")]

    for fichero in tqdm(files, desc="Cargando .npy"):
        path = os.path.join(dataset_dir, fichero)
        matriz = np.load(path)  # matriz.shape = (N, M)
        
        # Si queremos filtrar por embedding_length exacto:
        # (Si no deseas filtrar, elimina este if.)
        if matriz.shape[1] == embedding_length:
            data[fichero] = matriz
        else:
            print(f"Saltando {fichero} por tamaño no coincidente: {matriz.shape[1]} vs {embedding_length}")
            pass

    return data

# --------------------------------
# 🔍 FUNCIÓN PARA GENERAR PARES GENUINOS E IMPOSTORES
# --------------------------------
def generar_pares(data):
    """
    Genera listas de pares genuinos e impostores.
      - Los pares genuinos provienen de muestras distintas de la misma persona.
      - Los pares impostores provienen de muestras de diferentes personas.
    """
    genuinos = []
    impostores = []

    # Filtramos solo usuarios con 2 o más muestras
    personas_validas = [p for p in data if data[p].shape[0] >= 2]
    random.shuffle(personas_validas)  # Mezclamos para muestrear
    personas_muestreadas = personas_validas[:min(NUM_IDENTIDADES, len(personas_validas))]

    print(f"Personas seleccionadas para pares genuinos: {len(personas_muestreadas)}")

    # Generar pares genuinos
    for persona in tqdm(personas_muestreadas, desc="Generando pares genuinos"):
        muestras = data[persona]  # shape (N, EMBEDDING_LENGTH)
        # Combinamos índices para formar pares entre todas las muestras
        indices = list(range(muestras.shape[0]))
        for i1, i2 in itertools.combinations(indices, 2):
            genuinos.append((muestras[i1], muestras[i2]))

    # Generar pares impostores
    personas = list(data.keys())
    for _ in tqdm(range(NUM_IMPOSTOR_PAIRS), desc="Generando pares impostores"):
        p1, p2 = random.sample(personas, 2)  # dos usuarios distintos
        # Escogemos una muestra aleatoria de cada uno
        e1 = data[p1][random.randint(0, data[p1].shape[0] - 1)]
        e2 = data[p2][random.randint(0, data[p2].shape[0] - 1)]
        impostores.append((e1, e2))

    return genuinos, impostores

# --------------------------------
# ⚙️ CALCULAR DISTANCIAS DE HAMMING
# --------------------------------
def calcular_distancias(pares):
    """
    Dada una lista de pares (embedding1, embedding2),
    calcula la distancia de Hamming para cada par y la devuelve como lista.
    """
    distancias = []
    for (p1, p2) in pares:
        distancias.append(hamming(p1, p2))
    return distancias

# --------------------------------
# 📊 EVALUAR UMBRAL (FAR y FRR)
# --------------------------------
def evaluar_umbral(dist_genuinos, dist_impostores, umbrales):
    """
    Dadas las distancias de los pares genuinos y de los impostores,
    calcula FAR y FRR para cada valor de umbral en 'umbrales'.
    Devuelve dos listas paralelas (fars, frrs).
    """
    fars, frrs = [], []
    total_genuinos = len(dist_genuinos)
    total_impostores = len(dist_impostores)

    for umbral in umbrales:
        # FAR: proporción de impostores cuya distancia está por debajo (o igual) al umbral
        FAR = sum(d <= umbral for d in dist_impostores) / total_impostores
        # FRR: proporción de genuinos cuya distancia está por encima del umbral
        FRR = sum(d > umbral for d in dist_genuinos) / total_genuinos

        fars.append(FAR * 100)  # Convertimos a %
        frrs.append(FRR * 100)

    return fars, frrs

# --------------------------------
# 🔎 CÁLCULO DE EER
# --------------------------------
def encontrar_eer(umbrales, fars, frrs):
    """
    Encuentra el EER (Equal Error Rate) buscando el punto donde |FAR - FRR| es mínimo.
    Devuelve el umbral correspondiente y el valor de EER en %.
    """
    min_diff = float("inf")
    eer = None
    eer_threshold = None

    for u, far, frr in zip(umbrales, fars, frrs):
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2
            eer_threshold = u

    return eer_threshold, eer

# --------------------------------
# 📈 GRAFICAR RESULTADOS
# --------------------------------
def graficar(umbrales, fars, frrs, show=True):
    """
    Genera la gráfica de FAR vs FRR y muestra el punto EER.
    """
    # Calculamos el EER y el umbral en el que ocurre
    eer_threshold, eer = encontrar_eer(umbrales, fars, frrs)
    print(f"\n🔍 EER encontrado en umbral = {eer_threshold} con tasa (EER) ≈ {eer:.2f}%")

    # Gráfica
    plt.plot(umbrales, fars, label="FAR (False Acceptance Rate)")
    plt.plot(umbrales, frrs, label="FRR (False Rejection Rate)")

    # Punto del EER
    plt.axvline(x=eer_threshold, color='gray', linestyle='--', label=f"Umbral EER = {eer_threshold}")
    plt.scatter([eer_threshold], [eer], color='red', zorder=5)
    plt.text(eer_threshold + 1, eer + 1, f"EER ≈ {eer:.2f}%", color='red')

    plt.xlabel("Umbral de Hamming")
    plt.ylabel("Tasa (%)")
    plt.title("Curva FAR vs FRR con Punto EER")
    plt.grid(True)
    plt.legend()
    if show:
        plt.show()

# --------------------------------
# 🚀 MAIN
# --------------------------------
def main(show=True):
    print("Cargando embeddings desde ficheros .npy...")
    data = cargar_embeddings(DATASET_DIR, EMBEDDING_LENGTH)

    print("Generando pares genuinos e impostores...")
    genuinos, impostores = generar_pares(data)

    print("Calculando distancias de Hamming...")
    dist_g = calcular_distancias(genuinos)
    dist_i = calcular_distancias(impostores)

    print("Evaluando métricas para cada umbral...")
    fars, frrs = evaluar_umbral(dist_g, dist_i, THRESHOLDS)

    print("Mostrando resultados...")
    graficar(THRESHOLDS, fars, frrs, show=show)

if __name__ == "__main__":
    main(True)
