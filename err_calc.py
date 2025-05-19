import os
import random
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
# from scipy.spatial.distance import hamming


from tqdm import tqdm  

# -------------------------------
# 🔧 VARIABLES CONFIGURABLES
# -------------------------------
DATASET_DIR = "embeddings512_4bits_02" 
# EMBEDDING_LENGTH = 512*4
# DATASET_DIR = "FERET_BBDD" 
EMBEDDING_LENGTH = 512*4 
# EMBEDDING_LENGTH = (128*2*3)-1  # Esto es para el formato de los embeddings que extrajo Paula de la base de FERET
NUM_IDENTIDADES = 4000 
NUM_IMPOSTOR_PAIRS = 100000  # Númeor de pares impostores aleatorios
THRESHOLDS = list(range(1, EMBEDDING_LENGTH + 1))  
# THRESHOLDS = list(range(100, 250))  

def hamming(l1,l2):
    count=0
    for e1,e2 in zip(l1, l2):
        if e1!= e2: count += 1
    return count
    # return sum([1 for e1,e2 in zip(l1,l2) if e1 != e2])

# -------------------------------
# 🔃 FUNCIÓN PARA CARGAR EMBEDDINGS
# -------------------------------
def cargar_embeddings(dataset_dir):
    data = defaultdict(list)
    carpetas = [p for p in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, p))]
    
    for persona in tqdm(carpetas, desc="Cargando personas"):
        persona_path = os.path.join(dataset_dir, persona)
        for archivo in os.listdir(persona_path):
            if archivo.endswith(".txt"):
                with open(os.path.join(persona_path, archivo), 'r') as f:
                    linea = f.read().strip()
                    if len(linea) == EMBEDDING_LENGTH:
                        if len(linea.split()) >2: data[persona].append([int(bit) for bit in linea.split()])                          
                        else: data[persona].append([int(bit) for bit in linea])
                        
    return data

# -------------------------------
# 🔍 FUNCIÓN PARA GENERAR PARES
# -------------------------------
def generar_pares(data):
    genuinos = []
    impostores = []

    personas_validas = [p for p in data if len(data[p]) >= 2] # Solo son válidos los que tienen mas de dos muestras
    personas_muestreadas = random.sample(personas_validas, min(NUM_IDENTIDADES, len(personas_validas)))

    print(f"Personas seleccionadas para pares genuinos: {len(personas_muestreadas)}")

    # Pares genuinos
    for persona in tqdm(personas_muestreadas, desc="Generando pares genuinos"):
        muestras = data[persona]
        pares = list(itertools.combinations(muestras, 2))
        genuinos.extend(pares)

    # Pares impostores
    personas = list(data.keys())
    for _ in tqdm(range(NUM_IMPOSTOR_PAIRS), desc="Generando pares impostores"):
        p1, p2 = random.sample(personas, 2)
        if data[p1] and data[p2]:
            e1 = random.choice(data[p1])
            e2 = random.choice(data[p2])
            impostores.append((e1, e2))

    return genuinos, impostores
# -------------------------------
# ⚙️ CALCULAR DISTANCIAS DE HAMMING
# -------------------------------
def calcular_distancias(pares):
    return [int(hamming(p1, p2) ) for p1, p2 in pares]# * EMBEDDING_LENGTH
 
# -------------------------------
# 📊 GENERAR CURVA FAR/FRR
# -------------------------------
def evaluar_umbral(dist_genuinos, dist_impostores, umbrales):
    fars, frrs = [], []
    total_genuinos = len(dist_genuinos)
    total_impostores = len(dist_impostores)

    for umbral in umbrales:
        FAR = sum(d <= umbral for d in dist_impostores) / total_impostores
        FRR = sum(d > umbral for d in dist_genuinos) / total_genuinos
        fars.append(FAR * 100)
        frrs.append(FRR * 100)

    return fars, frrs

# -------------------------------
# 📈 GRAFICAR RESULTADOS
# -------------------------------
def encontrar_eer(umbrales, fars, frrs):
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

def graficar(umbrales, fars, frrs, show):
    # Calcular el EER y el umbral correspondiente
    eer_threshold, eer = encontrar_eer(umbrales, fars, frrs)
    print(f"\n🔍 EER encontrado en umbral = {eer_threshold} con tasa (err) ≈ {eer:.2f}%")

    # Graficar las curvas
    plt.plot(umbrales, fars, label="FAR (False Acceptance Rate)")
    plt.plot(umbrales, frrs, label="FRR (False Rejection Rate)")

    # Marcar el punto EER en la gráfica
    plt.axvline(x=eer_threshold, color='gray', linestyle='--', label=f"Umbral EER = {eer_threshold}")
    plt.scatter([eer_threshold], [eer], color='red', zorder=5)
    plt.text(eer_threshold + 5, eer + 1, f"EER ≈ {eer:.2f}%", color='red')

    # Etiquetas
    plt.xlabel("Umbral de Hamming")
    plt.ylabel("Tasa (%)")
    plt.title("Curva FAR vs FRR con Punto EER")
    plt.legend()
    plt.grid(True)
    if show: plt.show()



# -------------------------------
# 🚀 MAIN
# -------------------------------
def main(show):
    print("Cargando embeddings...")
    data = cargar_embeddings(DATASET_DIR)

    print("Generando pares genuinos e impostores...")
    genuinos, impostores = generar_pares(data)

    print("Calculando distancias de Hamming...")
    dist_g = calcular_distancias(genuinos) # Deveulve una lista con la distacia hamming entre los pares genuinos
    dist_i = calcular_distancias(impostores) # Devuelve una lista con la distancia hamming entre los pares de impostores

    print("Evaluando métricas para cada umbral...")
    fars, frrs = evaluar_umbral(dist_g, dist_i, THRESHOLDS)

    print("Graficando resultados...")
    graficar(THRESHOLDS, fars, frrs, show)

if __name__ == "__main__":
    main(True)

