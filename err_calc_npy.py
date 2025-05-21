import os
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

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


def load_float_embeddings(dataset_dir: str,
                          float_dim: int) -> dict[str, np.ndarray]:
    """
    Carga los embeddings en punto flotante desde dataset_dir
    y devuelve un dict {nombre: array} donde el array tiene
    dimensión (num_samples, float_dim).
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Directorio no encontrado: {dataset_dir}")
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Se esperaba un directorio: {dataset_dir}")
    if not os.access(dataset_dir, os.R_OK):
        raise PermissionError(f"Acceso denegado: {dataset_dir}")
    if float_dim <= 0:
        raise ValueError(f"Dimensión inválida: {float_dim}")
    if not os.listdir(dataset_dir):
        raise ValueError(f"Directorio vacío: {dataset_dir}")
    # Cargar los embeddings
    # y filtrar por dimensión
    data = {}
    for fn in os.listdir(dataset_dir):
        if not fn.endswith(".npy"):
            continue
        arr = np.load(os.path.join(dataset_dir, fn))
        if arr.ndim == 2 and arr.shape[1] == float_dim:
            data[fn] = arr
    return data

def _binarize_3bits(embedding, th):
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


def _binarize_4bits(embedding, th1, th2):
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

def binarize_all(data_f: dict[str, np.ndarray],
                 bits: int,
                 t1: float=None,
                 t2: float=None) -> dict[str, np.ndarray]:
    """
    Binariza todos los embeddings en data_f y devuelve un dict
    {nombre: array} donde el array tiene dimensión (num_samples, bits*dim).
    bits: 3 o 4 (número de bits por componente tras binarizar).
    t1, t2: umbrales de binarización:
      · bits==3 ➜ t1 (único umbral)
      · bits==4 ➜ t1=umbral bajo, t2=umbral alto
    """
    if bits not in (3, 4):
        raise ValueError(f"bits debe ser 3 o 4, no {bits}")
    if bits == 3 and t1 is None:
        raise ValueError("Para 3 bits necesitas t1")
    if bits == 4 and (t1 is None or t2 is None):
        raise ValueError("Para 4 bits necesitas t1 (umbral bajo) y t2 (umbral alto)")
    # Binarizar
    out = {}
    for name, mat in data_f.items():
        bins = []
        for emb in mat:
            if bits == 3:
                if t1 is None:
                    raise ValueError("Para 3 bits necesitas t1")
                bins.append(_binarize_3bits(emb, t1))
            else:
                if t1 is None or t2 is None:
                    raise ValueError("Para 4 bits necesitas t2 (umbral bajo) y t3 (umbral alto)")
                # aquí mapeamos t1→th_low, t2→th_high
                bins.append(_binarize_4bits(emb, t1, t2))
        out[name] = np.stack(bins, axis=0)
    return out



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
    for persona in tqdm(personas_validas, desc="Generando pares genuinos"):
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

def graficar_interactivo(umbrales, fars, frrs, show=True):
    # Calcula el EER y el umbral en el que ocurre
    eer_threshold, eer = encontrar_eer(umbrales, fars, frrs)
    print(f"\n🔍 EER encontrado en umbral = {eer_threshold} con tasa (EER) ≈ {eer:.2f}%")

    fig = go.Figure()

    # FAR curve
    fig.add_trace(go.Scatter(x=umbrales, y=fars, mode='lines', name='FAR (False Acceptance Rate)'))
    # FRR curve
    fig.add_trace(go.Scatter(x=umbrales, y=frrs, mode='lines', name='FRR (False Rejection Rate)'))
    # EER vertical line
    fig.add_shape(
        type="line",
        x0=eer_threshold, x1=eer_threshold,
        y0=0, y1=eer,
        line=dict(color="gray", dash="dash"),
        name="Umbral EER"
    )
    # EER point
    fig.add_trace(go.Scatter(
        x=[eer_threshold], y=[eer],
        mode='markers+text',
        marker=dict(color='red', size=10),
        text=[f"EER ≈ {eer:.2f}%"],
        textposition="top right",
        name="EER"
    ))

    fig.update_layout(
        title="Curva FAR vs FRR con Punto EER",
        xaxis_title="Umbral de Hamming",
        yaxis_title="Tasa (%)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white"
    )

    if show:
        fig.show()

def compute_err(
    data_b: dict[str, np.ndarray],
):
    """
    Calcula FAR, FRR y EER sobre embeddings binarizados y muestra la gráfica.
    """
    # 1) Cargar y binarizar
    # data_f = load_float_embeddings(dataset_dir, float_dim)
    # data_b = binarize_all(data_f, bits, t1=t1, t2=t2)

    # 2) Generar pares y distancias
    genuinos, impostores = generar_pares(data_b)
    dist_g = calcular_distancias(genuinos)
    dist_i = calcular_distancias(impostores)

    # 3) FAR / FRR / EER
    bin_length = next(iter(data_b.values())).shape[1]
    thresholds = list(range(0, bin_length + 1))
    fars, frrs = evaluar_umbral(dist_g, dist_i, thresholds)
    eer_th, eer_val = encontrar_eer(thresholds, fars, frrs)

    # 4) Gráfica Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=fars, name="FAR", mode="lines"))
    fig.add_trace(go.Scatter(x=thresholds, y=frrs, name="FRR", mode="lines"))
    fig.add_trace(go.Scatter(
        x=[eer_th], y=[eer_val],
        name="EER", mode="markers+text",
        text=[f"{eer_val:.2f}%"], textposition="top right"
    ))
    fig.add_shape(
        type="line",
        x0=eer_th, x1=eer_th,
        y0=0, y1=max(max(fars), max(frrs)),
        line=dict(dash="dash"),
    )
    fig.update_layout(
        title="FAR vs FRR con EER",
        xaxis_title="Umbral de Hamming",
        yaxis_title="Tasa (%)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white"
    )
    print(f"\n🔍 EER encontrado en umbral = {eer_th} con tasa (EER) ≈ {eer_val:.2f}%")
    fig.show()

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
