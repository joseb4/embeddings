# err_module.py

import os
import random
import itertools
import numpy as np
import plotly.graph_objs as go

# --------------------------------------------------
# 🔧 Helpers de binarización
# --------------------------------------------------
def _binarize_3bits(emb: np.ndarray, th: float) -> np.ndarray:
    out = []
    for v in emb:
        if v <= -th:
            out.extend([0,0,0])
        elif v <= 0:
            out.extend([0,0,1])
        elif v <= th:
            out.extend([0,1,1])
        else:
            out.extend([1,1,1])
    return np.array(out, dtype=np.uint8)

def _binarize_4bits(emb: np.ndarray, th_low: float, th_high: float) -> np.ndarray:
    out = []
    for v in emb:
        if v <= -th_high:
            out.extend([0,0,0,0])
        elif v <= -th_low:
            out.extend([0,0,0,1])
        elif v <= th_low:
            out.extend([0,0,1,1])
        elif v <= th_high:
            out.extend([0,1,1,1])
        else:
            out.extend([1,1,1,1])
    return np.array(out, dtype=np.uint8)

# --------------------------------------------------
# 🔃 Carga y binarización en memoria
# --------------------------------------------------
def load_float_embeddings(dataset_dir: str,
                          float_dim: int) -> dict[str, np.ndarray]:
    """Carga todos los .npy de `dataset_dir` con dimensión float_dim."""
    data = {}
    for fn in os.listdir(dataset_dir):
        if not fn.endswith(".npy"):
            continue
        arr = np.load(os.path.join(dataset_dir, fn))
        if arr.ndim == 2 and arr.shape[1] == float_dim:
            data[fn] = arr
    return data

def binarize_all(data_f: dict[str, np.ndarray],
                 bits: int,
                 t1: float=None,
                 t2: float=None) -> dict[str, np.ndarray]:
    """
    Binariza todos los embeddings según bits=3|4.
      - bits=3: usa t1
      - bits=4: usa t1 (lower) y t2 (higher)
    """
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

# --------------------------------------------------
# ⚙️ Generación de pares y distancia
# --------------------------------------------------
def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int((a != b).sum())

def generate_pairs(data_b: dict[str, np.ndarray],
                   num_identities=4000,
                   num_impostor_pairs=100_000):
    """Devuelve (genuinos, impostores) donde cada lista es [(emb1, emb2), ...]."""
    # genuinos
    personas = [k for k,v in data_b.items() if v.shape[0] >= 2]
    random.shuffle(personas)
    muest = personas[:min(num_identities, len(personas))]
    genuinos = []
    for p in muest:
        m = data_b[p]
        for i,j in itertools.combinations(range(m.shape[0]), 2):
            genuinos.append((m[i], m[j]))
    # impostores
    impostores = []
    claves = list(data_b.keys())
    for _ in range(num_impostor_pairs):
        p1, p2 = random.sample(claves, 2)
        e1 = data_b[p1][random.randrange(data_b[p1].shape[0])]
        e2 = data_b[p2][random.randrange(data_b[p2].shape[0])]
        impostores.append((e1, e2))
    return genuinos, impostores

def _calc_distances(pares: list[tuple[np.ndarray,np.ndarray]]) -> list[int]:
    return [ _hamming(a,b) for a,b in pares ]

# --------------------------------------------------
# 📊 Evaluación de umbrales y EER
# --------------------------------------------------
def evaluate_thresholds(dist_g: list[int],
                        dist_i: list[int],
                        ths: list[int]) -> tuple[list[float], list[float]]:
    total_g = len(dist_g)
    total_i = len(dist_i)
    fars, frrs = [], []
    for th in ths:
        fars.append( sum(d <= th for d in dist_i) / total_i * 100 )
        frrs.append( sum(d >  th for d in dist_g) / total_g * 100 )
    return fars, frrs

def find_eer(ths: list[int],
             fars: list[float],
             frrs: list[float]) -> tuple[int, float]:
    best, eer, t_eer = float("inf"), 0.0, ths[0]
    for t,f,fr in zip(ths, fars, frrs):
        d = abs(f - fr)
        if d < best:
            best, eer, t_eer = d, (f+fr)/2, t
    return t_eer, eer

# --------------------------------------------------
# 🚀 Función pública
# --------------------------------------------------
def compute_err(
    dataset_dir: str,
    float_dim: int,
    bits: int,
    *,
    t1: float | None = None,
    t2: float | None = None,
    num_identities: int = 4_000,
    num_impostor_pairs: int = 100_000,
    save_plot: bool = False,
    output_dir: str | None = None,
) -> dict:
    """
    Calcula FAR, FRR y EER sobre embeddings binarizados y devuelve un JSON-safe dict.

    Parámetros
    ----------
    dataset_dir        Carpeta con los .npy de embeddings en punto flotante.
    float_dim          Dimensión (en float) del embedding original (128, 512…).
    bits               3 ó 4 (número de bits por componente tras binarizar).
    t1, t2             Umbrales de binarización:
                         · bits==3 ➜ t1 (único umbral)
                         · bits==4 ➜ t1=umbral bajo, t2=umbral alto
    num_identities     Personas distintas muestreadas para pares genuinos.
    num_impostor_pairs Número total de pares impostores aleatorios.
    save_plot          Si True, guarda la gráfica como PNG en output_dir.
    output_dir         Carpeta donde escribir la imagen (requerida si save_plot).

    Devuelve
    --------
    dict JSON-serializable con:
      thresholds  Lista de umbrales evaluados (enteros)
      fars        Lista FAR (%) por umbral
      frrs        Lista FRR (%) por umbral
      eer         {'threshold': int, 'value': float}
      plot        Estructura Plotly completa (fig.to_dict())
      plot_path   Ruta del PNG (solo si save_plot=True)
    """

    # 1) Cargar y binarizar
    data_f = load_float_embeddings(dataset_dir, float_dim)
    data_b = binarize_all(data_f, bits, t1=t1, t2=t2)

    # 2) Generar pares y distancias
    genuinos, impostores = generate_pairs(
        data_b, num_identities, num_impostor_pairs
    )
    dist_g = _calc_distances(genuinos)
    dist_i = _calc_distances(impostores)

    # 3) FAR / FRR / EER
    bin_length = next(iter(data_b.values())).shape[1]
    thresholds = list(range(0, bin_length + 1))
    fars, frrs = evaluate_thresholds(dist_g, dist_i, thresholds)
    eer_th, eer_val = find_eer(thresholds, fars, frrs)

    # 4) Gráfica Plotly
    traces = [
        go.Scatter(x=thresholds, y=fars, name="FAR", mode="lines"),
        go.Scatter(x=thresholds, y=frrs, name="FRR", mode="lines"),
        go.Scatter(
            x=[eer_th],
            y=[eer_val],
            name="EER",
            mode="markers+text",
            text=[f"{eer_val:.2f}%"],
            textposition="top right",
        ),
    ]

    layout = go.Layout(
        title="FAR vs FRR con EER",
        xaxis=dict(title="Umbral de Hamming"),
        yaxis=dict(title="Tasa (%)"),
        shapes=[
            dict(
                type="line",
                x0=eer_th,
                x1=eer_th,
                y0=0,
                y1=max(max(fars), max(frrs)),
                line=dict(dash="dash"),
            )
        ],
    )

    fig = go.Figure(data=traces, layout=layout)
    plot_dict = fig.to_dict()  # ← 100 % JSON-serializable

    # 5) Guardar PNG si procede
    plot_path = None
    if save_plot:
        if output_dir is None:
            raise ValueError("Si save_plot=True debes indicar output_dir")
        os.makedirs(output_dir, exist_ok=True)
        fname = f"err_dim{float_dim}_{bits}bits"
        if bits == 3:
            fname += f"_t1{t1}"
        else:
            fname += f"_t2{t1}_t3{t2}"
        fname += ".png"
        fig.write_image(os.path.join(output_dir, fname))
        plot_path = os.path.join(output_dir, fname)

    # 6) Empaquetar resultado
    return {
        "thresholds": thresholds,
        "fars": fars,
        "frrs": frrs,
        "eer": {"threshold": eer_th, "value": eer_val},
        "plot": plot_dict,
        **({"plot_path": plot_path} if plot_path else {}),
    }
