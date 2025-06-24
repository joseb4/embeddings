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
# 🔧 CONFIGURABLE VARIABLES
# --------------------------------
DATASET_DIR = "embeddings128_3bits_LFW_nuevosumbrales"
EMBEDDING_LENGTH = 128*3       
NUM_IDENTITIES = 4000             # Number of identities to sample for genuine pairs
NUM_IMPOSTOR_PAIRS = 100000       # Number of random impostor pairs
# THRESHOLDS = list(range(1, EMBEDDING_LENGTH + 1)) 
THRESHOLDS = list(range(1, 450)) 
# --------------------------------
# ⚙️ FUNCTION TO CALCULATE HAMMING DISTANCE
# --------------------------------
def hamming(arr1, arr2):
    """
    Calculates the Hamming distance between two 1D binary arrays (0/1).
    """
    # Using NumPy for comparison (faster than zip on large lists).
    return np.sum(arr1 != arr2)

# --------------------------------
# 🔃 FUNCTION TO LOAD EMBEDDINGS FROM .NPY FILES
# --------------------------------
def load_embeddings(dataset_dir, embedding_length):
    """
    Loads all .npy files into a dictionary.
      - Key: file name (or an ID extracted from the name).
      - Value: 2D numpy array with samples [n_samples, embedding_length].
    """
    data = {}
    files = [f for f in os.listdir(dataset_dir) if f.endswith(".npy")]

    for file in tqdm(files, desc="Loading .npy"):
        path = os.path.join(dataset_dir, file)
        matrix = np.load(path)  # matrix.shape = (N, M)
        
        # If you want to filter by exact embedding_length:
        # (If you don't want to filter, remove this if.)
        if matrix.shape[1] == embedding_length:
            data[file] = matrix
        else:
            print(f"Skipping {file} due to mismatched size: {matrix.shape[1]} vs {embedding_length}")
            pass

    return data


def load_float_embeddings(dataset_dir: str,
                          float_dim: int) -> dict[str, np.ndarray]:
    """
    Loads float embeddings from dataset_dir
    and returns a dict {name: array} where the array has
    shape (num_samples, float_dim).
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Expected a directory: {dataset_dir}")
    if not os.access(dataset_dir, os.R_OK):
        raise PermissionError(f"Access denied: {dataset_dir}")
    if float_dim <= 0:
        raise ValueError(f"Invalid dimension: {float_dim}")
    if not os.listdir(dataset_dir):
        raise ValueError(f"Empty directory: {dataset_dir}")
    # Load embeddings and filter by dimension
    data = {}
    for fn in os.listdir(dataset_dir):
        if not fn.endswith(".npy"):
            continue
        arr = np.load(os.path.join(dataset_dir, fn))
        if arr.ndim == 2 and arr.shape[1] == float_dim:
            data[fn] = arr
    return data

def _binarize_3bits(embedding, th):
    bin_embedding = []
    for emb in embedding:
        if emb <= -th:
            bin_embedding.extend([0, 0, 0])
        elif -th < emb <= 0:
            bin_embedding.extend([0, 0, 1])
        elif 0 < emb <= th:
            bin_embedding.extend([0, 1, 1])
        else:
            bin_embedding.extend([1, 1, 1])
    return np.array(bin_embedding, dtype=np.uint8)


def _binarize_4bits(embedding, th1, th2):
    bin_embedding = []
    for emb in embedding:
        if emb <= -th1:
            bin_embedding.extend([0, 0, 0, 0])
        elif -th1 < emb <= -th2:
            bin_embedding.extend([0, 0, 0, 1])
        elif -th2 < emb <= th2:
            bin_embedding.extend([0, 0, 1, 1])
        elif th2 < emb <= th1:
            bin_embedding.extend([0, 1, 1, 1])
        else:
            bin_embedding.extend([1, 1, 1, 1])
    return np.array(bin_embedding, dtype=np.uint8)

def binarize_all(data_f: dict[str, np.ndarray],
                 bits: int,
                 t1: float=None,
                 t2: float=None) -> dict[str, np.ndarray]:
    """
    Binarizes all embeddings in data_f and returns a dict
    {name: array} where the array has shape (num_samples, bits*dim).
    bits: 3 or 4 (number of bits per component after binarization).
    t1, t2: binarization thresholds:
      · bits==3 ➜ t1 (single threshold)
      · bits==4 ➜ t1=low threshold, t2=high threshold
    """
    if bits not in (3, 4):
        raise ValueError(f"bits must be 3 or 4, not {bits}")
    if bits == 3 and t1 is None:
        raise ValueError("For 3 bits you need t1")
    if bits == 4 and (t1 is None or t2 is None):
        raise ValueError("For 4 bits you need t1 (low threshold) and t2 (high threshold)")
    # Binarize
    out = {}
    for name, mat in data_f.items():
        bins = []
        for emb in mat:
            if bits == 3:
                if t1 is None:
                    raise ValueError("For 3 bits you need t1")
                bins.append(_binarize_3bits(emb, t1))
            else:
                if t1 is None or t2 is None:
                    raise ValueError("For 4 bits you need t2 (low threshold) and t3 (high threshold)")
                # here we map t1→th_low, t2→th_high
                bins.append(_binarize_4bits(emb, t1, t2))
        out[name] = np.stack(bins, axis=0)
    return out



# --------------------------------
# 🔍 FUNCTION TO GENERATE GENUINE AND IMPOSTOR PAIRS
# --------------------------------
def gen_pairs(data):
    """
    Generates lists of genuine and impostor pairs.
      - Genuine pairs come from different samples of the same person.
      - Impostor pairs come from samples of different people.
    """
    genuines = []
    impostors = []

    # Filter only users with 2 or more samples
    valid_people = [p for p in data if data[p].shape[0] >= 2]
    random.shuffle(valid_people)  # Shuffle for sampling
    sampled_people = valid_people[:min(NUM_IDENTITIES, len(valid_people))]

    print(f"People selected for genuine pairs: {len(sampled_people)}")

    # Generate genuine pairs
    for person in tqdm(valid_people, desc="Generating genuine pairs"):
        samples = data[person]  # shape (N, EMBEDDING_LENGTH)
        # Combine indices to form pairs among all samples
        indices = list(range(samples.shape[0]))
        for i1, i2 in itertools.combinations(indices, 2):
            genuines.append((samples[i1], samples[i2]))

    # Generate impostor pairs
    people = list(data.keys())
    for _ in tqdm(range(NUM_IMPOSTOR_PAIRS), desc="Generating impostor pairs"):
        p1, p2 = random.sample(people, 2)  # two different users
        # Pick a random sample from each
        e1 = data[p1][random.randint(0, data[p1].shape[0] - 1)]
        e2 = data[p2][random.randint(0, data[p2].shape[0] - 1)]
        impostors.append((e1, e2))

    return genuines, impostors

# --------------------------------
# ⚙️ CALCULATE HAMMING DISTANCES
# --------------------------------
def cal_distances(pairs):
    """
    Given a list of pairs (embedding1, embedding2),
    calculates the Hamming distance for each pair and returns it as a list.
    """
    distances = []
    for (p1, p2) in pairs:
        distances.append(hamming(p1, p2))
    return distances

# --------------------------------
# 📊 EVALUATE THRESHOLD (FAR and FRR)
# --------------------------------
def evaluate_threshold(dist_genuine, dist_impostor, thresholds):
    """
    Given the distances of genuine and impostor pairs,
    calculates FAR and FRR for each value in 'thresholds'.
    Returns two parallel lists (fars, frrs).
    """
    fars, frrs = [], []
    total_genuine = len(dist_genuine)
    total_impostor = len(dist_impostor)

    for threshold in thresholds:
        # FAR: proportion of impostors whose distance is below (or equal to) the threshold
        FAR = sum(d <= threshold for d in dist_impostor) / total_impostor
        # FRR: proportion of genuines whose distance is above the threshold
        FRR = sum(d > threshold for d in dist_genuine) / total_genuine

        fars.append(FAR * 100)  # Convert to %
        frrs.append(FRR * 100)

    return fars, frrs

# --------------------------------
# 🔎 EER CALCULATION
# --------------------------------
def find_eer(thresholds, fars, frrs):
    """
    Finds the EER (Equal Error Rate) by searching for the point where |FAR - FRR| is minimal.
    Returns the corresponding threshold and the EER value in %.
    """
    min_diff = float("inf")
    eer = None
    eer_threshold = None

    for u, far, frr in zip(thresholds, fars, frrs):
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2
            eer_threshold = u

    return eer_threshold, eer

# --------------------------------
# 📈 PLOT RESULTS
# --------------------------------

def plot_interactive(thresholds, fars, frrs, show=True):
    # Calculate the EER and the threshold where it occurs
    eer_threshold, eer = find_eer(thresholds, fars, frrs)
    print(f"\n🔍 EER found at threshold = {eer_threshold} with rate (EER) ≈ {eer:.2f}%")

    fig = go.Figure()

    # FAR curve
    fig.add_trace(go.Scatter(x=thresholds, y=fars, mode='lines', name='FAR (False Acceptance Rate)'))
    # FRR curve
    fig.add_trace(go.Scatter(x=thresholds, y=frrs, mode='lines', name='FRR (False Rejection Rate)'))
    # EER vertical line
    fig.add_shape(
        type="line",
        x0=eer_threshold, x1=eer_threshold,
        y0=0, y1=eer,
        line=dict(color="gray", dash="dash"),
        name="EER Threshold"
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
        title="FAR vs FRR Curve with EER Point",
        xaxis_title="Hamming Threshold",
        yaxis_title="Rate (%)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white"
    )

    if show:
        fig.show()

def compute_err(
    data_b: dict[str, np.ndarray],
):
    """
    Calculates FAR, FRR, and EER on binarized embeddings and shows the plot.
    """
    # 1) Load and binarize
    # data_f = load_float_embeddings(dataset_dir, float_dim)
    # data_b = binarize_all(data_f, bits, t1=t1, t2=t2)

    # 2) Generate pairs and distances
    genuines, impostors = gen_pairs(data_b)
    dist_g = cal_distances(genuines)
    dist_i = cal_distances(impostors)

    # 3) FAR / FRR / EER
    bin_length = next(iter(data_b.values())).shape[1]
    thresholds = list(range(0, bin_length + 1))
    fars, frrs = evaluate_threshold(dist_g, dist_i, thresholds)
    eer_th, eer_val = find_eer(thresholds, fars, frrs)

    # 4) Plot with Plotly
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
        title="FAR vs FRR with EER",
        xaxis_title="Hamming Threshold",
        yaxis_title="Rate (%)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white"
    )
    print(f"\n🔍 EER found at threshold = {eer_th} with rate (EER) ≈ {eer_val:.2f}%")
    fig.show()

# --------------------------------
# 🚀 MAIN
# --------------------------------
def main(show=True):
    print("Loading embeddings from .npy files...")
    data = load_embeddings(DATASET_DIR, EMBEDDING_LENGTH)

    print("Generating genuine and impostor pairs...")
    genuines, impostors = gen_pairs(data)

    print("Calculating Hamming distances...")
    dist_g = cal_distances(genuines)
    dist_i = cal_distances(impostors)

    print("Evaluating metrics for each threshold...")
    fars, frrs = evaluate_threshold(dist_g, dist_i, THRESHOLDS)

    print("Showing results...")
    plot_interactive(THRESHOLDS, fars, frrs, show=show)

if __name__ == "__main__":
    main(True)