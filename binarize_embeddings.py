import os
import numpy as np

# === Binarización de los embeddings ===
def binarize_embedding(embedding, th):
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

# === Procesamiento principal ===
def procesar_embeddings(input_dir, output_dir, mode="3bits", th=None, th1=None, th2=None):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.endswith(".npy"):
            continue

        input_path = os.path.join(input_dir, file)
        embeddings = np.load(input_path)  # shape: (N, D)
        embeddings_binarizados = []

        for emb in embeddings:
            if mode == "3bits":
                emb_bin = binarize_embedding(emb, th)
            elif mode == "4bits":
                if th1 is None or th2 is None:
                    raise ValueError("Debes indicar 'th1' y 'th2' para modo 4bits.")
                emb_bin = binarize_embedding_4bits(emb, th1, th2)
            else:
                raise ValueError("Modo no reconocido. Usa '3bits' o '4bits'.")

            embeddings_binarizados.append(emb_bin)

        # Guardar
        output_path = os.path.join(output_dir, file)
        np.save(output_path, np.stack(embeddings_binarizados))
        print(f"[✓] Guardado: {output_path}")

THESHOLD = 0.1      # UMBRAL PARA BINARIZACIÓN DE 3 BITS
THESHOLD1 = 0.12    # UMBRAL PARA BINARIZACIÓN DE 4 BITS
THESHOLD2 = 0.04    # UMBRAL PARA BINARIZACIÓN DE 4 BITS

def binariza_todo():# LFW 512 floats
    procesar_embeddings(
        input_dir="embeddings512_float_LFW",
        output_dir="embeddings512_4bits_LFW",
        mode="4bits",
        th1=THESHOLD1,
        th2=THESHOLD2
    )

    procesar_embeddings(
        input_dir="embeddings512_float_LFW",
        output_dir="embeddings512_3bits_LFW",
        mode="3bits",
        th=THESHOLD
    )

    # FERET 512 floats
    procesar_embeddings(
        input_dir="embeddings512_float_feret",
        output_dir="embeddings512_4bits_feret",
        mode="4bits",
        th1=THESHOLD1,
        th2=THESHOLD2
    )

    procesar_embeddings(
        input_dir="embeddings512_float_feret",
        output_dir="embeddings512_3bits_feret",
        mode="3bits",
        th=THESHOLD
    )

    # LFW 128 floats
    procesar_embeddings(
        input_dir="embeddings128_float_LFW",
        output_dir="embeddings128_4bits_LFW",
        mode="4bits",
        th1=THESHOLD1,
        th2=THESHOLD2
    )

    procesar_embeddings(
        input_dir="embeddings128_float_LFW",
        output_dir="embeddings128_3bits_LFW",
        mode="3bits",
        th=THESHOLD
    )

    # feret 128 floats
    procesar_embeddings(
        input_dir="embeddings128_float_feret",
        output_dir="embeddings128_4bits_feret",
        mode="4bits",
        th1=THESHOLD1,
        th2=THESHOLD2
    )

    procesar_embeddings(
        input_dir="embeddings128_float_feret",
        output_dir="embeddings128_3bits_feret",
        mode="3bits",
        th=THESHOLD
    )




procesar_embeddings(
        input_dir="embeddings128_float_LFW",
        output_dir="embeddings128_3bits_LFW_nuevosumbrales_015_02",
        mode="3bits",
        th=0.15
)

