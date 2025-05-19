from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly.express as px
import plotly.io as pio
import json

app = Flask(__name__)

import plotly.graph_objects as go
import numpy as np, json, plotly.io as pio

def compute_graph(params: dict):
    # ① Lee las opciones del usuario (usa defaults si faltan)
    m1 = float(params.get("m1", 1))
    b1 = float(params.get("b1", 0))
    m2 = float(params.get("m2", -1))
    b2 = float(params.get("b2", 5))

    # ② Calcula las rectas
    x = np.linspace(-10, 10, 400)
    y1 = m1 * x + b1
    y2 = m2 * x + b2

    # ③ Figura y trazas
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode="lines",
                             name=f"y = {m1}·x + {b1}"))
    fig.add_trace(go.Scatter(x=x, y=y2, mode="lines",
                             name=f"y = {m2}·x + {b2}"))

    # ④ Intersección (si las pendientes no son iguales)
    if m1 != m2:
        xi = (b2 - b1) / (m1 - m2)
        yi = m1 * xi + b1
        fig.add_trace(go.Scatter(x=[xi], y=[yi], mode="markers",
                                 marker=dict(size=10),
                                 name=f"Intersección ({xi:.2f}, {yi:.2f})"))
        fig.update_layout(title="Dos rectas y su punto de corte")
    else:
        fig.update_layout(title="Rectas paralelas (sin corte)")

    fig.update_layout(margin=dict(l=20, r=20, t=60, b=30),
                      xaxis_title="x", yaxis_title="y")
    return json.loads(pio.to_json(fig))


@app.route("/api/plot", methods=["POST"])
def api_plot():
    params = request.json or {}
    print(f'Parametros recibidos: {params}')
    fig_json = compute_graph(params)
    print(f'gráfica devuelta {fig_json}')
    return jsonify(fig_json)

@app.route("/")
def index():
    return render_template("index.html")   # formulario + contenedor de la gráfica

if __name__ == "__main__":
    app.run(ssl_context=("ssl/cert.pem", "ssl/key.pem"))

