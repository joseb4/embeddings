/* static/app.js */
document.addEventListener('DOMContentLoaded', () => {
  // ──────────────────────────────
  //  📌 Elementos del DOM
  // ──────────────────────────────
  const form         = document.getElementById('opts');
  const bitsSelect   = document.getElementById('bits-select');
  const th3Block     = document.getElementById('thresholds-3');
  const th4Block     = document.getElementById('thresholds-4');
  const resultsDiv   = document.getElementById('results');
  const btnErr       = document.getElementById('boton-err');
  const btnHamming   = document.getElementById('boton-hamming');

  // ──────────────────────────────
  //  👀 Mostrar/ocultar inputs de umbral
  // ──────────────────────────────
  function toggleThresholdInputs() {
    const is3 = bitsSelect.value === '3';
    th3Block.classList.toggle('d-none', !is3);
    th4Block.classList.toggle('d-none',  is3);
    th3Block.querySelector('input[name="t1"]').required = is3;
    th4Block.querySelectorAll('input').forEach(inp => inp.required = !is3);
  }
  bitsSelect.addEventListener('change', toggleThresholdInputs);
  toggleThresholdInputs();  // estado inicial

  // ──────────────────────────────
  //  🛠️ Construir payload común
  // ──────────────────────────────
  function construirPayload() {
    const fd = new FormData(form);
    const payload = { model: fd.get('model'), bits: fd.get('bits') };
    if (payload.bits === '3') {
      payload.t1 = fd.get('t1');
    } else {
      payload.t2 = fd.get('t2');
      payload.t3 = fd.get('t3');
    }
    return payload;
  }

  // ──────────────────────────────
  //  🚀 Enviar y renderizar sin borrar lo anterior
  // ──────────────────────────────
  async function enviarPayload(payload, url) {
    // 1) Spinner
    const spinner = document.createElement('div');
    spinner.innerHTML = `
      <div class="text-center my-3">
        <div class="spinner-border" role="status"></div>
        <p class="mt-2">Procesando…</p>
      </div>`;
    resultsDiv.prepend(spinner);

    try {
      // 2) Esperar al servidor
      const resp = await fetch(url, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload)
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();

      // 3) Nuevo contenedor para este resultado
      const wrapper = document.createElement('div');
      wrapper.className = 'my-4';
      resultsDiv.prepend(wrapper);

      // 4) Dibujar: detecta si viene json.plot (ERR) o json.hist (hamming)
      const fig = json.plot ?? json.hist;
      if (fig) {
        Plotly.newPlot(wrapper, fig.data, fig.layout, { responsive: true });
      } else {
        wrapper.innerHTML = `
          <div class="alert alert-warning" role="alert">
            No se ha recibido ningún gráfico.
          </div>`;
      }
    } catch (err) {
      // 5) Mostrar error sin borrar lo anterior
      const alert = document.createElement('div');
      alert.className = 'alert alert-danger';
      alert.role = 'alert';
      alert.innerText = `⚠️ Error: ${err.message}`;
      resultsDiv.prepend(alert);
      console.error(err);
    } finally {
      spinner.remove();
    }
  }

  // ──────────────────────────────
  //  🎯 Botón “Calcular err”
  // ──────────────────────────────
  btnErr.addEventListener('click', async evt => {
    evt.preventDefault();
    const formulario = new FormData(form);
    let url = formulario.get('tabla');
    const payload = construirPayload();
    await enviarPayload(payload, `/api/${url}`);
  });

  // ──────────────────────────────
  //  🎯 Botón “Calcular histograma de hamming”
  // ──────────────────────────────
  btnHamming.addEventListener('click', async evt => {
    evt.preventDefault();
    resultsDiv.innerHTML='';
  });
});
