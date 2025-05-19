/* static/app.js */
document.addEventListener('DOMContentLoaded', () => {
  // ──────────────────────────────────────────────────────────────
  // 📌 Elementos del DOM
  // ──────────────────────────────────────────────────────────────
  const form       = document.getElementById('opts');
  const bitsSelect = document.getElementById('bits-select');
  const th3Block   = document.getElementById('thresholds-3');
  const th4Block   = document.getElementById('thresholds-4');
  const plotDiv    = document.getElementById('plot');

  // ──────────────────────────────────────────────────────────────
  // 👀 Mostrar/ocultar inputs de umbral
  // ──────────────────────────────────────────────────────────────
  function toggleThresholdInputs () {
    if (bitsSelect.value === '3') {
      th3Block.classList.remove('d-none');
      th4Block.classList.add   ('d-none');
      th3Block.querySelector('input[name="t1"]').required = true;
      th4Block.querySelectorAll('input').forEach(inp => inp.required = false);
    } else {
      th3Block.classList.add   ('d-none');
      th4Block.classList.remove('d-none');
      th3Block.querySelector('input[name="t1"]').required = false;
      th4Block.querySelectorAll('input').forEach(inp => inp.required = true);
    }
  }
  bitsSelect.addEventListener('change', toggleThresholdInputs);
  toggleThresholdInputs();          // estado inicial

  // ──────────────────────────────────────────────────────────────
  // 🚀 Envío del formulario
  // ──────────────────────────────────────────────────────────────
  form.addEventListener('submit', async evt => {
    evt.preventDefault();

    // ▸ Construir payload desde los campos del form
    const fd      = new FormData(form);
    const payload = {
      model: fd.get('model'),      // "512" | "128" (string → back-end lo convierte)
      bits : fd.get('bits')        // "3" | "4"
    };
    if (payload.bits === '3') {
      payload.t1 = fd.get('t1');
    } else {
      payload.t2 = fd.get('t2');   // umbral bajo
      payload.t3 = fd.get('t3');   // umbral alto
    }

    // ▸ Spinner de carga provisional
    Plotly.purge(plotDiv);
    plotDiv.innerHTML =
      `<div class="text-center my-5">
         <div class="spinner-border" role="status"></div>
         <p class="mt-3">Calculando…</p>
       </div>`;

    // ▸ Llamada al API
    try {
      const resp = await fetch('/api/plot', {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify(payload)
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();

      // ▸ Dibujar con Plotly.react
      Plotly.react(plotDiv, json.plot.data, json.plot.layout, { responsive: true });
    } catch (err) {
      plotDiv.innerHTML =
        `<div class="alert alert-danger" role="alert">⚠️ Error: ${err.message}</div>`;
      console.error(err);
    }
  });
});
