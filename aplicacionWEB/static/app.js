/**
 * app.js – controla la UI y envía la petición al backend Flask
 * Autor: tú 😉
 */
document.addEventListener('DOMContentLoaded', () => {
  const form        = document.getElementById('opts');
  const bitsSelect  = document.getElementById('bits-select');
  const thr3Div     = document.getElementById('thresholds-3');
  const thr4Div     = document.getElementById('thresholds-4');

  /* Mostrar / ocultar campos según opción de bits */
  function toggleThresholds () {
    if (bitsSelect.value === '3') {
      thr3Div.classList.remove('d-none');
      thr4Div.classList.add   ('d-none');
    } else {
      thr3Div.classList.add   ('d-none');
      thr4Div.classList.remove('d-none');
    }
  }
  bitsSelect.addEventListener('change', toggleThresholds);
  toggleThresholds();                     // estado inicial

  /* Envío del formulario */
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Serializar datos
    const fd = new FormData(form);
    if (bitsSelect.value === '3') {
      fd.delete('t2'); fd.delete('t3');
    } else {
      fd.delete('t1');
    }
    const payload = Object.fromEntries(fd);

    // Petición al backend
    const resp = await fetch('/api/plot', {
      method : 'POST',
      headers: {'Content-Type': 'application/json'},
      body   : JSON.stringify(payload)
    });
    const fig = await resp.json();

    // Dibujar / actualizar gráfica
    Plotly.react('plot', fig.data, fig.layout, {responsive: true});
  });
});
