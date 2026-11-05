import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pathlib
    import subprocess
    import urllib.request
    import json
    import sys
    return mo, pathlib, subprocess, urllib, json, sys


@app.cell
def _(mo):
    mo.md("""
    # DavidAU 27B · Q4_K_M MTP · CUDA

    Laboratorio interactivo. Seleccioná GPU en Compute antes de preparar.
    El preparador verifica el GGUF exacto y compila llama.cpp con CUDA.
    Las pruebas completas detienen el servidor al finalizar.

    **Al terminar: guardá los resultados y usá Your notebooks → menú →
    Shutdown notebook. Detener una celda o el servidor no apaga el sandbox.**
    No se inician túneles ni endpoints externos.
    """)
    return


@app.cell
def _(mo):
    prepare = mo.ui.run_button(label="Preparar y ejecutar pruebas completas")
    refresh = mo.ui.run_button(label="Actualizar estado")
    start = mo.ui.run_button(label="Cargar modelo para conversar")
    stop = mo.ui.run_button(label="Detener servidor y liberar VRAM")
    mo.vstack([mo.hstack([prepare, refresh]), mo.hstack([start, stop])])
    return prepare, refresh, start, stop


@app.cell
def _(mo, pathlib, subprocess, urllib, sys, prepare):
    mo.stop(not prepare.value)
    _gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
    if _gpu.returncode:
        raise RuntimeError('GPU no disponible. Activala en Compute antes de descargar.')
    _base = pathlib.Path('/marimo/davidau-gguf-lab')
    _code = _base / 'restore-code'
    _code.mkdir(parents=True, exist_ok=True)
    _pidfile = _base / 'interactive-session.pid'
    if _pidfile.exists():
        _cmdline = pathlib.Path('/proc') / _pidfile.read_text().strip() / 'cmdline'
        if _cmdline.exists() and b'davidau_mtp_session.py' in _cmdline.read_bytes():
            raise RuntimeError('Ya hay una preparación activa. Usá Actualizar estado.')
    _pin = 'f682a4fe4c4320494fedab18827a172520383816'
    for _name in ['davidau_mtp_prepare.py', 'davidau_mtp_validate.py',
                  'davidau_mtp_service.py', 'davidau_opencode_smoke.py', 'davidau_mtp_session.py']:
        _url = 'https://raw.githubusercontent.com/joselofarias-byte/kaggle-tpu-lab/' + _pin + '/molab/' + _name
        with urllib.request.urlopen(_url, timeout=30) as _response:
            (_code / _name).write_bytes(_response.read())
    with (_base / 'interactive-session.log').open('w') as _log:
        _process = subprocess.Popen([sys.executable, str(_code/'davidau_mtp_session.py'),
                                     '--base', str(_base)], stdout=_log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    _pidfile.write_text(str(_process.pid))
    mo.md('Preparación iniciada. Usá **Actualizar estado**; no vuelvas a iniciarla mientras corre.')
    return


@app.cell
def _(mo, pathlib, subprocess, refresh):
    mo.stop(not refresh.value)
    _base = pathlib.Path('/marimo/davidau-gguf-lab')
    _gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
    _parts = [_gpu.stdout or _gpu.stderr]
    for _name in ['interactive-session.log', 'session-prepare.log', 'build-resume.log', 'session-result.json']:
        if (_base/_name).exists():
            _parts.append(_name + '\n' + (_base/_name).read_text()[-4000:])
    mo.plain_text('\n\n'.join(_parts))
    return


@app.cell
def _(mo, subprocess, sys, start):
    mo.stop(not start.value)
    _path = '/marimo/davidau-gguf-lab/restore-code/davidau_mtp_service.py'
    _result = subprocess.run([sys.executable, _path, 'start'], cwd='/marimo',
                             capture_output=True, text=True, timeout=210)
    if _result.returncode:
        raise RuntimeError(_result.stderr[-2000:])
    mo.plain_text(_result.stdout)
    return


@app.cell
def _(mo, subprocess, sys, stop):
    mo.stop(not stop.value)
    _path = '/marimo/davidau-gguf-lab/restore-code/davidau_mtp_service.py'
    _result = subprocess.run([sys.executable, _path, 'stop'], cwd='/marimo',
                             capture_output=True, text=True, timeout=30)
    if _result.returncode:
        raise RuntimeError(_result.stderr[-2000:])
    mo.md('Servidor detenido. **Apagá también el sandbox con Shutdown notebook.**')
    return


@app.cell
def _(mo):
    question = mo.ui.text_area(label="Mensaje en español", value="Explicá brevemente qué es CUDA.").form(submit_button_label="Generar respuesta")
    question
    return (question,)


@app.cell
def _(mo, pathlib, urllib, json, question):
    mo.stop(question.value is None)
    _base = pathlib.Path('/marimo/davidau-gguf-lab')
    _key = (_base/'.private/api-key').read_text().strip()
    _body = {'model': 'jolufasoft-davidau-mtp', 'messages': [{'role':'user','content':question.value}],
             'max_tokens':512, 'temperature':0.2, 'chat_template_kwargs':{'enable_thinking':False}}
    _request = urllib.request.Request('http://127.0.0.1:18993/v1/chat/completions',
        data=json.dumps(_body).encode(), headers={'Content-Type':'application/json','Authorization':'Bearer '+_key})
    with urllib.request.urlopen(_request, timeout=180) as _response:
        _result = json.load(_response)
    mo.vstack([mo.plain_text(_result['choices'][0]['message']['content']),
               mo.plain_text(json.dumps({'usage':_result.get('usage'),'timings':_result.get('timings')}, indent=2))])
    return


if __name__ == "__main__":
    app.run()
