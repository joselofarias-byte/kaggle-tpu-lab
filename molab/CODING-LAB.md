# JolufaSoft Coding Lab: GPU + OpenCode

## Estado
- IDE Oracle: instalador disponible, no verificado en servidor.
- Molab: servidor vLLM propuesto; no verificado en sesión GPU.
- Streaming + tools: script de smoke test; no afirmar compatibilidad sin PASS.
- La sesión GPU gratuita puede caducar y el acceso remoto puede estar restringido.

## En Molab
Seleccionar GPU en Compute, abrir terminal y ejecutar:
```bash
export MODEL_ID=Qwen/Qwen3-8B
export MOLAB_API_KEY="$(openssl rand -hex 32)"
bash launch_vllm_coding.sh
```
El servidor escucha únicamente en 127.0.0.1:18080.
**No abrir un túnel público sin autenticación HTTPS.** Molab puede no permitir túneles.
El Qwen3-8B se utiliza para validar tool calling; 27B GGUF requiere otra ruta de carga, no asumir que vLLM lo acepta.

## Oracle
Una vez que exista una URL HTTPS privada estable, con /v1 y autenticación:
```bash
export MOLAB_URL=https://endpoint-privado.example
export MOLAB_API_KEY=... # misma clave, introducir por canal seguro
export MODEL_ID=Qwen/Qwen3-8B
python3 smoke_test_coding_api.py
bash configure_opencode_molab.sh
```
No ejecutar el configurador hasta que el smoke test indique READY.
No almacenar claves en Git ni imprimirlas en logs.
La configuración previa en OpenCode debe revisarse para activar herramientas y streaming.

## Advertencia
El script anterior molab_inference_api.py es una API experimental **sin** tool calling ni streaming; no utilizar para programación autónoma. El servidor vLLM es la ruta propuesta.
