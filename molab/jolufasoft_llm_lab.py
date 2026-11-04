# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.17", "torch>=2.5", "transformers>=4.51,<5", "accelerate>=1.0", "huggingface-hub>=0.30", "safetensors"]
# ///
import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import gc
    import time
    return mo, torch, gc, time


@app.cell
def _(mo, torch):
    gpu = torch.cuda.is_available()
    if gpu:
        props = torch.cuda.get_device_properties(0)
        status = f"GPU: {props.name} | VRAM total: {props.total_memory / 2**30:.1f} GiB"
    else:
        status = "GPU no disponible. En Molab: Notebook specs > GPU."
    mo.md(f"# JolufaSoft LLM Lab — Molab\n\n**{status}**\n\nInterfaz para probar modelos Hugging Face en la GPU de Molab. El notebook no publica endpoints ni abre túneles.")
    return


@app.cell
def _(mo):
    model_id = mo.ui.text(value="Qwen/Qwen3-8B", label="Repositorio de modelo (Hugging Face)", full_width=True)
    precision = mo.ui.dropdown(options={"BF16 (recomendado)": "bf16", "FP16": "fp16", "Automático": "auto"}, value="BF16 (recomendado)", label="Precisión")
    remote = mo.ui.checkbox(value=False, label="Permitir código remoto del repositorio (solo fuentes confiables)")
    load = mo.ui.run_button(label="Cargar modelo en GPU")
    mo.vstack([model_id, precision, remote, load])
    return model_id, precision, remote, load


@app.cell
def _(mo, torch, gc, time, model_id, precision, remote, load):
    mo.stop(not load.value, mo.md("Seleccioná un modelo y pulsá **Cargar modelo en GPU**."))
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("GPU no disponible: activala en Molab antes de cargar el modelo.")
    chosen = model_id.value.strip()
    if not chosen or "/" not in chosen:
        raise ValueError("Indicá un repositorio Hugging Face válido: propietario/modelo.")
    gc.collect()
    torch.cuda.empty_cache()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "auto": "auto"}[precision.value]
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(chosen, trust_remote_code=remote.value)
    model = AutoModelForCausalLM.from_pretrained(
        chosen,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=remote.value,
    )
    model.eval()
    load_seconds = time.monotonic() - start
    used_gib = torch.cuda.memory_allocated() / 2**30
    mo.md(f"**Modelo cargado:** `{chosen}` — {load_seconds:.1f} s — {used_gib:.2f} GiB de VRAM asignada")
    return tokenizer, model, chosen


@app.cell
def _(mo):
    prompt = mo.ui.text_area(value="Explicá en español qué ventajas tiene ejecutar un LLM localmente.", label="Mensaje", rows=5, full_width=True)
    system = mo.ui.text_area(value="Respondé en español, con precisión técnica y sin inventar resultados.", label="Instrucción del sistema", rows=2, full_width=True)
    max_new = mo.ui.slider(start=32, stop=2048, step=32, value=256, label="Máximo de tokens nuevos")
    temp = mo.ui.slider(start=0.0, stop=1.5, step=0.05, value=0.7, label="Temperatura")
    run = mo.ui.run_button(label="Generar respuesta")
    mo.vstack([system, prompt, mo.hstack([max_new, temp]), run])
    return prompt, system, max_new, temp, run


@app.cell
def _(mo, torch, time, model, tokenizer, chosen, prompt, system, max_new, temp, run):
    mo.stop(not run.value, mo.md("Pulsá **Generar respuesta** para ejecutar inferencia."))
    messages = [{"role": "system", "content": system.value}, {"role": "user", "content": prompt.value}]
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (ValueError, TypeError, AttributeError):
        rendered = system.value + "\n\nUsuario: " + prompt.value + "\nAsistente:"
    inputs = tokenizer(rendered, return_tensors="pt")
    target = next(model.parameters()).device
    inputs = {k: v.to(target) for k, v in inputs.items()}
    generation_args = {
        "max_new_tokens": max_new.value,
        "do_sample": temp.value > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temp.value > 0:
        generation_args["temperature"] = temp.value
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_args)
    seconds = time.monotonic() - start
    generated = output[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    rate = len(generated) / max(seconds, 0.001)
    mo.vstack([
        mo.md(f"### Respuesta — {chosen}"),
        mo.md(response.replace("<", "&lt;")),
        mo.md(f"**{len(generated)} tokens | {seconds:.1f} s | {rate:.1f} tok/s**"),
    ])
    return


if __name__ == "__main__":
    app.run()
