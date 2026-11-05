# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.17", "torch>=2.8", "transformers==4.57.6", "accelerate>=1.0", "huggingface-hub>=0.34,<1", "safetensors", "kernels>=0.9,<0.12"]
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

    class LLMRuntime:
        """Own exactly one model; repeated load clicks reuse it."""
        def __init__(self):
            self.model = None
            self.tokenizer = None
            self.key = None
            self.load_seconds = None

        def load(self, model_id, dtype, trust_remote_code=False):
            if "gguf" in model_id.lower():
                raise ValueError("GGUF requiere llama.cpp; usá molab/lab.py, no AutoModelForCausalLM.")
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError("Esta GPU no admite BF16; seleccioná FP16 explícitamente.")
            key = (model_id, str(dtype), trust_remote_code)
            if self.model is not None and self.key == key:
                return
            self.model = None
            self.tokenizer = None
            self.key = None
            gc.collect()
            torch.cuda.empty_cache()
            from transformers import AutoTokenizer, AutoModelForCausalLM
            if model_id.startswith("openai/gpt-oss"):
                from transformers.utils import is_kernels_available, is_triton_available
                if not is_kernels_available() or not is_triton_available():
                    raise RuntimeError("MXFP4 requiere kernels y Triton. Instalá dependencias y reiniciá el kernel; fallback BF16 bloqueado.")
            t0 = time.monotonic()
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, device_map="cuda",
                low_cpu_mem_usage=True, trust_remote_code=trust_remote_code,
            )
            self.model.eval()
            torch.cuda.synchronize()
            self.load_seconds = time.monotonic() - t0
            self.key = key

        def generate(self, messages, max_new_tokens, temperature):
            if self.model is None:
                raise RuntimeError("Cargá un modelo primero.")
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = self.tokenizer(rendered, return_tensors="pt").to("cuda")
            args = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id}
            if temperature > 0:
                args["temperature"] = temperature
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.monotonic()
            with torch.inference_mode():
                output = self.model.generate(**inputs, **args)
            torch.cuda.synchronize()
            seconds = time.monotonic() - t0
            generated = output[0][inputs["input_ids"].shape[-1]:]
            return {
                "text": self.tokenizer.decode(generated, skip_special_tokens=True),
                "tokens": len(generated), "seconds": seconds,
                "tokens_per_second": len(generated) / max(seconds, 0.001),
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            }

    session = LLMRuntime()
    return mo, torch, session


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
    model_id = mo.ui.text(value="Qwen/Qwen3-32B", label="Repositorio de modelo (Hugging Face)", full_width=True)
    precision = mo.ui.dropdown(options={"BF16 (recomendado)": "bf16", "FP16": "fp16", "Automático": "auto"}, value="BF16 (recomendado)", label="Precisión")
    remote = mo.ui.checkbox(value=False, label="Permitir código remoto del repositorio (solo fuentes confiables)")
    load = mo.ui.run_button(label="Cargar modelo en GPU")
    mo.vstack([model_id, precision, remote, load])
    return model_id, precision, remote, load


@app.cell
def _(mo, torch, session, model_id, precision, remote, load):
    mo.stop(not load.value, mo.md("Seleccioná un modelo y pulsá **Cargar modelo en GPU**."))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA no disponible.")
    chosen = model_id.value.strip()
    if not chosen or "/" not in chosen:
        raise ValueError("Indicá propietario/modelo.")
    _dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "auto": "auto"}[precision.value]
    session.load(chosen, _dtype, remote.value)
    mo.md(f"**Modelo cargado:** {chosen} | {session.load_seconds:.2f} s | VRAM {torch.cuda.memory_allocated()/2**30:.3f} GiB | cuantizado={getattr(session.model, 'is_quantized', False)}")
    return (chosen,)


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
def _(mo, session, chosen, prompt, system, max_new, temp, run):
    mo.stop(not run.value, mo.md("Pulsá **Generar respuesta**."))
    mo.stop(session.key is None or session.key[0] != chosen, mo.md("Cargá primero el modelo seleccionado."))
    _result = session.generate(
        [{"role": "system", "content": system.value}, {"role": "user", "content": prompt.value}],
        max_new.value, temp.value,
    )
    mo.vstack([
        mo.md(f"### Respuesta — {chosen}"),
        mo.md(_result["text"].replace("<", "&lt;")),
        mo.md(f"**{_result['tokens']} tokens | {_result['seconds']:.3f} s | {_result['tokens_per_second']:.2f} tok/s | pico VRAM {_result['peak_vram_gib']:.3f} GiB**"),
    ])
    return


if __name__ == "__main__":
    app.run()
