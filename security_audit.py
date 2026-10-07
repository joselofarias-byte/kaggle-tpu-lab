#!/usr/bin/env python3
"""
One-touch multi-model security audit for an owned/local repository.

Flow:
  1) Qwen3.8-27B baseline audit (normal checkpoint)
  2) Qwen3.8-27B abliterated adversarial review
  3) Qwen3.8-27B normal checkpoint again as evidence-checking judge
  4) Save SECURITY_AUDIT.md and stop every Kaggle TPU session after each phase

The repo is exposed read-only through launch.py's audit tools. Secrets, build
artifacts and binaries stay excluded by that layer.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import launch as ktl


HERE = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "storage" / "downloads"

BASE_PROFILE = {
    "label": "Qwen3.8-27B normal",
    "hf_model_id": "Qwen/Qwen3.8-27B",
    "served_model_name": "qwen3.8-27b",
    "weights_dataset": ktl.WEIGHTS_DATASET,
    "mtp": 3,
}

ABLITERATED_PROFILE = {
    "label": "Qwen3.8-27B abliterated",
    "hf_model_id": "hotdogs/Qwen3.8-27B-abliterated",
    "served_model_name": "qwen3.8-27b-abliterated",
    # No Kaggle mirror yet: download directly from Hugging Face.
    "weights_dataset": "",
    # Keep the first red-team rollout conservative until this checkpoint's MTP
    # head is validated separately.
    "mtp": 0,
}


def say(msg):
    ktl.say(msg)


def install_shortcut():
    prefix = os.environ.get("PREFIX")
    if not prefix:
        return
    try:
        target = Path(prefix) / "bin" / "security-audit"
        target.parent.mkdir(parents=True, exist_ok=True)
        script = (
            "#!/data/data/com.termux/files/usr/bin/bash\n"
            'exec python "$HOME/kaggle-tpu-lab/security_audit.py" "$@"\n'
        )
        if not target.exists() or target.read_text(errors="ignore") != script:
            target.write_text(script)
            target.chmod(0o755)
    except Exception as e:
        say(f"AVISO: no pude instalar el comando corto security-audit: {e}")


def pick_instructions(explicit=None):
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise SystemExit(f"No existe el archivo de instrucciones: {p}")
        return p
    return ktl._pick_audit_instructions(None)


def active_state():
    if not ktl.STATE_FILE.exists():
        return None, None
    try:
        st = ktl.load_state()
        status, _ = ktl._kernel_status(st["kernel"])
        return st, status
    except Exception:
        return None, None


def stop_active():
    st, status = active_state()
    if st and status in ("QUEUED", "RUNNING"):
        say(f"Apagando {st['kernel']} ({status})...")
        ktl.cmd_stop(argparse.Namespace())


def wait_new_state(old_topic, child, deadline_s=150):
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        if ktl.STATE_FILE.exists():
            try:
                st = ktl.load_state()
                if st.get("topic") and st.get("topic") != old_topic:
                    return st
            except Exception:
                pass
        if child.poll() is not None:
            raise RuntimeError("El launcher termino antes de crear la nueva instancia.")
        time.sleep(2)
    raise RuntimeError("No aparecio el estado de la nueva instancia dentro del plazo.")


def start_profile(profile, queue_timeout, start_timeout, allow_reuse=False):
    st, status = active_state()
    if st and status in ("QUEUED", "RUNNING"):
        same_model = st.get("hf_model_id") == profile["hf_model_id"]
        if allow_reuse and same_model:
            say(f"Reutilizando instancia activa: {profile['label']}.")
            ready, reason = ktl._wait_for_public_endpoint(
                st["kernel"],
                st["topic"],
                start_timeout_s=start_timeout,
                queue_timeout_s=queue_timeout,
            )
            if not ready:
                raise RuntimeError(f"La instancia reutilizada no produjo endpoint ({reason}).")
            return st, ready

        raise RuntimeError(
            "Ya hay una sesion Kaggle activa que no corresponde a esta fase. "
            "Deja terminar el flujo anterior o apagalo antes de ejecutar security-audit."
        )

    old_topic = st.get("topic") if isinstance(st, dict) else None
    cmd = [
        sys.executable,
        str(HERE / "launch.py"),
        "serve",
        "--fast-start",
        "--text-only",
        "--keepalive-min",
        "180",
        "--mtp",
        str(profile["mtp"]),
        "--hf-model-id",
        profile["hf_model_id"],
        "--served-model-name",
        profile["served_model_name"],
        "--weights-dataset",
        profile["weights_dataset"],
    ]

    say(f"Iniciando {profile['label']}...")
    child = subprocess.Popen(cmd, cwd=str(HERE))
    try:
        st = wait_new_state(old_topic, child)
        ready, reason = ktl._wait_for_public_endpoint(
            st["kernel"],
            st["topic"],
            start_timeout_s=start_timeout,
            queue_timeout_s=queue_timeout,
        )
        if not ready:
            raise RuntimeError(f"{profile['label']} no produjo endpoint ({reason}).")

        # Detach the launcher follower only; the Kaggle server keeps running.
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=12)
            except subprocess.TimeoutExpired:
                child.terminate()
        return st, ready
    except Exception:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
        raise


def write_role_prompt(base_path, role, extra):
    base = base_path.read_text(errors="replace")
    fd, temp_name = tempfile.mkstemp(prefix=f"security-{role}-", suffix=".md")
    os.close(fd)
    p = Path(temp_name)
    p.write_text(
        base
        + "\n\n---\n\n"
        + f"ROL ADICIONAL PARA ESTA PASADA: {role}\n\n"
        + extra.strip()
        + "\n"
    )
    return p


def run_repo_audit(repo, instructions, output_path, max_rounds, timeout):
    args = argparse.Namespace(
        repo=str(repo),
        instructions=str(instructions),
        reasoning_effort="xhigh",
        max_rounds=max_rounds,
        timeout=timeout,
        output_path=str(output_path),
        print_report=False,
    )
    ktl.cmd_repo_audit(args)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"No se genero el informe esperado: {output_path}")


def baseline_phase(repo, base_prompt, args):
    out = DOWNLOADS / "SECURITY_BASELINE_QWEN.md"
    role_prompt = write_role_prompt(
        base_prompt,
        "AUDITOR PRINCIPAL",
        """
Realiza una revision de seguridad y robustez de referencia. Prioriza evidencia
reproducible y referencias archivo:linea. Inspecciona arquitectura, limites de
confianza, autenticacion/licencias, validaciones, estado, concurrencia, errores
fail-open/fail-closed, manejo de secretos y superficies de actualizacion.
No intentes maximizar cantidad de hallazgos: maximiza precision.
""",
    )
    try:
        start_profile(
            BASE_PROFILE,
            queue_timeout=args.queue_timeout,
            start_timeout=args.start_timeout,
            allow_reuse=args.reuse_active_baseline,
        )
        run_repo_audit(repo, role_prompt, out, args.max_rounds, args.timeout)
        return out
    finally:
        role_prompt.unlink(missing_ok=True)
        stop_active()


def adversary_phase(repo, base_prompt, args):
    out = DOWNLOADS / "SECURITY_REDTEAM_ABLITERATED.md"
    role_prompt = write_role_prompt(
        base_prompt,
        "REVISOR ADVERSARIAL / RED TEAM",
        """
Trabaja como revisor adversarial del software autorizado que tienes delante.
Busca supuestos que el auditor principal podria pasar por alto: bypasses de
licencia y vinculacion, manipulacion local de estado, trust boundaries,
replay, downgrade, race conditions, desincronizacion cliente-servidor,
validaciones solo del lado cliente, fallos fail-open y rutas alternativas.
No modifiques el repositorio. No inventes exploits: cada hallazgo debe quedar
anclado en codigo real con archivo:linea, precondiciones e impacto. Distingue
claramente entre hallazgo confirmado, hipotesis y defensa en profundidad.
""",
    )
    try:
        start_profile(
            ABLITERATED_PROFILE,
            queue_timeout=args.queue_timeout,
            start_timeout=args.abliterated_start_timeout,
            allow_reuse=False,
        )
        run_repo_audit(repo, role_prompt, out, args.max_rounds, args.timeout)
        return out
    finally:
        role_prompt.unlink(missing_ok=True)
        stop_active()


def judge_phase(repo, base_prompt, baseline_report, adversary_report, args):
    out = DOWNLOADS / "SECURITY_AUDIT.md"
    baseline = baseline_report.read_text(errors="replace")
    adversary = adversary_report.read_text(errors="replace")

    fd, temp_name = tempfile.mkstemp(prefix="security-judge-", suffix=".md")
    os.close(fd)
    judge_prompt = Path(temp_name)
    judge_prompt.write_text(
        base_prompt.read_text(errors="replace")
        + "\n\n---\n\n"
        + """
ROL ADICIONAL PARA ESTA PASADA: JUEZ / VERIFICADOR FINAL

A continuacion recibes dos informes independientes. NO los des por ciertos.
Usa las herramientas de solo lectura del repositorio para comprobar los
hallazgos materiales contra el codigo actual. El informe final debe:

1. Separar CONFIRMADO, PROBABLE, NO CONFIRMADO y FALSO POSITIVO.
2. Citar archivo:linea para cada hallazgo confirmado/probable.
3. Fusionar duplicados entre ambos informes.
4. Indicar que modelo encontro cada hallazgo: BASELINE, ABLITERATED o AMBOS.
5. Priorizar Critical/High/Medium/Low con impacto y correccion concreta.
6. Incluir una seccion "Pruebas recomendadas" sin modificar el repo.
7. Cerrar con un orden de remediacion de maximo 10 acciones.

===== INFORME BASELINE =====
"""
        + baseline
        + "\n\n===== INFORME ABLITERATED =====\n"
        + adversary
        + "\n"
    )

    try:
        start_profile(
            BASE_PROFILE,
            queue_timeout=args.queue_timeout,
            start_timeout=args.start_timeout,
            allow_reuse=False,
        )
        run_repo_audit(repo, judge_prompt, out, args.judge_rounds, args.timeout)
        return out
    finally:
        judge_prompt.unlink(missing_ok=True)
        stop_active()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        default=str(Path.home() / "9router-license-test"),
        help="repo local autorizado a auditar",
    )
    ap.add_argument(
        "--instructions",
        help="prompt base; por defecto autodetecta el prompt Qwen mas reciente en Descargas",
    )
    ap.add_argument("--max-rounds", type=int, default=30)
    ap.add_argument("--judge-rounds", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--queue-timeout",
        type=int,
        default=7200,
        help="maximo en cola Kaggle por fase; no consume startup timeout",
    )
    ap.add_argument(
        "--start-timeout",
        type=int,
        default=2100,
        help="arranque del Qwen normal, contado desde RUNNING",
    )
    ap.add_argument(
        "--abliterated-start-timeout",
        type=int,
        default=3600,
        help="arranque del checkpoint abliterated; incluye descarga HF en fase RUNNING",
    )
    ap.add_argument(
        "--reuse-active-baseline",
        action="store_true",
        help="reutilizar una sesion BASE normal ya activa si su identidad coincide",
    )
    ap.add_argument(
        "--skip-judge",
        action="store_true",
        help="hacer solo baseline + abliterated y generar un paquete sin tercera pasada",
    )
    args = ap.parse_args()

    install_shortcut()
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        raise SystemExit(f"No existe el repo: {repo}")
    base_prompt = pick_instructions(args.instructions)

    st, status = active_state()
    if st and status in ("QUEUED", "RUNNING") and not args.reuse_active_baseline:
        raise SystemExit(
            "Hay una sesion Kaggle activa. No la voy a interrumpir. "
            "Dejala terminar y luego ejecuta security-audit."
        )

    say("==============================================")
    say(" SECURITY AUDIT MULTIMODELO")
    say("==============================================")
    say(f"Repo: {repo}")
    say(f"Prompt base: {base_prompt.name}")
    say("Fase 1/3: Qwen normal (baseline)")
    baseline = baseline_phase(repo, base_prompt, args)

    say("Fase 2/3: Qwen3.8 abliterated (red team)")
    adversary = adversary_phase(repo, base_prompt, args)

    if args.skip_judge:
        bundle = DOWNLOADS / "SECURITY_AUDIT_UNJUDGED.md"
        bundle.write_text(
            "# Security audit - sin pasada de juez\n\n"
            "## Baseline\n\n"
            + baseline.read_text(errors="replace")
            + "\n\n## Abliterated red team\n\n"
            + adversary.read_text(errors="replace")
        )
        say(f"Paquete guardado en {bundle}")
        return

    say("Fase 3/3: Qwen normal (juez y verificador)")
    final = judge_phase(repo, base_prompt, baseline, adversary, args)

    say("==============================================")
    say(" SECURITY AUDIT TERMINADA")
    say("==============================================")
    say(f"Baseline: {baseline}")
    say(f"Red team: {adversary}")
    say(f"Final verificado: {final}")
    say("TPU apagada al finalizar cada fase.")


if __name__ == "__main__":
    main()
