"""Deployment entrypoint. No credentials are ever passed to the GPU sandbox."""
import os

import httpx
import modal

from control import PROFILES, make_app

APP_NAME = "android-modal-control"
app = modal.App(APP_NAME)
state = modal.Dict.from_name("android-modal-control-state-v1", create_if_missing=True)
gpu_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.8.5")
    .entrypoint([])
    .run_commands("python3 -m pip install fastapi==0.115.12 uvicorn==0.34.2 httpx==0.28.1", "python3 -c 'import vllm; assert vllm.__version__ == \"0.8.5\"'")
    .add_local_file("worker.py", "/opt/worker.py", copy=True)
    .add_local_python_source("control")
)


class ModalBackend:
    name = "android-owned-model-v1"

    def find(self, name):
        try:
            return modal.Sandbox.from_name(APP_NAME, name).object_id
        except modal.exception.NotFoundError:
            return None

    def create(self, record):
        # Discovery precedes launch; a persisted intent will recover an existing handle.
        if self.find(record["name"]):
            raise RuntimeError("Reserved sandbox already exists; recover instead of launching")
        session = record["session"]
        profile = PROFILES[session["profile_id"]]
        sandbox = modal.Sandbox.create(
            "python3", "/opt/worker.py", app=app, name=self.name, image=gpu_image,
            gpu=profile["gpu"], cpu=(profile["cpu"], profile["cpu"]),
            memory=(profile["memory_mib"], profile["memory_mib"]),
            timeout=session["max_minutes"] * 60, encrypted_ports=[8080],
            env={"MODEL": profile["model"], "IDLE_SECONDS": str(session["idle_minutes"] * 60), "GATEWAY_TOKEN": record["gateway_token"]},
        )
        return sandbox.object_id

    def handle(self, record):
        return modal.Sandbox.from_id(record["sandbox_id"])

    def poll(self, record):
        return self.handle(record).poll()

    def stop(self, record):
        self.handle(record).terminate(wait=True)

    def request(self, record, method, path, payload=None, timeout=10):
        url = self.handle(record).tunnels(timeout=5)[8080].url
        response = httpx.request(method, url + path, headers={"Authorization": "Bearer " + record["gateway_token"]}, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def ready(self, record):
        try:
            return self.request(record, "GET", "/health")["ready"]
        except (httpx.HTTPError, modal.exception.SandboxTimeoutError):
            return False

    def logs(self, record):
        return self.request(record, "GET", "/logs")["lines"]

    def chat(self, record, payload):
        return self.request(record, "POST", "/chat", payload, timeout=180)


# Sharing the registered image prebuilds it and hydrates its handle before any POST.
@app.function(image=gpu_image, cpu=(0.25, 0.25), memory=(512, 512), secrets=[modal.Secret.from_name("android-modal-control")], max_containers=1, timeout=240, scaledown_window=60)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def controller():
    # Custom scoped bearer auth, not a Modal provider/proxy credential in the APK.
    return make_app(state, ModalBackend(), os.environ["CONTROLLER_TOKEN"])
