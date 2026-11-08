"""Safe local/Actions deployment operations; no worker or inference requests."""
from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

import httpx
import modal

APP_NAME = "android-modal-control"
DEFAULT_CONNECTION = Path.home() / ".config/kaggle-modal/connection.json"
DIRECTORY = Path(__file__).resolve().parent
BUNDLE_AAD = b"android-modal-control-connection-v1"


class OperationError(Exception):
    """A deliberately non-sensitive diagnostic."""


def validate_url(value):
    if not isinstance(value, str):
        raise OperationError("Missing controller HTTPS URL.")
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/") or any(c.isspace() for c in value)):
        raise OperationError("Expected an HTTPS origin without credentials, path, or query.")
    return value.rstrip("/")


def validate_token(token):
    if not isinstance(token, str) or len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
        raise OperationError("CONTROLLER_TOKEN must contain at least 32 ASCII characters without whitespace.")
    return token


def private_write(path, contents):
    path = Path(path).expanduser()
    path = path.parent.resolve() / path.name
    root = DIRECTORY.parent
    if path == root or root in path.parents:
        raise OperationError("Connection files must be saved outside the repository.")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".connection-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bundle_password(prompt=False):
    password = os.environ.get("CONTROLLER_BUNDLE_PASSWORD")
    if not password and prompt:
        password = getpass.getpass("Bundle password: ")
    if not password or len(password) < 16:
        raise OperationError("CONTROLLER_BUNDLE_PASSWORD must contain at least 16 characters.")
    return password


def derive_key(password, salt):
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    return Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(password.encode("utf-8"))


def encrypt_connection(connection, password):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt, nonce = secrets.token_bytes(16), secrets.token_bytes(12)
    ciphertext = AESGCM(derive_key(password, salt)).encrypt(
        nonce, json.dumps(connection).encode(), BUNDLE_AAD)
    encode = lambda value: base64.b64encode(value).decode("ascii")
    return json.dumps({"version": 1, "salt": encode(salt), "nonce": encode(nonce), "ciphertext": encode(ciphertext)}).encode()


def decrypt_connection(contents, password):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        envelope = json.loads(contents)
        if envelope["version"] != 1:
            raise ValueError()
        decode = lambda key: base64.b64decode(envelope[key], validate=True)
        salt, nonce, ciphertext = decode("salt"), decode("nonce"), decode("ciphertext")
        if len(salt) != 16 or len(nonce) != 12:
            raise ValueError()
        return json.loads(AESGCM(derive_key(password, salt)).decrypt(nonce, ciphertext, BUNDLE_AAD))
    except Exception:
        raise OperationError("Cannot decrypt bundle; check the password and file integrity.") from None


def load_connection(path):
    connection = json.loads(Path(path).expanduser().read_text())
    validate_token(connection["token"])
    connection["url"] = validate_url(connection["url"])
    return connection


def verify_controller(url, token):
    url, token = validate_url(url), validate_token(token)
    # No session/status/log requests: those may poll an existing worker.
    with httpx.Client(timeout=60, follow_redirects=False, trust_env=False) as client:
        anonymous = client.get(url + "/api/profiles")
        if anonymous.status_code != 401:
            raise OperationError("Controller did not reject an unauthenticated profiles request.")
        response = client.get(url + "/api/profiles", headers={"Authorization": "Bearer " + token})
        if response.status_code != 200:
            raise OperationError("Authenticated controller profiles check failed.")
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("profiles"), list) or not data["profiles"]:
            raise OperationError("Controller returned an invalid profiles response.")


def controller_url():
    return validate_url(modal.Function.from_name(APP_NAME, "controller").get_web_url())


def require_inactive_session():
    try:
        store = modal.Dict.from_name("android-modal-control-state-v1", create_if_missing=False)
        current = store.get("current")
    except modal.exception.NotFoundError:
        return
    if current is not None and current["session"]["status"] not in {"STOPPED", "ERROR"}:
        raise OperationError("Deployment refused: an active or unconfirmed session is persisted. Stop it and confirm its final state first.")


def read_controller_token():
    """Read the existing secret through a short-lived CPU-only Modal function."""
    try:
        app = modal.App()

        @app.function(
            image=modal.Image.debian_slim(python_version="3.12"),
            secrets=[modal.Secret.from_name(APP_NAME)],
            cpu=0.125, memory=128, timeout=60, max_containers=1,
            serialized=True, include_source=False,
        )
        def stored_token():
            import os

            return os.environ.get("CONTROLLER_TOKEN")

        with app.run():
            token = stored_token.remote()
    except Exception:
        raise OperationError("Cannot retrieve the existing CONTROLLER_TOKEN; no secret was changed.") from None
    return validate_token(token)


def deploy(args):
    if modal.__version__ != "1.4.2":
        raise OperationError("Install the repository requirements (Modal 1.4.2).")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        if not all(os.environ.get(key, "").strip() for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")):
            raise OperationError("GitHub Actions requires nonempty MODAL_TOKEN_ID and MODAL_TOKEN_SECRET repository secrets.")
    if os.environ.get("GITHUB_ACTIONS") == "true" and not args.encrypted_output:
        raise OperationError("GitHub Actions requires --encrypted-output; plaintext runner output is disabled.")
    password = bundle_password() if args.encrypted_output else None
    output = args.encrypted_output or args.output
    require_inactive_session()
    supplied = os.environ.get("CONTROLLER_TOKEN") or None
    existing = any(secret.name == APP_NAME for secret in modal.Secret.objects.list())
    if existing:
        token = read_controller_token()
        if supplied is not None and not secrets.compare_digest(validate_token(supplied), token):
            raise OperationError("Supplied CONTROLLER_TOKEN differs from the stored token; automatic rotation is disabled.")
    else:
        token = validate_token(supplied) if supplied else secrets.token_urlsafe(48)
    url = None
    if existing:
        try:
            url = controller_url()
        except modal.exception.NotFoundError:
            pass
        if url is not None:
            verify_controller(url, token)
    # Existing secrets are never updated, even when a token is supplied.
    connection = {"url": url, "token": token}

    def save():
        payload = encrypt_connection(connection, password) if password else json.dumps(connection).encode()
        private_write(output, payload)

    # Persist recovery credentials before creating a secret or deploying.
    save()
    if not existing:
        modal.Secret.objects.create(APP_NAME, {"CONTROLLER_TOKEN": token}, allow_existing=False)
    environment = os.environ.copy()
    for key in ("CONTROLLER_TOKEN", "CONTROLLER_BUNDLE_PASSWORD"):
        environment.pop(key, None)
    result = subprocess.run([sys.executable, "-m", "modal", "deploy", "deploy.py"],
                            cwd=DIRECTORY, env=environment, capture_output=True, check=False)
    if result.returncode:
        raise OperationError("Modal deployment failed; provider output suppressed. Saved credentials can be reused.")
    connection["url"] = controller_url()
    save()
    verify_controller(connection["url"], token)
    print(connection["url"])


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    deploy_cli = commands.add_parser("deploy", help="Deploy and verify profiles only; never starts a GPU session")
    outputs = deploy_cli.add_mutually_exclusive_group()
    outputs.add_argument("--output", type=Path, default=DEFAULT_CONNECTION)
    outputs.add_argument("--encrypted-output", type=Path)
    verify_cli = commands.add_parser("verify", help="Check authentication and profiles only")
    verify_cli.add_argument("--connection", type=Path, default=DEFAULT_CONNECTION)
    decrypt_cli = commands.add_parser("decrypt", help="Decrypt an artifact to a mode-0600 connection file")
    decrypt_cli.add_argument("bundle", type=Path)
    decrypt_cli.add_argument("--output", type=Path, default=DEFAULT_CONNECTION)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "deploy":
            deploy(args)
        elif args.command == "verify":
            connection = load_connection(args.connection)
            verify_controller(connection["url"], connection["token"])
            print(connection["url"])
        else:
            if os.environ.get("GITHUB_ACTIONS") == "true":
                raise OperationError("Decrypt bundles locally, not on a GitHub Actions runner.")
            connection = decrypt_connection(args.bundle.read_bytes(), bundle_password(prompt=True))
            validate_token(connection["token"])
            # A failed deployment can leave a recovery bundle with no URL yet.
            if connection.get("url") is not None:
                connection["url"] = validate_url(connection["url"])
            private_write(args.output, json.dumps(connection).encode())
            print("Connection saved with mode 0600; keep its token private.")
        return 0
    except OperationError as error:
        print(str(error), file=sys.stderr)
    except Exception:
        print("Operation failed; sensitive exception details suppressed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
