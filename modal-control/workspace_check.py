"""Read-only workspace check for the pinned Modal SDK.

A token identifies one workspace; this is not an account-wide workspace listing.
Only Secret names are inspected. No secret values, functions, or deployments are used.
"""

import argparse
import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import modal
from modal.config import _lookup_workspace, config


CONTROLLER_SECRET = "android-modal-control"


def lookup_workspace() -> str:
    """Resolve only the workspace associated with the configured token."""
    token_id = os.environ.get("MODAL_TOKEN_ID", "")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    if not token_id.strip() or not token_secret.strip():
        raise ValueError("Missing Modal credentials")
    # This is the same private lookup used by the pinned SDK's profile listing.
    workspace = asyncio.run(_lookup_workspace(config.get("server_url"), token_id, token_secret)).username
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError("Workspace lookup returned no name")
    return workspace


def check_workspace(request_id: str) -> dict:
    if re.fullmatch(r"[0-9a-fA-F]{32}", request_id) is None:
        raise ValueError("Invalid request ID")
    password = os.environ.get("CONTROLLER_BUNDLE_PASSWORD", "")
    if len(password) < 16:
        raise ValueError("Invalid bundle password")
    workspace = lookup_workspace()
    controller_secret_exists = any(secret.name == CONTROLLER_SECRET for secret in modal.Secret.objects.list())
    return {
        "workspaces": [workspace],
        "controller_secret_exists": controller_secret_exists,
        "password_proof": hmac.new(password.encode(), request_id.encode(), hashlib.sha256).hexdigest(),
        "request_id": request_id,
    }


def private_write(output: Path, record: dict) -> None:
    # Replace atomically, including existing symlinks, without a permissive window.
    fd, temporary = tempfile.mkstemp(prefix=".workspace-", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        private_write(args.output, check_workspace(args.request_id))
    except Exception:
        print("Workspace check failed; no secrets were changed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
