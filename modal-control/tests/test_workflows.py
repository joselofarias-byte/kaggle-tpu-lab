"""Static safety contracts; actionlint separately validates GitHub Actions syntax."""
from pathlib import Path
import re

import pytest

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def workflow(name):
    return (WORKFLOWS / f"{name}.yml").read_text()


def section(text, key, indent=2):
    match = re.search(rf"^{' ' * indent}{re.escape(key)}:\n(.*?)(?=^\S|^{' ' * indent}\S|\Z)",
                      text, flags=re.MULTILINE | re.DOTALL)
    assert match, f"Missing section: {key}"
    return match.group(1)


@pytest.mark.parametrize("name", ["modal-deploy", "modal-inference", "mobile-android-es", "python-syntax"])
def test_manual_entrypoints_exist_in_workflows_directory(name):
    assert section(workflow(name), "workflow_dispatch")


@pytest.mark.parametrize("name,consent", [("modal-deploy", "acknowledge_cost"), ("modal-inference", "accept_cost")])
def test_reusable_and_manual_entrypoints_default_to_validation(name, consent):
    for trigger in ("workflow_dispatch", "workflow_call"):
        inputs = section(workflow(name), trigger)
        assert re.search(r"      validate_only:\n(?:        .*\n)*?        type: boolean\n        default: true", inputs)
        assert re.search(rf"      {consent}:\n(?:        .*\n)*?        default: false", inputs)


@pytest.mark.parametrize("name,job,consent", [("modal-deploy", "deploy", "acknowledge_cost"), ("modal-inference", "inference", "accept_cost")])
def test_remote_jobs_require_validation_and_explicit_consent(name, job, consent):
    body = section(workflow(name), job)
    assert "    needs: validate\n" in body
    assert f"    if: ${{{{ inputs.{consent} && !inputs.validate_only }}}}\n" in body


@pytest.mark.parametrize("name", ["modal-deploy", "modal-inference"])
def test_validation_never_imports_modal_or_accesses_credentials(name):
    body = section(workflow(name), "validate")
    assert "python -m py_compile " in body
    assert "${{ github.sha }}" in body
    assert "${{ github.ref }}" in body
    assert "secrets." not in body
    assert "pip install" not in body
    assert not re.search(r"(?:python (?:operations|infer)\.py|modal deploy|\.remote\()", body)
    for path in re.findall(r"test -f (\S+)", body):
        assert (WORKFLOWS.parents[1] / path).is_file()


def test_mobile_bridge_skips_apk_and_forwards_safety_inputs():
    text = workflow("mobile-android-es")
    assert "if: ${{ !(github.event_name == 'workflow_dispatch' && inputs.modal_deploy) }}" in section(text, "build")
    body = section(text, "modal-deploy")
    assert "if: ${{ github.event_name == 'workflow_dispatch' && inputs.modal_deploy }}" in body
    assert "uses: ./.github/workflows/modal-deploy.yml" in body
    for name in ("acknowledge_cost", "validate_only"):
        assert f"      {name}: ${{{{ inputs.{name} }}}}" in body
    assert "    secrets: inherit" in body


def test_syntax_bridge_only_runs_inference_workflow_on_manual_dispatch():
    text = workflow("python-syntax")
    assert "if: ${{ github.event_name != 'workflow_dispatch' }}" in section(text, "syntax")
    body = section(text, "modal-inference")
    assert "if: ${{ github.event_name == 'workflow_dispatch' }}" in body
    assert "uses: ./.github/workflows/modal-inference.yml" in body
    for name in ("controller_url", "profile", "gpu_hourly_usd", "max_minutes", "accept_cost", "validate_only"):
        assert f"      {name}: ${{{{ inputs.{name} }}}}" in body
    assert "    secrets: inherit" in body


def test_inference_bridge_matches_manual_input_contract():
    assert section(workflow("python-syntax"), "workflow_dispatch") == section(workflow("modal-inference"), "workflow_dispatch")


def test_inference_cleanup_and_serialization_remain_enabled():
    text = workflow("modal-inference")
    assert "  group: modal-owned-controller\n  cancel-in-progress: false" in text
    assert "      - name: Always retry owned-session shutdown\n        if: ${{ always() }}" in text
    assert 'python infer.py stop --journal "$RUNNER_TEMP/inference.json"' in text
    assert "--accept-cost --journal" in text
