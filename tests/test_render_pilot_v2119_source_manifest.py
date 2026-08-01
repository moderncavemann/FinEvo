from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.render_pilot_v2119_source_manifest as renderer


def test_unpinned_source_bootstrap_is_draft_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = {
        "contract_id": "finevo-pilot-v2.11.9",
        "status": "frozen",
    }
    candidate = tmp_path / "contract.json"
    candidate.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        renderer.pilot_contract_module,
        "PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256",
        None,
    )

    with pytest.raises(ValueError, match="only an unpinned V2.11.9 draft"):
        renderer._load_contract_for_source_bootstrap(candidate)


def test_source_renderer_emits_canonical_bytes_and_binds_both_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = object()
    observed: dict[str, object] = {}
    manifest = {
        "schema_version": "finevo-pilot-v2.11.9-source-manifest-v1",
        "failed_release": {"contract_id": "finevo-pilot-v2.11.8"},
        "authority_release": {"contract_id": "finevo-pilot-v2.11.5"},
        "observed_p95_context_recovery": {
            "repair_changes_scientific_design": False,
        },
        "integrity": {"content_sha256": "a" * 64},
    }
    monkeypatch.setattr(
        renderer,
        "_load_contract_for_source_bootstrap",
        lambda path: contract,
    )

    def build(**kwargs):
        observed.update(kwargs)
        return manifest

    monkeypatch.setattr(renderer, "build_v2119_source_manifest", build)
    contract_path = tmp_path / "contract.json"
    repo_root = tmp_path / "child"
    failed_root = tmp_path / "v2118-no-go"
    authority_root = tmp_path / "v2115-authority"

    value, payload = renderer.render_source_manifest(
        contract_path=contract_path,
        repo_root=repo_root,
        failed_repo_root=failed_root,
        authority_repo_root=authority_root,
    )

    assert value == manifest
    assert payload == (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert observed == {
        "contract": contract,
        "repo_root": repo_root,
        "failed_repo_root": failed_root,
        "authority_repo_root": authority_root,
    }


def test_check_mode_is_read_only_and_fails_on_byte_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = {"integrity": {"content_sha256": "b" * 64}}
    payload = b'{"sealed":true}\n'
    output = tmp_path / "pilot_v2_11_9_source_manifest.json"
    output.write_bytes(payload)
    monkeypatch.setattr(
        renderer,
        "render_source_manifest",
        lambda **kwargs: (manifest, payload),
    )
    argv = [
        "--contract",
        str(tmp_path / "contract.json"),
        "--repo-root",
        str(tmp_path / "child"),
        "--failed-repo-root",
        str(tmp_path / "v2118-no-go"),
        "--authority-repo-root",
        str(tmp_path / "v2115-authority"),
        "--output",
        str(output),
        "--check",
    ]

    assert renderer.main(argv) == 0
    assert output.read_bytes() == payload

    output.write_bytes(b"drifted\n")
    with pytest.raises(SystemExit, match="tracked V2.11.9 source manifest drifted"):
        renderer.main(argv)
    assert output.read_bytes() == b"drifted\n"
