from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.render_pilot_v2116_source_manifest as renderer


def test_unpinned_source_bootstrap_is_draft_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = json.loads(renderer.TRACKED_CONTRACT_PATH.read_text(encoding="utf-8"))
    document["status"] = "frozen"
    candidate = tmp_path / "contract.json"
    candidate.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        renderer.pilot_contract_module,
        "PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256",
        None,
    )

    with pytest.raises(ValueError, match="only an unpinned V2.11.6 draft"):
        renderer._load_contract_for_source_bootstrap(candidate)


def test_source_renderer_emits_canonical_tracked_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = object()
    observed: dict[str, object] = {}
    manifest = {
        "schema_version": "finevo-pilot-v2.11.6-source-manifest-v1",
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

    monkeypatch.setattr(renderer, "build_v2116_source_manifest", build)
    contract_path = tmp_path / "contract.json"
    repo_root = tmp_path / "child"
    parent_root = tmp_path / "parent"

    value, payload = renderer.render_source_manifest(
        contract_path=contract_path,
        repo_root=repo_root,
        parent_repo_root=parent_root,
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
        "parent_repo_root": parent_root,
    }
