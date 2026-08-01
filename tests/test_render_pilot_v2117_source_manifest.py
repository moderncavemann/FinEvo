from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.render_pilot_v2117_source_manifest as renderer


def test_unpinned_source_bootstrap_is_draft_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = {
        "contract_id": "finevo-pilot-v2.11.7",
        "status": "frozen",
    }
    candidate = tmp_path / "contract.json"
    candidate.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        renderer.pilot_contract_module,
        "PILOT_CONTRACT_V2_11_7_SCIENCE_DESIGN_SHA256",
        None,
    )

    with pytest.raises(ValueError, match="only an unpinned V2.11.7 draft"):
        renderer._load_contract_for_source_bootstrap(candidate)


def test_source_renderer_emits_canonical_tracked_bytes_and_binds_both_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = object()
    observed: dict[str, object] = {}
    manifest = {
        "schema_version": "finevo-pilot-v2.11.7-source-manifest-v1",
        "failed_release": {"contract_id": "finevo-pilot-v2.11.6"},
        "authority_release": {"contract_id": "finevo-pilot-v2.11.5"},
        "remaining_science_implementation_equivalence": {
            "equivalence_claim": (
                "science_core_equal_with_explicit_successor_adapter"
            ),
            "reviewed_changed_function_sha256": {},
        },
        "current_runtime_sources": {
            "pilot_contract_top_level_ast_inventory_sha256": {},
            "orchestrator_top_level_ast_inventory_sha256": {},
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

    monkeypatch.setattr(renderer, "build_v2117_source_manifest", build)
    contract_path = tmp_path / "contract.json"
    repo_root = tmp_path / "child"
    failed_root = tmp_path / "v2116-no-go"
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
