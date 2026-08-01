from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.render_pilot_v21110_source_manifest as renderer
from verified_memory.pilot_v21110_continuation import (
    PilotV21110ContinuationError,
)


def test_unpinned_source_bootstrap_is_draft_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = {
        "contract_id": "finevo-pilot-v2.11.10",
        "status": "frozen",
    }
    candidate = tmp_path / "contract.json"
    candidate.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        renderer.pilot_contract_module,
        "PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256",
        None,
    )

    with pytest.raises(ValueError, match="only an unpinned V2.11.10 draft"):
        renderer._load_contract_for_source_bootstrap(candidate)


def test_unpinned_draft_uses_temporary_design_pin_and_restores_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = {
        "contract_id": "finevo-pilot-v2.11.10",
        "status": "draft",
    }
    candidate = tmp_path / "contract.json"
    candidate.write_text(json.dumps(document), encoding="utf-8")
    sentinel = object()
    observed: list[object] = []
    monkeypatch.setattr(
        renderer.pilot_contract_module,
        "PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256",
        None,
    )
    monkeypatch.setattr(renderer, "science_design_sha256", lambda value: "c" * 64)

    def load(path: Path):
        assert path == candidate
        observed.append(
            renderer.pilot_contract_module.PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256
        )
        return sentinel

    monkeypatch.setattr(renderer, "load_pilot_contract", load)

    assert renderer._load_contract_for_source_bootstrap(candidate) is sentinel
    assert observed == ["c" * 64]
    assert (
        renderer.pilot_contract_module.PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256
        is None
    )


def test_source_renderer_emits_canonical_bytes_and_binds_three_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = object()
    observed: dict[str, object] = {}
    manifest = {
        "schema_version": "finevo-pilot-v2.11.10-source-manifest-v1",
        "contract_id": "finevo-pilot-v2.11.10",
        "failed_release": {"contract_id": "finevo-pilot-v2.11.9"},
        "authority_release": {"contract_id": "finevo-pilot-v2.11.5"},
        "observed_p95_authority_adapter_recovery": {
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

    monkeypatch.setattr(renderer, "build_v21110_source_manifest", build)
    contract_path = tmp_path / "contract.json"
    repo_root = tmp_path / "v21110-current"
    failed_root = tmp_path / "v2119-no-go"
    authority_root = tmp_path / "v2115-authority"
    for root in (repo_root, failed_root, authority_root):
        root.mkdir()

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
        "repo_root": repo_root.resolve(),
        "failed_repo_root": failed_root.resolve(),
        "authority_repo_root": authority_root.resolve(),
    }


@pytest.mark.parametrize(
    ("aliased_left", "aliased_right"),
    [
        ("current", "failed"),
        ("current", "authority"),
        ("failed", "authority"),
    ],
)
def test_source_renderer_rejects_every_alias_root_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    aliased_left: str,
    aliased_right: str,
) -> None:
    roots = {
        "current": tmp_path / "v21110-current",
        "failed": tmp_path / "v2119-no-go",
        "authority": tmp_path / "v2115-authority",
    }
    for root in roots.values():
        root.mkdir()
    roots[aliased_right] = roots[aliased_left]
    called = False

    def build(**kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(renderer, "build_v21110_source_manifest", build)
    with pytest.raises(PilotV21110ContinuationError, match="roots must be distinct"):
        renderer.render_source_manifest(
            contract_path=tmp_path / "contract.json",
            repo_root=roots["current"],
            failed_repo_root=roots["failed"],
            authority_repo_root=roots["authority"],
        )
    assert called is False


def test_source_renderer_rejects_symlink_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "v21110-current"
    failed_real = tmp_path / "v2119-no-go-real"
    failed_alias = tmp_path / "v2119-no-go"
    authority_root = tmp_path / "v2115-authority"
    for root in (repo_root, failed_real, authority_root):
        root.mkdir()
    failed_alias.symlink_to(failed_real, target_is_directory=True)
    monkeypatch.setattr(
        renderer,
        "build_v21110_source_manifest",
        lambda **kwargs: pytest.fail("build must not run for a symlink root"),
    )

    with pytest.raises(
        PilotV21110ContinuationError,
        match="must be a real non-symlink directory",
    ):
        renderer.render_source_manifest(
            contract_path=tmp_path / "contract.json",
            repo_root=repo_root,
            failed_repo_root=failed_alias,
            authority_repo_root=authority_root,
        )


def test_check_mode_is_read_only_and_fails_on_byte_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = {"integrity": {"content_sha256": "b" * 64}}
    payload = b'{"sealed":true}\n'
    output = tmp_path / "pilot_v2_11_10_source_manifest.json"
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
        str(tmp_path / "v21110-current"),
        "--failed-repo-root",
        str(tmp_path / "v2119-no-go"),
        "--authority-repo-root",
        str(tmp_path / "v2115-authority"),
        "--output",
        str(output),
        "--check",
    ]

    assert renderer.main(argv) == 0
    assert output.read_bytes() == payload

    output.write_bytes(b"drifted\n")
    with pytest.raises(SystemExit, match="tracked V2.11.10 source manifest drifted"):
        renderer.main(argv)
    assert output.read_bytes() == b"drifted\n"
