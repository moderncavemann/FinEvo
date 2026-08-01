from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_contract as pilot_contract_module
from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory import pilot_v21111_release as release
import scripts.render_pilot_v21111_source_manifest as renderer


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_11.yaml"


def _contract_module_source(
    *,
    canonical: str,
    source_file: str,
    source_content: str,
    historical: str = "1" * 64,
) -> str:
    return f"""\
PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256 = "{historical}"
PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256 = {canonical}
PILOT_V2_11_11_SOURCE_MANIFEST_FILE_SHA256 = {source_file}
PILOT_V2_11_11_SOURCE_MANIFEST_CONTENT_SHA256 = {source_content}
UNCHANGED = "full-ast-bound"
"""


def _ci_module_source(
    *,
    source_file: str,
    source_content: str,
    historical: str = "2" * 64,
) -> str:
    return f"""\
SCIENTIFIC_SOURCE_MANIFEST_ANCHORS = (
    {{
        "path": "experiments/pilot_v2_11_10_source_manifest.json",
        "file_sha256": "{historical}",
        "content_sha256": "{'3' * 64}",
    }},
    {{
        "path": "experiments/pilot_v2_11_11_source_manifest.json",
        "file_sha256": {source_file},
        "content_sha256": {source_content},
    }},
)
UNCHANGED = "full-ast-bound"
"""


def test_contract_ast_normalization_is_identical_across_three_legal_freeze_phases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pilot_contract.py"
    bindings = []
    for values in (
        ("None", "None", "None"),
        ("None", repr("a" * 64), repr("b" * 64)),
        (repr("c" * 64), repr("a" * 64), repr("b" * 64)),
    ):
        path.write_text(
            _contract_module_source(
                canonical=values[0],
                source_file=values[1],
                source_content=values[2],
            ),
            encoding="utf-8",
        )
        bindings.append(release.normalized_contract_module_ast_binding(path))
    assert bindings[0] == bindings[1] == bindings[2]
    assert "bootstrap_state" not in bindings[0]
    assert bindings[0]["replaced_cycle_pins"] == sorted(
        release._CYCLIC_CONTRACT_PIN_NAMES
    )


def test_contract_ast_rejects_half_sealed_sources_and_binds_historical_pins(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pilot_contract.py"
    path.write_text(
        _contract_module_source(
            canonical="None",
            source_file=repr("a" * 64),
            source_content="None",
        ),
        encoding="utf-8",
    )
    with pytest.raises(release.PilotV21111ReleaseError, match="atomically"):
        release.normalized_contract_module_ast_binding(path)

    path.write_text(
        _contract_module_source(
            canonical="None", source_file="None", source_content="None"
        ),
        encoding="utf-8",
    )
    first = release.normalized_contract_module_ast_binding(path)
    path.write_text(
        _contract_module_source(
            canonical="None",
            source_file="None",
            source_content="None",
            historical="4" * 64,
        ),
        encoding="utf-8",
    )
    second = release.normalized_contract_module_ast_binding(path)
    assert first["normalized_ast_sha256"] != second["normalized_ast_sha256"]


def test_ci_ast_normalization_is_stable_but_historical_anchor_is_fully_bound(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ci_release_receipt.py"
    path.write_text(
        _ci_module_source(source_file="None", source_content="None"),
        encoding="utf-8",
    )
    draft = release.normalized_ci_release_module_ast_binding(path)
    path.write_text(
        _ci_module_source(source_file=repr("a" * 64), source_content=repr("b" * 64)),
        encoding="utf-8",
    )
    frozen = release.normalized_ci_release_module_ast_binding(path)
    assert draft == frozen
    assert "bootstrap_state" not in draft

    path.write_text(
        _ci_module_source(
            source_file=repr("a" * 64),
            source_content=repr("b" * 64),
            historical="5" * 64,
        ),
        encoding="utf-8",
    )
    drifted = release.normalized_ci_release_module_ast_binding(path)
    assert draft["normalized_ast_sha256"] != drifted["normalized_ast_sha256"]


def test_ci_ast_rejects_duplicate_key_in_v21111_anchor(tmp_path: Path) -> None:
    path = tmp_path / "ci_release_receipt.py"
    source = _ci_module_source(source_file="None", source_content="None")
    source = source.replace(
        '        "file_sha256": None,\n        "content_sha256": None,',
        '        "file_sha256": None,\n'
        '        "file_sha256": "a" * 64,\n'
        '        "content_sha256": None,',
    )
    path.write_text(source, encoding="utf-8")
    with pytest.raises(release.PilotV21111ReleaseError, match="duplicate keys"):
        release.normalized_ci_release_module_ast_binding(path)


def test_release_roots_and_bound_file_parents_reject_symlink_aliases(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_root = real_parent / "repo"
    real_root.mkdir(parents=True)
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(release.PilotV21111ReleaseError, match="symlink parents"):
        release._real_root(alias_parent / "repo", name="aliased root")

    with pytest.raises(release.PilotV21111ReleaseError, match="must be distinct"):
        release._require_distinct_roots(child=real_root, terminal=real_root)

    data_real = real_root / "data-real"
    data_real.mkdir()
    (data_real / "profiles.json").write_text("{}\n", encoding="utf-8")
    (real_root / "data").symlink_to(data_real, target_is_directory=True)
    with pytest.raises(release.PilotV21111ReleaseError, match="symlink component"):
        release._source_file_binding(real_root, "data/profiles.json")


def test_runtime_inventory_covers_complete_trees_entries_and_bound_data() -> None:
    binding = release._current_runtime_source_bindings(ROOT)
    paths = set(binding["release_python_source_paths"])
    expected_verified = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "verified_memory").rglob("*.py")
    }
    expected_foundation = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "ai_economist/foundation").rglob("*.py")
    }
    assert expected_verified <= paths
    assert expected_foundation <= paths
    assert release._RELEASE_ENTRY_PATHS <= paths
    assert [row["path"] for row in binding["bound_data_files"]] == [
        "config.yaml",
        "data/profiles.json",
    ]


def test_runtime_binding_is_byte_stable_across_source_then_canonical_pin_freeze(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    for directory in ("verified_memory", "ai_economist/foundation"):
        shutil.copytree(ROOT / directory, repo / directory)
    for relative in (*release._RELEASE_ENTRY_PATHS, *release._BOUND_DATA_PATHS):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)

    draft = release._current_runtime_source_bindings(repo)
    contract_path = repo / "verified_memory/pilot_contract.py"
    contract_source = contract_path.read_text(encoding="utf-8")
    for name, value in (
        ("PILOT_V2_11_11_SOURCE_MANIFEST_FILE_SHA256", "a" * 64),
        ("PILOT_V2_11_11_SOURCE_MANIFEST_CONTENT_SHA256", "b" * 64),
    ):
        current = getattr(pilot_contract_module, name)
        assert isinstance(current, str)
        old = f'{name}: Optional[str] = (\n    "{current}"\n)'
        new = f'{name}: Optional[str] = (\n    "{value}"\n)'
        assert contract_source.count(old) == 1
        contract_source = contract_source.replace(old, new)
    contract_path.write_text(contract_source, encoding="utf-8")

    ci_path = repo / "verified_memory/ci_release_receipt.py"
    ci_source = ci_path.read_text(encoding="utf-8")
    marker = '"path": "experiments/pilot_v2_11_11_source_manifest.json"'
    start = ci_source.index(marker)
    end = ci_source.index("    },", start)
    row = ci_source[start:end]
    current_file = pilot_contract_module.PILOT_V2_11_11_SOURCE_MANIFEST_FILE_SHA256
    current_content = (
        pilot_contract_module.PILOT_V2_11_11_SOURCE_MANIFEST_CONTENT_SHA256
    )
    assert isinstance(current_file, str)
    assert isinstance(current_content, str)
    assert row.count(f'"{current_file}"') == 1
    assert row.count(f'"{current_content}"') == 1
    row = row.replace(f'"{current_file}"', f'"{"a" * 64}"')
    row = row.replace(f'"{current_content}"', f'"{"b" * 64}"')
    ci_path.write_text(ci_source[:start] + row + ci_source[end:], encoding="utf-8")
    source_sealed = release._current_runtime_source_bindings(repo)

    contract_source = contract_path.read_text(encoding="utf-8")
    current_canonical = pilot_contract_module.PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256
    assert isinstance(current_canonical, str)
    old = (
        "PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256: Optional[str] = (\n"
        f'    "{current_canonical}"\n'
        ")"
    )
    new = (
        "PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256: Optional[str] = (\n"
        f'    "{"c" * 64}"\n'
        ")"
    )
    assert contract_source.count(old) == 1
    contract_path.write_text(contract_source.replace(old, new), encoding="utf-8")
    frozen = release._current_runtime_source_bindings(repo)

    assert draft == source_sealed == frozen


def test_source_seal_matches_contract_loader_content_hash_convention() -> None:
    value = release._seal({"schema_version": "fixture", "value": 1})
    unsigned = deepcopy(value)
    unsigned["integrity"].pop("content_sha256")
    assert value["integrity"]["content_sha256"] == canonical_sha256(unsigned)
    release._verify_seal(value, name="fixture")


def test_validate_replays_exact_bytes_and_rejects_resealed_payload_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    path = repo.joinpath(*release.V21111_SOURCE_MANIFEST_PATH.parts)
    path.parent.mkdir(parents=True)
    expected = release._seal(
        {
            "schema_version": release.V21111_SOURCE_MANIFEST_SCHEMA_VERSION,
            "contract_id": release.V21111_CONTRACT_ID,
            "current_runtime_sources": {},
            "release_lineage": {},
        }
    )
    path.write_text(json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8")
    binding = {
        "path": release.V21111_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": release.V21111_SOURCE_MANIFEST_SCHEMA_VERSION,
        "file_sha256": release._file_sha256(path),
        "content_sha256": expected["integrity"]["content_sha256"],
    }

    class Contract:
        v21111_fresh_cohort_boundary = {"source_manifest": binding}

    monkeypatch.setattr(
        release,
        "build_v21111_source_manifest",
        lambda **_kwargs: deepcopy(expected),
    )
    assert (
        release.validate_v21111_source_manifest(
            contract=Contract(),
            repo_root=repo,
            v21110_repo_root=tmp_path / "unused-terminal",
            v2115_repo_root=tmp_path / "unused-authority",
        )
        == expected
    )

    tampered = release._seal(
        {
            **{key: value for key, value in expected.items() if key != "integrity"},
            "x": 1,
        }
    )
    path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(release.PilotV21111ReleaseError, match="replay drifted"):
        release.validate_v21111_source_manifest(
            contract=Contract(),
            repo_root=repo,
            v21110_repo_root=tmp_path / "unused-terminal",
            v2115_repo_root=tmp_path / "unused-authority",
        )


def test_renderer_rejects_hybrid_contract_before_loading_or_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    tracked = repo / "experiments" / "pilot_v2_11_11.yaml"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("{}\n", encoding="utf-8")
    hybrid = tmp_path / "other.yaml"
    hybrid.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        renderer,
        "_load_contract_for_source_bootstrap",
        lambda _path: pytest.fail("hybrid contract was loaded"),
    )
    with pytest.raises(ValueError, match="tracked contract below repo-root"):
        renderer.render_source_manifest(
            contract_path=hybrid,
            repo_root=repo,
            v21110_repo_root=tmp_path / "terminal",
            v2115_repo_root=tmp_path / "authority",
        )


def test_locked_parent_source_replay_precedes_provenance_ledgers_and_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    order: list[str] = []
    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "require_exact_raw_namespace",
        lambda **_kwargs: order.append("namespace"),
    )
    monkeypatch.setattr(
        orchestrator,
        "require_provider_keys_absent",
        lambda: order.append("keys"),
    )

    def reject_replay(**_kwargs):
        order.append("replay")
        raise release.PilotV21111ReleaseError("fixture source drift")

    monkeypatch.setattr(orchestrator, "replay_v21111_source_manifest", reject_replay)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: pytest.fail("provenance preceded source replay"),
    )
    monkeypatch.setattr(
        orchestrator,
        "PilotRunLedger",
        lambda *_args, **_kwargs: pytest.fail("ledger preceded source replay"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        lambda *_args, **_kwargs: pytest.fail("provider preceded source replay"),
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="source-manifest replay failed before release-attestation",
    ):
        orchestrator._execute_stage_locked(
            contract_path=CONTRACT_PATH,
            stage_id="parent-import",
            resume=False,
            raw_root=tmp_path / "raw",
            repo_root=tmp_path,
            parent_repo_root=tmp_path / "terminal",
            authority_repo_root=tmp_path / "authority",
        )
    assert order == ["namespace", "keys", "replay"]
