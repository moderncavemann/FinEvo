from __future__ import annotations

from pathlib import Path

import pytest

from verified_memory import pilot_v2114_acceptance as acceptance
from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_4.yaml"


def _raw_tree(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "release"
    raw_root = repo_root / "experiment_results" / "pilot-v2.11.4" / "raw"
    raw_root.mkdir(parents=True)
    return repo_root, raw_root


def test_v2114_exact_roots_accepts_only_its_raw_namespace(tmp_path: Path) -> None:
    repo_root, raw_root = _raw_tree(tmp_path)

    assert acceptance._exact_roots(repo_root, raw_root) == (
        repo_root.resolve(),
        raw_root.absolute(),
    )

    sibling = repo_root / "experiment_results" / "pilot-v2.11.3" / "raw"
    sibling.mkdir(parents=True)
    with pytest.raises(
        acceptance.PilotV2114AcceptanceError,
        match="exact ignored raw namespace",
    ):
        acceptance._exact_roots(repo_root, sibling)


def test_v2114_exact_roots_rejects_symlinked_namespace_component(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "release"
    outside = tmp_path / "outside"
    repo_root.mkdir()
    (outside / "pilot-v2.11.4" / "raw").mkdir(parents=True)
    (repo_root / "experiment_results").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        acceptance.PilotV2114AcceptanceError,
        match="raw namespace contains a symlink",
    ):
        acceptance._exact_roots(
            repo_root,
            repo_root / "experiment_results" / "pilot-v2.11.4" / "raw",
        )


def test_v2114_pre_science_namespace_accepts_exact_operational_allowlist(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    allowed = sorted(acceptance._expected_pre_science_file_allowlist(contract))
    for relative in allowed:
        path = raw_root / relative
        if relative == ".real-stage-execution.lock":
            path.touch()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

    assert acceptance._audit_pre_science_namespace(raw_root, contract) == {
        "scientific_stage_paths_present": 0,
        "legacy_scientific_run_paths_present": 0,
        "decoded_scientific_completion_reuse": False,
    }


@pytest.mark.parametrize(
    "relative",
    (
        "experiment-a/copied-result.json",
        (
            "unexpected/finevo-pilot-v2.11.2--experiment-c--gpt52_main--"
            "full--registered-rate-shock--stage0-selected--s1099057501/decoded.json"
        ),
        (
            "unexpected/finevo-pilot-v2.11.3--cross-model--gpt56_diagnostic--"
            "full--registered-rate-shock--stage0-selected--s1099057501/decoded.json"
        ),
        "parent-import/copied_decoded.json",
    ),
)
def test_v2114_pre_science_namespace_rejects_science_legacy_and_extra_files(
    tmp_path: Path,
    relative: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "raw"
    path = raw_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        acceptance.PilotV2114AcceptanceError,
        match="pre-science raw namespace contains scientific artifacts",
    ):
        acceptance._audit_pre_science_namespace(raw_root, contract)


def test_v2114_acceptance_entry_rejects_wrong_contract_namespace_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, raw_root = _raw_tree(tmp_path)
    monkeypatch.setattr(
        acceptance,
        "load_pilot_contract",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("wrong contract path must stop before contract load")
        ),
    )

    with pytest.raises(
        acceptance.PilotV2114AcceptanceError,
        match="requires experiments/pilot_v2_11_4.yaml",
    ):
        acceptance.accept_v2114_scientific_dispatch(
            contract_path=repo_root / "experiments" / "pilot_v2_11_3.yaml",
            repo_root=repo_root,
            raw_root=raw_root,
            scientific_launch_input_path=raw_root / "scientific_launch_input.json",
        )
