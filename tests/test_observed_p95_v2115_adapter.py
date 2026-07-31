from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from verified_memory import observed_p95_authority as authority
from verified_memory import pilot_contract as pilot_contract_module
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v2115_gate as gate
from verified_memory.pilot_checkpoint import config_from_dict
from verified_memory.pilot_v2115_parent_import import (
    _atomic_json,
    build_v2115_parent_import,
)
from verified_memory.runner import (
    ShockEvent,
    VerifiedRunConfig,
    has_sealed_observed_p95_authority,
    observed_p95_authority_repo_context,
    serialized_has_sealed_observed_p95_authority,
    validate_preflight_p95_reservations,
)

from tests.test_pilot_v2115_gate import FrozenContract


REPO = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO / "experiments" / "pilot_v2_11_5_source_manifest.json"
EXACT_RECEIPT = Path(
    "experiment_results/pilot-v2.11.5/raw/long-context-preflight/"
    "post_gate_authority.json"
)


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("/usr/bin/git", *arguments),
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture(scope="module")
def v2114_terminal_release_root() -> Path:
    configured = os.environ.get("FINEVO_V2114_SCIENCE_RELEASE_ROOT")
    root = (
        Path(configured).expanduser().absolute()
        if configured
        else REPO.parent / "finevo-pilot-v2-11-4-science"
    )
    if not (root / "experiment_results" / "pilot-v2.11.4" / "raw").is_dir():
        pytest.skip(
            "exact V2.11.4 lineage replay requires its ignored terminal raw tree"
        )
    return root


@pytest.fixture
def frozen_v2115_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v2112_parent_release_root: Path,
    v2114_terminal_release_root: Path,
) -> dict[str, Any]:
    repo_root = tmp_path / "release"
    (repo_root / "experiments").mkdir(parents=True)
    (repo_root / "experiments" / SOURCE_MANIFEST.name).write_bytes(
        SOURCE_MANIFEST.read_bytes()
    )
    contract = FrozenContract()
    (repo_root / "experiments" / "pilot_v2_11_5.yaml").write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (repo_root / ".gitignore").write_text(
        "/experiment_results/\n",
        encoding="utf-8",
    )
    _git(repo_root, "init", "--quiet")
    _git(repo_root, "config", "user.name", "FinEvo Test")
    _git(repo_root, "config", "user.email", "finevo-test@example.invalid")
    _git(repo_root, "add", ".gitignore", "experiments")
    _git(repo_root, "commit", "--quiet", "-m", "fixture release")
    release_commit = _git(repo_root, "rev-parse", "HEAD")
    _git(
        repo_root,
        "tag",
        "-a",
        gate.V2115_SCIENCE_TAG,
        "-m",
        "fixture science tag",
    )
    raw_root = repo_root / "experiment_results" / "pilot-v2.11.5" / "raw"
    parent_receipt = build_v2115_parent_import(
        repo_root=repo_root,
        contract=contract,
        child_git_commit=release_commit,
        source_repo_root=v2112_parent_release_root,
        lineage_repo_root=v2114_terminal_release_root,
        evidence_repo_root=REPO,
    )
    _atomic_json(
        raw_root / "parent-import" / "parent_import_receipt.json",
        parent_receipt,
        repo_root=repo_root,
    )
    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        gate.persist_v2115_resealed_observed_p95_authority(
            repo_root=repo_root,
            raw_root=raw_root,
            contract=contract,
            model_id=model_id,
            expected_git_commit=release_commit,
            parent_import_receipt=parent_receipt,
        )
    bindings = {
        model_id: gate.verified_v2115_observed_p95_authority_binding(
            gate.v2115_observed_p95_receipt_path(raw_root, model_id),
            repo_root=repo_root,
            raw_root=raw_root,
            expected_git_commit=release_commit,
            contract=contract,
        )
        for model_id in ("gpt52_main", "gpt56_diagnostic")
    }
    receipt_path, receipt = gate.persist_v2115_post_gate_authority(
        repo_root=repo_root,
        raw_root=raw_root,
        contract=contract,
        expected_git_commit=release_commit,
        parent_import_receipt=parent_receipt,
        per_model_authority_bindings=bindings,
        ledger_event_chain_head="c" * 64,
    )

    original_contract_loader = gate._contract

    def select_fixture_contract(
        root: Path,
        selected: Any,
        expected_git_commit: str,
    ) -> Any:
        return original_contract_loader(
            root,
            contract if selected is None else selected,
            expected_git_commit,
        )

    monkeypatch.setattr(gate, "_contract", select_fixture_contract)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_5_CANONICAL_SHA256",
        contract.canonical_hash,
    )
    return {
        "repo_root": repo_root,
        "raw_root": raw_root,
        "release_commit": release_commit,
        "contract": contract,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_relative": EXACT_RECEIPT.as_posix(),
    }


def _install_provider_constructor_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    constructions: list[str] = []

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        constructions.append("forbidden")
        raise AssertionError("authority verification reached provider construction")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden_provider)
    monkeypatch.setattr("llm_providers.create_llm_provider", forbidden_provider)
    return constructions


def _scientific_config(
    case: dict[str, Any],
    reservations: dict[str, Any],
) -> VerifiedRunConfig:
    return VerifiedRunConfig(
        run_id="v2115-generic-consumer-regression",
        seed=2010922376,
        num_agents=2,
        episode_length=6,
        context_mode="full",
        enable_episodic_retrieval=True,
        enable_semantic=True,
        semantic_proposal_after=3,
        semantic_proposal_interval=3,
        shock_schedule=tuple(
            ShockEvent(
                decision_t=decision_t,
                phase="baseline",
                interest_rate=0.03,
            )
            for decision_t in range(6)
        ),
        scientific_scope="preregistered_mechanism_micro_pilot",
        pilot_contract_hash=case["contract"].canonical_hash,
        pilot_tag=gate.V2115_SCIENCE_TAG,
        allow_scientific_scope=True,
        preflight_p95_reservations=reservations,
    )


def test_v2115_schema_is_registered_without_replacing_parent_adapter() -> None:
    registry = authority.DEDICATED_OBSERVED_P95_BINDING_SCHEMA_REGISTRY

    assert registry[gate.V2115_GATE_SCHEMA_VERSION] == ("v2.11.5-post-gate-authority")
    assert registry["finevo-pilot-v2.11.4-post-gate-authority-v1"] == (
        "v2.11.4-post-gate-authority"
    )
    assert registry["finevo-pilot-v2.11.2-post-gate-authority-v1"] == (
        "v2.11.2-post-gate-authority"
    )


def test_v2115_gate_reaches_generic_binding(
    frozen_v2115_release: dict[str, Any],
) -> None:
    case = frozen_v2115_release
    dedicated = gate.verified_v2115_gate_authority_binding(
        case["receipt_relative"],
        repo_root=case["repo_root"],
        expected_git_commit=case["release_commit"],
        expected_contract_sha256=case["contract"].canonical_hash,
        contract=case["contract"],
    )
    generic = authority.verified_observed_p95_authority_binding(
        case["receipt_relative"],
        repo_root=case["repo_root"],
        expected_git_commit=case["release_commit"],
    )
    rows = authority.verify_observed_p95_authority_receipt(
        case["receipt_relative"],
        repo_root=case["repo_root"],
        expected_git_commit=case["release_commit"],
    )

    assert generic == dedicated
    assert rows == generic["reservations"]
    assert set(rows) == {
        "openai/gpt-5.2-2025-12-11",
        "openai/gpt-5.6-sol",
    }


def test_v2115_gate_reaches_runner_consumers_before_provider(
    frozen_v2115_release: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = frozen_v2115_release
    constructions = _install_provider_constructor_sentinels(monkeypatch)
    binding = authority.verified_observed_p95_authority_binding(
        case["receipt_relative"],
        repo_root=case["repo_root"],
        expected_git_commit=case["release_commit"],
    )
    runner_reservations = gate.runner_reservations_from_v2115_gate_binding(binding)
    payloads: list[dict[str, Any]] = []
    with observed_p95_authority_repo_context(case["repo_root"]):
        for runtime_model, by_kind in runner_reservations.items():
            config = _scientific_config(case, {runtime_model: by_kind})
            assert set(
                validate_preflight_p95_reservations(
                    config,
                    provider_model_name=runtime_model,
                )
            ) == {"action", "semantic"}
            assert has_sealed_observed_p95_authority(config) is True
            payload = config.to_dict()
            payloads.append(payload)
            restored = config_from_dict(payload)
            assert restored.to_dict() == payload
            assert has_sealed_observed_p95_authority(restored) is True
    assert payloads
    assert all(
        serialized_has_sealed_observed_p95_authority(
            payload,
            authority_repo_root=case["repo_root"],
        )
        for payload in payloads
    )
    assert constructions == []


def test_v2115_mapping_only_and_alias_paths_fail_closed(
    frozen_v2115_release: dict[str, Any],
) -> None:
    case = frozen_v2115_release
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="requires a repository-relative receipt path",
    ):
        authority.verify_observed_p95_authority_receipt(
            case["receipt"],
            repo_root=case["repo_root"],
            expected_git_commit=case["release_commit"],
        )

    alias = Path("experiment_results/pilot-v2.11.5/raw/alias/post_gate_authority.json")
    alias_path = case["repo_root"] / alias
    alias_path.parent.mkdir(parents=True)
    shutil.copyfile(case["receipt_path"], alias_path)
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="outside its exact frozen current-release path",
    ):
        authority.verified_observed_p95_authority_binding(
            alias.as_posix(),
            repo_root=case["repo_root"],
            expected_git_commit=case["release_commit"],
        )


def test_v2115_wrong_contract_and_binding_substitution_fail_closed(
    frozen_v2115_release: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = frozen_v2115_release
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_5_CANONICAL_SHA256",
        "9" * 64,
    )
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="selected contract hash drifted",
    ):
        authority.verified_observed_p95_authority_binding(
            case["receipt_relative"],
            repo_root=case["repo_root"],
            expected_git_commit=case["release_commit"],
        )

    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_5_CANONICAL_SHA256",
        case["contract"].canonical_hash,
    )
    original = gate.verified_v2115_gate_authority_binding

    def substitute_binding(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = deepcopy(original(*args, **kwargs))
        result["receipt_file_sha256"] = "8" * 64
        return result

    monkeypatch.setattr(
        gate,
        "verified_v2115_gate_authority_binding",
        substitute_binding,
    )
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="bytes or flat binding changed",
    ):
        authority.verified_observed_p95_authority_binding(
            case["receipt_relative"],
            repo_root=case["repo_root"],
            expected_git_commit=case["release_commit"],
        )


def test_v2115_receipt_bytes_tamper_fails_closed(
    frozen_v2115_release: dict[str, Any],
) -> None:
    case = frozen_v2115_release
    receipt_path = Path(case["receipt_path"])
    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["go"] = False
    receipt_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="failed validation",
    ):
        authority.verified_observed_p95_authority_binding(
            case["receipt_relative"],
            repo_root=case["repo_root"],
            expected_git_commit=case["release_commit"],
        )


def test_v2115_symlink_receipt_fails_closed(
    frozen_v2115_release: dict[str, Any],
) -> None:
    case = frozen_v2115_release
    receipt_path = Path(case["receipt_path"])
    backup = receipt_path.with_name("post_gate_authority.backup.json")
    receipt_path.replace(backup)
    receipt_path.symlink_to(backup.name)

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="cannot be opened safely",
    ):
        authority.verified_observed_p95_authority_binding(
            case["receipt_relative"],
            repo_root=case["repo_root"],
            expected_git_commit=case["release_commit"],
        )
