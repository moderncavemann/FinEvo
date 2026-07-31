from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

import tests.test_pilot_v211_gate as v211_fixtures
import tests.test_pilot_v2111_gate as v2111_fixtures
import tests.test_pilot_v2112_gate as gate_fixtures
from verified_memory import observed_p95_authority as authority
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v2112_gate as gate
from verified_memory.pilot_checkpoint import config_from_dict
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256,
)
from verified_memory.runner import (
    ShockEvent,
    VerifiedRunConfig,
    has_sealed_observed_p95_authority,
    observed_p95_authority_repo_context,
    serialized_has_sealed_observed_p95_authority,
    validate_preflight_p95_reservations,
)


EXACT_RECEIPT = Path(
    "experiment_results/pilot-v2.11.2/raw/long-context-preflight/"
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


def _write_receipt(
    repo_root: Path,
    relative: Path,
    receipt: dict[str, Any],
) -> Path:
    target = repo_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _install_gate_fixture_identity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    release_commit: str,
    contract_sha256: str,
) -> None:
    monkeypatch.setattr(
        gate_fixtures,
        "CONTRACT_SHA256",
        contract_sha256,
    )
    monkeypatch.setattr(gate_fixtures, "RELEASE_COMMIT", release_commit)
    monkeypatch.setattr(
        gate_fixtures,
        "MANIFEST_FILE_SHA256",
        gate.V2112_SOURCE_MANIFEST_FILE_SHA256,
    )
    monkeypatch.setattr(
        gate_fixtures,
        "MANIFEST_CONTENT_SHA256",
        gate.V2112_SOURCE_MANIFEST_CONTENT_SHA256,
    )
    monkeypatch.setattr(v211_fixtures, "CONTRACT_SHA256", contract_sha256)
    monkeypatch.setattr(v2111_fixtures, "CONTRACT_SHA256", contract_sha256)


@pytest.fixture
def frozen_v2112_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    contract_sha256 = PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256
    assert contract_sha256 is not None
    repo_root = tmp_path / "release"
    repo_root.mkdir()
    _git(repo_root, "init", "--quiet")
    _git(repo_root, "config", "user.name", "FinEvo Test")
    _git(repo_root, "config", "user.email", "finevo-test@example.invalid")
    (repo_root / ".gitignore").write_text(
        "experiment_results/\n",
        encoding="utf-8",
    )
    _git(repo_root, "add", ".gitignore")
    _git(repo_root, "commit", "--quiet", "-m", "fixture release")
    release_commit = _git(repo_root, "rev-parse", "HEAD")
    _git(
        repo_root,
        "tag",
        "-a",
        gate.V2112_GATE_RELEASE_TAG,
        "-m",
        "fixture science tag",
    )
    _install_gate_fixture_identity(
        monkeypatch,
        release_commit=release_commit,
        contract_sha256=contract_sha256,
    )
    receipt = gate.build_v2112_post_gate_authority(**gate_fixtures._inputs())
    receipt_path = _write_receipt(repo_root, EXACT_RECEIPT, receipt)
    return {
        "repo_root": repo_root,
        "release_commit": release_commit,
        "contract_sha256": contract_sha256,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_relative": EXACT_RECEIPT.as_posix(),
    }


def _scientific_config(
    case: dict[str, Any],
    reservations: dict[str, Any],
) -> VerifiedRunConfig:
    episode_length = 6
    return VerifiedRunConfig(
        run_id="v2112-generic-consumer-regression",
        seed=2010922376,
        num_agents=2,
        episode_length=episode_length,
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
            for decision_t in range(episode_length)
        ),
        scientific_scope="preregistered_mechanism_micro_pilot",
        pilot_contract_hash=case["contract_sha256"],
        pilot_tag=gate.V2112_GATE_RELEASE_TAG,
        allow_scientific_scope=True,
        preflight_p95_reservations=reservations,
    )


def _install_provider_constructor_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    provider_constructions: list[str] = []

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        provider_constructions.append("forbidden")
        raise AssertionError("observed-p95 verification reached provider construction")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden_provider)
    monkeypatch.setattr(
        "llm_providers.create_llm_provider",
        forbidden_provider,
    )
    return provider_constructions


def test_v2112_schema_is_registered_without_replacing_prior_adapters() -> None:
    registry = authority.DEDICATED_OBSERVED_P95_BINDING_SCHEMA_REGISTRY

    assert registry[gate.V2112_GATE_SCHEMA_VERSION] == ("v2.11.2-post-gate-authority")
    assert (
        registry["finevo-pilot-v2.10.2-resealed-observed-p95-authority-v1"]
        == "v2.10.2-resealed-with-sibling-projection"
    )
    assert (
        registry["finevo-pilot-v2.11-post-gate-authority-v1"]
        == "v2.11-post-gate-authority"
    )


def test_v2112_gate_reaches_generic_binding(
    frozen_v2112_release: dict[str, Any],
) -> None:
    case = frozen_v2112_release
    dedicated = gate.verified_v2112_gate_authority_binding(
        case["receipt_relative"],
        repo_root=case["repo_root"],
        expected_git_commit=case["release_commit"],
        expected_contract_sha256=case["contract_sha256"],
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


def test_v2112_gate_reaches_runner_public_consumers_before_provider(
    frozen_v2112_release: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = frozen_v2112_release
    provider_constructions = _install_provider_constructor_sentinels(monkeypatch)
    binding = authority.verified_observed_p95_authority_binding(
        case["receipt_relative"],
        repo_root=case["repo_root"],
        expected_git_commit=case["release_commit"],
    )
    reservations = gate.runner_reservations_from_v2112_gate_binding(binding)

    payloads: list[dict[str, Any]] = []
    with observed_p95_authority_repo_context(case["repo_root"]):
        for runtime_model, by_kind in reservations.items():
            config = _scientific_config(
                case,
                {runtime_model: by_kind},
            )
            validated = validate_preflight_p95_reservations(
                config,
                provider_model_name=runtime_model,
            )
            assert set(validated) == {"action", "semantic"}
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
    assert provider_constructions == []


def test_v2112_generic_adapter_rejects_copy_outside_frozen_path(
    frozen_v2112_release: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = frozen_v2112_release
    provider_constructions = _install_provider_constructor_sentinels(monkeypatch)
    alias = Path(
        "experiment_results/pilot-v2.11.2/raw/alias/" "post_gate_authority.json"
    )
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
    assert provider_constructions == []


def test_v2112_mapping_only_verification_fails_closed(
    frozen_v2112_release: dict[str, Any],
) -> None:
    case = frozen_v2112_release

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="requires a repository-relative receipt path",
    ):
        authority.verify_observed_p95_authority_receipt(
            case["receipt"],
            repo_root=case["repo_root"],
            expected_git_commit=case["release_commit"],
        )


def test_v2112_generic_adapter_rejects_wrong_frozen_contract(
    frozen_v2112_release: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = frozen_v2112_release
    wrong_contract = "9" * 64
    _install_gate_fixture_identity(
        monkeypatch,
        release_commit=case["release_commit"],
        contract_sha256=wrong_contract,
    )
    wrong = gate.build_v2112_post_gate_authority(**gate_fixtures._inputs())
    _write_receipt(case["repo_root"], EXACT_RECEIPT, wrong)

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="contract binding mismatch",
    ):
        authority.verified_observed_p95_authority_binding(
            case["receipt_relative"],
            repo_root=case["repo_root"],
            expected_git_commit=case["release_commit"],
        )


def test_v2112_generic_adapter_rejects_dedicated_binding_substitution(
    frozen_v2112_release: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = frozen_v2112_release
    sibling_relative = EXACT_RECEIPT.with_name("sibling.json")
    sibling_inputs = gate_fixtures._inputs()
    sibling_inputs["ledger_event_chain_head"] = "f" * 64
    _write_receipt(
        case["repo_root"],
        sibling_relative,
        gate.build_v2112_post_gate_authority(**sibling_inputs),
    )
    original = gate.verified_v2112_gate_authority_binding

    def substitute_sibling(
        _receipt_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return original(sibling_relative.as_posix(), **kwargs)

    monkeypatch.setattr(
        gate,
        "verified_v2112_gate_authority_binding",
        substitute_sibling,
    )
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="changed during dedicated verification",
    ):
        authority.verified_observed_p95_authority_binding(
            case["receipt_relative"],
            repo_root=case["repo_root"],
            expected_git_commit=case["release_commit"],
        )


def test_v2112_generic_adapter_rejects_symlinked_receipt(
    frozen_v2112_release: dict[str, Any],
) -> None:
    case = frozen_v2112_release
    target = case["receipt_path"].with_name("real.json")
    case["receipt_path"].rename(target)
    case["receipt_path"].symlink_to(target.name)

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="cannot be opened safely",
    ):
        authority.verified_observed_p95_authority_binding(
            case["receipt_relative"],
            repo_root=case["repo_root"],
            expected_git_commit=case["release_commit"],
        )
