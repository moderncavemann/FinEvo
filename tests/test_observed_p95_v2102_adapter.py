from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from verified_memory import observed_p95_authority as authority
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v2102_parent_import as parent_import
from verified_memory import pilot_v210_parent_import as v210_parent_import
from verified_memory.pilot_checkpoint import config_from_dict
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.runner import (
    OBSERVED_P95_AUTHORITY_ID,
    OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
    OBSERVED_P95_SOURCE_KIND,
    ShockEvent,
    VerifiedRunConfig,
    has_sealed_observed_p95_authority,
    observed_p95_authority_repo_context,
    serialized_has_sealed_observed_p95_authority,
    validate_preflight_p95_reservations,
)


ROOT = Path(__file__).resolve().parents[1]


def _reservation(
    *,
    prompt_tokens: float,
    completion_tokens: float,
    cost_usd: float,
    sample_count: int,
) -> dict[str, Any]:
    reserved_prompt = int(prompt_tokens * 1.25 + 0.999999)
    reserved_completion = int(completion_tokens * 1.25 + 0.999999)
    reserved_cost = cost_usd * 1.25
    return {
        "sample_count": sample_count,
        "raw_p95": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost_usd,
        },
        "reserved_p95": {
            "prompt_tokens": reserved_prompt,
            "completion_tokens": reserved_completion,
            "total_tokens": reserved_prompt + reserved_completion,
            "cost_usd": reserved_cost,
        },
        "reserve_multiplier": 1.25,
    }


def _source_authority(
    *,
    profile_id: str,
    served_model: str,
) -> dict[str, Any]:
    return {
        "authority_id": OBSERVED_P95_AUTHORITY_ID,
        "source_kind": OBSERVED_P95_SOURCE_KIND,
        "pilot_contract_hash": parent_import.V28_CONTRACT_CANONICAL_SHA256,
        "pilot_tag": parent_import.V28_SCIENCE_TAG,
        "source_projection_schema_version": (
            OBSERVED_P95_PROJECTION_SCHEMA_VERSION
        ),
        "source_projection_file_sha256": "1" * 64,
        "source_projection_content_sha256": "2" * 64,
        "source_preflight_run_id": f"fixture-{profile_id}-preflight",
        "source_preflight_run_spec_sha256": "3" * 64,
        "source_model_id": profile_id,
        "source_served_model": served_model,
        "source_execution_artifact_sha256": "4" * 64,
        "source_provider_call_journal_sha256": "5" * 64,
    }


def _v29_source_binding(contract: Any, profile_id: str) -> dict[str, Any]:
    profile = contract.provider_profiles[profile_id]
    runtime_model = f"{profile.transport}/{profile.requested_model}"
    cost = 0.01 if profile.transport == "openai" else 0.0
    reservations = {
        runtime_model: {
            "action": {
                "authority": _source_authority(
                    profile_id=profile_id,
                    served_model=profile.served_model,
                ),
                "reservation": _reservation(
                    prompt_tokens=100.0,
                    completion_tokens=20.0,
                    cost_usd=cost,
                    sample_count=36,
                ),
            },
            "semantic": {
                "authority": _source_authority(
                    profile_id=profile_id,
                    served_model=profile.served_model,
                ),
                "reservation": _reservation(
                    prompt_tokens=240.0,
                    completion_tokens=60.0,
                    cost_usd=cost * 2,
                    sample_count=10,
                ),
            },
        }
    }
    nested = {
        "profile_id": profile_id,
        "source_contract_id": parent_import.V28_CONTRACT_ID,
        "source_contract_sha256": (
            parent_import.V28_CONTRACT_CANONICAL_SHA256
        ),
        "source_git_commit": parent_import.V28_SCIENCE_COMMIT,
        "source_git_tag": parent_import.V28_SCIENCE_TAG,
        "authority": {
            "path": (
                "experiment_results/pilot-v2.8/raw/parent-import/"
                f"observed_p95/{profile_id}/"
                "observed_p95_authority_receipt.json"
            ),
            "schema_version": (
                "finevo-pilot-v2.8-inherited-observed-p95-authority-v1"
            ),
            "file_sha256": "6" * 64,
            "content_sha256": "7" * 64,
        },
        "projection": {
            "path": (
                "experiment_results/pilot-v2.8/raw/parent-import/"
                f"observed_p95/{profile_id}/projection_p95.json"
            ),
            "schema_version": OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
            "file_sha256": "8" * 64,
            "content_sha256": "9" * 64,
        },
        "runtime_model": runtime_model,
        "served_model": profile.served_model,
        "reservations": reservations,
    }
    normalized = v210_parent_import.normalize_v29_observed_p95_binding(
        nested,
        profile_id=profile_id,
    )
    return {
        "source_path_kind": (
            "byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"
        ),
        "v2_9_terminal_parent": {
            "contract_id": parent_import.V29_CONTRACT_ID,
            "contract_sha256": parent_import.V29_CONTRACT_CANONICAL_SHA256,
            "science_tag": parent_import.V29_SCIENCE_TAG,
            "science_commit": parent_import.V29_SCIENCE_COMMIT,
            "raw_file_count": parent_import.V29_RAW_FILE_COUNT,
            "raw_storage_bytes": parent_import.V29_RAW_STORAGE_BYTES,
            "raw_inventory_sha256": parent_import.V29_RAW_INVENTORY_SHA256,
            "terminal_status": "complete-with-no-go",
            "implementation_root_cause": (
                "imported-p95-runner-binding-shape-mismatch"
            ),
        },
        "v2_8_observed_p95_origin": nested,
        "normalized_v2_9_binding": normalized,
    }


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture(
    params=("gpt52_main", "llama33_local_controlled"),
)
def current_release_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> dict[str, Any]:
    profile_id = str(request.param)
    repo_root = tmp_path / "release"
    shutil.copytree(ROOT / "experiments", repo_root / "experiments")
    contract_path = (
        repo_root
        / parent_import.V2102_EXPANDED_CONTRACT_PATH.as_posix()
    )
    contract = load_pilot_contract(contract_path)

    _git(repo_root, "init", "--quiet")
    _git(repo_root, "config", "user.name", "FinEvo Test")
    _git(repo_root, "config", "user.email", "finevo-test@example.invalid")
    (repo_root / ".gitignore").write_text(
        "experiment_results/\n",
        encoding="utf-8",
    )
    _git(repo_root, "add", "experiments", ".gitignore")
    _git(repo_root, "commit", "--quiet", "-m", "fixture release")
    commit = _git(repo_root, "rev-parse", "HEAD")
    _git(
        repo_root,
        "tag",
        "-a",
        parent_import.V2102_SCIENCE_TAG,
        "-m",
        "fixture science tag",
    )

    monkeypatch.setattr(
        parent_import,
        "_validate_target_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        parent_import,
        "_verify_release_identity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        parent_import,
        "_load_current_contract",
        lambda _root, _selected=None: contract,
    )
    source = _v29_source_binding(contract, profile_id)
    raw_root = repo_root.joinpath(*parent_import.V2102_RAW_ROOT.parts)
    built = parent_import.build_v2102_resealed_observed_p95_authority(
        repo_root=repo_root,
        contract=contract,
        raw_root=raw_root,
        profile_id=profile_id,
        expected_git_commit=commit,
        verified_v2_9_source_binding=source,
    )
    receipt_path = parent_import.v2102_observed_p95_receipt_path(
        raw_root,
        profile_id,
    )
    projection_path = parent_import.v2102_observed_p95_projection_path(
        raw_root,
        profile_id,
    )
    _write_json(receipt_path, built["receipt"])
    _write_json(projection_path, built["projection"])
    receipt_relative = receipt_path.relative_to(repo_root).as_posix()

    monkeypatch.setattr(
        parent_import,
        "v2_9_p95_source_binding_v2102",
        lambda **kwargs: (
            deepcopy(source)
            if kwargs
            == {
                "child_raw_root": raw_root,
                "profile_id": profile_id,
            }
            else pytest.fail("dedicated verifier used a non-frozen raw root")
        ),
    )
    return {
        "repo_root": repo_root,
        "raw_root": raw_root,
        "contract": contract,
        "profile_id": profile_id,
        "commit": commit,
        "receipt_path": receipt_path,
        "receipt_relative": receipt_relative,
        "projection_path": projection_path,
        "built": built,
    }


def test_v2102_producer_schema_is_registered_by_generic_consumer() -> None:
    assert authority.DEDICATED_OBSERVED_P95_BINDING_SCHEMA_REGISTRY == {
        "finevo-pilot-v2.10.1-resealed-observed-p95-authority-v1": (
            "v2.10.1-resealed-with-sibling-projection"
        ),
        parent_import.V2102_RESEALED_P95_AUTHORITY_SCHEMA_VERSION: (
            "v2.10.2-resealed-with-sibling-projection"
        )
    }


def test_v2102_producer_pair_reaches_generic_binding(
    current_release_pair: dict[str, Any],
) -> None:
    case = current_release_pair
    dedicated = parent_import.verified_v2102_observed_p95_authority_binding(
        case["receipt_relative"],
        repo_root=case["repo_root"],
        raw_root=case["raw_root"],
        expected_git_commit=case["commit"],
        contract=case["contract"],
    )

    generic = authority.verified_observed_p95_authority_binding(
        case["receipt_relative"],
        repo_root=case["repo_root"],
        expected_git_commit=case["commit"],
    )
    verified_rows = authority.verify_observed_p95_authority_receipt(
        case["receipt_relative"],
        repo_root=case["repo_root"],
        expected_git_commit=case["commit"],
    )

    assert generic == dedicated
    assert generic["reservations"] == case["built"]["receipt"]["reservations"]
    assert verified_rows == generic["reservations"]
    assert set(next(iter(generic["reservations"].values()))) == {
        "action",
        "semantic",
    }


def _runner_reservations(binding: dict[str, Any]) -> dict[str, Any]:
    reservations = deepcopy(binding["reservations"])
    for by_kind in reservations.values():
        for entry in by_kind.values():
            entry["authority"].update(
                {
                    "source_authority_receipt_path": binding["receipt_path"],
                    "source_authority_receipt_file_sha256": (
                        binding["receipt_file_sha256"]
                    ),
                    "source_authority_receipt_content_sha256": (
                        binding["receipt_content_sha256"]
                    ),
                    "source_release_commit": binding["git_commit"],
                }
            )
    return reservations


def _scientific_runner_config(
    case: dict[str, Any],
    binding: dict[str, Any],
) -> VerifiedRunConfig:
    episode_length = 6
    return VerifiedRunConfig(
        run_id=f"runner-consumer-{case['profile_id']}",
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
        pilot_contract_hash=case["contract"].canonical_hash,
        pilot_tag=parent_import.V2102_SCIENCE_TAG,
        allow_scientific_scope=True,
        preflight_p95_reservations=_runner_reservations(binding),
    )


def _install_provider_constructor_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    provider_constructions: list[str] = []

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        provider_constructions.append("forbidden")
        raise AssertionError(
            "observed-p95 verification reached provider construction"
        )

    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        forbidden_provider,
    )
    monkeypatch.setattr(
        orchestrator,
        "create_llm_provider",
        forbidden_provider,
    )
    monkeypatch.setattr(
        "llm_providers.create_llm_provider",
        forbidden_provider,
    )
    return provider_constructions


def test_v2102_producer_pair_reaches_runner_public_consumers_before_provider(
    current_release_pair: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = current_release_pair
    provider_constructions = _install_provider_constructor_sentinels(
        monkeypatch
    )

    binding = authority.verified_observed_p95_authority_binding(
        case["receipt_relative"],
        repo_root=case["repo_root"],
        expected_git_commit=case["commit"],
    )
    runtime_model = next(iter(binding["reservations"]))
    with observed_p95_authority_repo_context(case["repo_root"]):
        config = _scientific_runner_config(case, binding)
        validated = validate_preflight_p95_reservations(
            config,
            provider_model_name=runtime_model,
        )
        assert set(validated) == {"action", "semantic"}
        assert has_sealed_observed_p95_authority(config) is True

        payload = config.to_dict()
        restored = config_from_dict(payload)
        assert restored.to_dict() == payload
        assert has_sealed_observed_p95_authority(restored) is True

    assert serialized_has_sealed_observed_p95_authority(
        payload,
        authority_repo_root=case["repo_root"],
    ) is True
    assert provider_constructions == []


def test_v2102_generic_binding_rejects_sibling_projection_tamper(
    current_release_pair: dict[str, Any],
) -> None:
    case = current_release_pair
    tampered = deepcopy(case["built"]["projection"])
    tampered["bindings"]["source_kind"] = "tampered-current-release-source"
    tampered = parent_import._seal(tampered)
    _write_json(case["projection_path"], tampered)

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="projection differs from its receipt/source",
    ):
        authority.verified_observed_p95_authority_binding(
            case["receipt_relative"],
            repo_root=case["repo_root"],
            expected_git_commit=case["commit"],
        )


def test_v2102_mapping_only_verification_fails_closed_without_projection(
    current_release_pair: dict[str, Any],
) -> None:
    case = current_release_pair

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="requires a repository-relative receipt path",
    ):
        authority.verify_observed_p95_authority_receipt(
            case["built"]["receipt"],
            repo_root=case["repo_root"],
            expected_git_commit=case["commit"],
        )


def test_unknown_dedicated_schema_fails_closed_before_provider(
    current_release_pair: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = current_release_pair
    provider_constructions = _install_provider_constructor_sentinels(
        monkeypatch
    )
    unknown = deepcopy(case["built"]["receipt"])
    unknown["schema_version"] = (
        "finevo-pilot-v2.10.2-unknown-observed-p95-authority-v1"
    )
    unknown = parent_import._seal(unknown)
    _write_json(case["receipt_path"], unknown)

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="top-level shape or schema drifted",
    ):
        authority.verified_observed_p95_authority_binding(
            case["receipt_relative"],
            repo_root=case["repo_root"],
            expected_git_commit=case["commit"],
        )
    assert provider_constructions == []


def test_v2102_generic_binding_rejects_pair_copied_outside_frozen_path(
    current_release_pair: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = current_release_pair
    provider_constructions = _install_provider_constructor_sentinels(
        monkeypatch
    )
    alias_dir = (
        case["repo_root"]
        / "experiment_results"
        / "alias"
        / case["profile_id"]
    )
    alias_dir.mkdir(parents=True)
    alias_receipt = alias_dir / "observed_p95_authority_receipt.json"
    shutil.copyfile(case["receipt_path"], alias_receipt)
    shutil.copyfile(
        case["projection_path"],
        alias_dir / "projection_p95.json",
    )

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="outside its exact frozen current-release path",
    ):
        authority.verified_observed_p95_authority_binding(
            alias_receipt.relative_to(case["repo_root"]).as_posix(),
            repo_root=case["repo_root"],
            expected_git_commit=case["commit"],
        )
    assert provider_constructions == []


def test_v2102_generic_binding_rejects_symlinked_sibling_projection(
    current_release_pair: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = current_release_pair
    provider_constructions = _install_provider_constructor_sentinels(
        monkeypatch
    )
    projection_target = case["projection_path"].with_name(
        "projection-target.json"
    )
    case["projection_path"].rename(projection_target)
    case["projection_path"].symlink_to(projection_target.name)

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="projection path cannot be opened safely",
    ):
        authority.verified_observed_p95_authority_binding(
            case["receipt_relative"],
            repo_root=case["repo_root"],
            expected_git_commit=case["commit"],
        )
    assert provider_constructions == []


def test_v2102_guarded_projection_snapshot_blocks_transient_valid_swap(
    current_release_pair: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = current_release_pair
    provider_constructions = _install_provider_constructor_sentinels(
        monkeypatch
    )
    valid_projection = case["projection_path"].read_bytes()
    tampered = deepcopy(case["built"]["projection"])
    tampered["bindings"]["source_kind"] = "transient-tampered-source"
    tampered = parent_import._seal(tampered)
    _write_json(case["projection_path"], tampered)

    original_reader = authority._read_json_source
    swaps: list[str] = []

    def restore_valid_after_guarded_read(
        repo_root: Path,
        relative: Any,
        *,
        name: str,
    ) -> tuple[dict[str, Any], bytes]:
        result = original_reader(
            repo_root,
            relative,
            name=name,
        )
        if name == "V2.10.2 observed-p95 projection":
            case["projection_path"].write_bytes(valid_projection)
            swaps.append(relative.as_posix())
        return result

    monkeypatch.setattr(
        authority,
        "_read_json_source",
        restore_valid_after_guarded_read,
    )
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="projection differs from its receipt/source",
    ):
        authority.verified_observed_p95_authority_binding(
            case["receipt_relative"],
            repo_root=case["repo_root"],
            expected_git_commit=case["commit"],
        )

    assert swaps == [
        case["receipt_path"].with_name("projection_p95.json").relative_to(
            case["repo_root"]
        ).as_posix()
    ]
    assert case["projection_path"].read_bytes() == valid_projection
    assert provider_constructions == []
