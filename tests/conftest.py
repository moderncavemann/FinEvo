from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_providers import MultiModelLLM
from verified_memory.budget import BudgetLimits, RunBudget, UsageRecord
from verified_memory.observed_p95_authority import (
    build_observed_p95_authority_receipt,
    verified_observed_p95_authority_binding,
)
from verified_memory.pilot_budget import preflight_p95
from verified_memory.pilot_checkpoint import (
    PilotCheckpoint,
    build_closed_loop_preflight_checkpoint,
    verify_closed_loop_preflight_checkpoint,
)
from verified_memory.pilot_contract import PilotContract, load_pilot_contract
from verified_memory.pilot_evaluation_amendment import (
    build_capability_import,
    load_evaluator_amendment_receipt,
    persist_evaluator_correction_receipt,
)
from verified_memory.pilot_orchestrator import (
    PILOT_PREFLIGHT_CHECKPOINT_RECEIPT_SCHEMA_VERSION,
    PILOT_PROJECTION_SCHEMA_VERSION,
    GitProvenance,
    _CheckpointPreflightResult,
    _file_sha256,
    _preflight_config,
    _seal_bound_payload,
    _usage_projection_rows,
    _verified_provider_call_journal_binding,
)
from verified_memory.pilot_preflight_amendment import (
    build_capability_bootstrap_projection,
    build_preflight_amendment_control,
    preflight_amendment_control_path,
    runner_reservations_from_bootstrap_projection,
)
from verified_memory.runner import bootstrap_config_binding_sha256
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


V23_FULL_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_3.yaml"
V23_TEST_COMMIT = "1" * 40
GPT52_MODEL_ID = "gpt52_main"
GPT52_RUNTIME_MODEL = "openai/gpt-5.2-2025-12-11"
GPT52_SERVED_MODEL = "gpt-5.2-2025-12-11"


def _write_json(path: Path, value: Any) -> None:
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


class OfflineCountingGPT52Provider(ScriptedDiagnosticProvider):
    """Run the frozen GPT-5.2 transport contract without network access."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def get_model_name(self) -> str:
        return GPT52_RUNTIME_MODEL

    def get_structured_completion(self, messages, **kwargs):
        self.prompts.append(self._prompt(messages))
        result = super().get_structured_completion(messages, **kwargs)
        return replace(
            result,
            usage=UsageRecord(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cost_usd=0.0001,
            ),
            model=GPT52_SERVED_MODEL,
            provider="openai",
            request_id=f"req_offline_v23_{len(self.prompts):02d}",
            response_model=GPT52_SERVED_MODEL,
            response_provider="OpenAI-direct",
            response_route="direct",
            request_profile_id="offline-v23-gpt52-profile",
            request_provider_pin=("OpenAI-direct",),
            request_artifact_identity=(
                ("served_snapshot", GPT52_SERVED_MODEL),
            ),
            request_price_snapshot_source=(
                "offline-fixture-price-snapshot"
            ),
            request_price_snapshot_captured_at="2026-07-24T00:00:00Z",
            finish_reason="stop",
            native_finish_reason="stop",
            response_completed=True,
            provider_sdk_name="offline-scripted-provider",
            provider_sdk_version="0.0.test",
            request_parameters=(
                "max_tokens",
                "messages",
                "model",
                "reasoning_effort",
                "response_format",
                "seed",
                "temperature",
                "top_p",
            ),
            temperature_dispatch="explicit",
            parameter_dispatch=(
                ("reasoning", "explicit_supported"),
                ("response_format", "explicit_supported"),
                ("seed", "explicit_supported"),
                ("temperature", "explicit_supported"),
                ("top_p", "explicit_supported"),
            ),
        )


@dataclass
class ObservedP95SourceChain:
    repo_root: Path
    raw_root: Path
    contract: PilotContract
    capability_spec: Any
    preflight_spec: Any
    paid: GitProvenance
    config: Any
    provider: OfflineCountingGPT52Provider
    checkpoint: PilotCheckpoint
    exactness: dict[str, Any]
    source_paths: dict[str, Path]
    source_bytes: dict[str, bytes]
    receipt: dict[str, Any]
    receipt_binding: dict[str, Any]
    reservations: dict[str, dict[str, Any]]

    def restore_sources(self) -> None:
        for name, raw in self.source_bytes.items():
            path = self.source_paths[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)

    def refresh_receipt_backed_reservations(self) -> None:
        relative = self.source_paths["receipt"].relative_to(
            self.repo_root
        ).as_posix()
        binding = verified_observed_p95_authority_binding(
            relative,
            repo_root=self.repo_root,
            expected_git_commit=self.paid.head_commit,
        )
        reservations = deepcopy(binding["reservations"])
        receipt_authority = {
            "source_authority_receipt_path": binding["receipt_path"],
            "source_authority_receipt_file_sha256": (
                binding["receipt_file_sha256"]
            ),
            "source_authority_receipt_content_sha256": (
                binding["receipt_content_sha256"]
            ),
            "source_release_commit": binding["git_commit"],
        }
        for by_kind in reservations.values():
            for entry in by_kind.values():
                entry["authority"].update(receipt_authority)
        self.receipt_binding = binding
        self.reservations = reservations


@pytest.fixture
def observed_p95_source_chain() -> ObservedP95SourceChain:
    """Build the complete V2.3 p95 authority chain without a provider call.

    Every generated source is kept below the repository's ignored
    ``experiment_results/`` tree.  The fixture reads the tracked evaluator
    correction and full frozen contract, then executes only the scripted
    12-action + 4-semantic checkpoint path.
    """

    ignored_root = ROOT / "experiment_results"
    ignored_root.mkdir(parents=True, exist_ok=True)
    raw_root = Path(
        tempfile.mkdtemp(
            prefix="pytest-observed-p95-",
            dir=ignored_root,
        )
    )
    try:
        contract = load_pilot_contract(V23_FULL_CONTRACT_PATH)
        capability_spec = contract.expand(
            stage="capability-gate",
            model=GPT52_MODEL_ID,
        )[0]
        preflight_spec = contract.expand(
            stage="closed-loop-preflight",
            model=GPT52_MODEL_ID,
        )[0]
        evaluator_receipt, _ = load_evaluator_amendment_receipt(
            repo_root=ROOT,
            contract=contract,
        )
        _, evaluator_control_path = persist_evaluator_correction_receipt(
            repo_root=ROOT,
            raw_root=raw_root,
            contract=contract,
        )

        capability = build_capability_import(
            contract,
            capability_spec,
            evaluator_receipt,
        )
        capability_dir = (
            raw_root
            / capability_spec.stage_id
            / "runs"
            / capability_spec.run_id
        )
        capability_path = capability_dir / "capability.json"
        _write_json(capability_path, capability)
        capability_file_sha256 = hashlib.sha256(
            capability_path.read_bytes()
        ).hexdigest()

        preflight_control = build_preflight_amendment_control(contract)
        assert preflight_control is not None
        preflight_control_path = preflight_amendment_control_path(
            raw_root=raw_root
        )
        _write_json(preflight_control_path, preflight_control)

        paid = GitProvenance(
            git_tag=str(contract.implementation["required_git_tag"]),
            head_commit=V23_TEST_COMMIT,
            tag_commit=V23_TEST_COMMIT,
            tag_object_type="tag",
            worktree_clean=True,
            contract_binding={},
            release_attestation=None,
        )
        provisional = _preflight_config(
            contract,
            preflight_spec,
            paid=paid,
        )
        authorized_config_sha256 = bootstrap_config_binding_sha256(
            provisional,
            measurement_role="closed_loop_preflight",
        )
        bootstrap_kwargs = {
            "source_capability_path": capability_path,
            "source_capability_file_sha256": capability_file_sha256,
            "git_tag": paid.git_tag,
            "git_commit": paid.head_commit,
            "authorized_config_sha256": authorized_config_sha256,
        }
        bootstrap_projection = build_capability_bootstrap_projection(
            contract,
            capability_spec,
            preflight_spec,
            capability,
            **bootstrap_kwargs,
        )
        bootstrap_path = (
            capability_dir / "bootstrap_projection_p95.json"
        )
        _write_json(bootstrap_path, bootstrap_projection)
        bootstrap_reservations = (
            runner_reservations_from_bootstrap_projection(
                bootstrap_projection,
                contract=contract,
                capability_spec=capability_spec,
                target_preflight_spec=preflight_spec,
                capability=capability,
                **bootstrap_kwargs,
            )
        )
        config = _preflight_config(
            contract,
            preflight_spec,
            paid=paid,
            contract_bootstrap_reservations=bootstrap_reservations,
        )

        run_dir = (
            raw_root
            / preflight_spec.stage_id
            / "runs"
            / preflight_spec.run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        journal_path = (
            raw_root
            / preflight_spec.stage_id
            / "provider_call_journals"
            / f"{preflight_spec.run_id}--preflight.json"
        )
        provider = OfflineCountingGPT52Provider()
        checkpoint = build_closed_loop_preflight_checkpoint(
            config,
            llm=MultiModelLLM(provider, num_workers=2),
            budget=RunBudget(
                BudgetLimits(max_calls=20, max_cost_usd=1.0),
                budget_id="offline-v23-observed-p95",
            ),
            env_config_source=ROOT / "config.yaml",
            call_journal_path=journal_path,
        )
        checkpoint_path = run_dir / "preflight_checkpoint.json"
        checkpoint.write_json(checkpoint_path)

        exactness_value = verify_closed_loop_preflight_checkpoint(
            checkpoint
        )
        exactness = _seal_bound_payload(
            {
                "schema_version": (
                    PILOT_PREFLIGHT_CHECKPOINT_RECEIPT_SCHEMA_VERSION
                ),
                "bindings": {
                    "contract_sha256": contract.canonical_hash,
                    "git_tag": paid.git_tag,
                    "git_commit": paid.head_commit,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_file_sha256": _file_sha256(
                        checkpoint_path
                    ),
                    "checkpoint_hash": checkpoint.checkpoint_hash,
                },
                "exactness": exactness_value,
            }
        )
        exactness_path = (
            run_dir / "preflight_checkpoint_exactness.json"
        )
        _write_json(exactness_path, exactness)

        journal_binding = _verified_provider_call_journal_binding(
            journal_path,
            expected_run_id=config.run_id,
            expected_contract_hash=contract.canonical_hash,
        )
        source_bindings = {
            "contract_sha256": contract.canonical_hash,
            "git_tag": paid.git_tag,
            "git_commit": paid.head_commit,
            "source_capability": str(capability_path),
            "source_capability_sha256": capability_file_sha256,
            "source_provider_call_journal": str(journal_path),
            "source_provider_call_journal_file_sha256": (
                journal_binding["file_sha256"]
            ),
            "source_provider_call_journal_sha256": (
                journal_binding["journal_sha256"]
            ),
            "source_bootstrap_projection": str(bootstrap_path),
            "source_bootstrap_projection_file_sha256": _file_sha256(
                bootstrap_path
            ),
            "source_bootstrap_projection_content_sha256": (
                bootstrap_projection["integrity"]["content_sha256"]
            ),
            "source_preflight_amendment_control": str(
                preflight_control_path
            ),
            "source_preflight_amendment_control_file_sha256": (
                _file_sha256(preflight_control_path)
            ),
            "source_preflight_amendment_control_content_sha256": (
                preflight_control["integrity"]["content_sha256"]
            ),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_file_sha256": _file_sha256(
                checkpoint_path
            ),
            "source_checkpoint_hash": checkpoint.checkpoint_hash,
            "source_checkpoint_exactness": str(exactness_path),
            "source_checkpoint_exactness_file_sha256": _file_sha256(
                exactness_path
            ),
            "source_checkpoint_exactness_content_sha256": (
                exactness["integrity"]["content_sha256"]
            ),
        }
        result = _CheckpointPreflightResult(checkpoint)
        usage_rows = _usage_projection_rows(capability, result)
        output_contract_map = contract.preflight_bootstrap_amendment[
            "bootstrap_policy"
        ]["source_output_contract_map"]
        normalized_usage_rows = [
            {
                **row,
                "call_kind": output_contract_map.get(
                    row["call_kind"],
                    row["call_kind"],
                ),
            }
            for row in usage_rows
        ]
        projection = preflight_p95(
            normalized_usage_rows,
            reserve_multiplier=float(
                contract.budgets["pre_dispatch_projection"][
                    "reserve_multiplier"
                ]
            ),
        )
        projection_payload = _seal_bound_payload(
            {
                "schema_version": PILOT_PROJECTION_SCHEMA_VERSION,
                "model_id": GPT52_MODEL_ID,
                "served_model": GPT52_SERVED_MODEL,
                "bindings": source_bindings,
                "projection": projection,
            }
        )
        projection_path = run_dir / "projection_p95.json"
        _write_json(projection_path, projection_payload)

        raw_root_relative = raw_root.relative_to(ROOT).as_posix()
        receipt = build_observed_p95_authority_receipt(
            repo_root=ROOT,
            contract_path="experiments/pilot_v2_3.yaml",
            raw_root=raw_root_relative,
            model_id=GPT52_MODEL_ID,
            expected_git_commit=paid.head_commit,
        )
        receipt_path = (
            run_dir / "observed_p95_authority_receipt.json"
        )
        _write_json(receipt_path, receipt)
        receipt_binding = verified_observed_p95_authority_binding(
            receipt_path.relative_to(ROOT).as_posix(),
            repo_root=ROOT,
            expected_git_commit=paid.head_commit,
        )
        reservations = deepcopy(receipt_binding["reservations"])
        receipt_authority = {
            "source_authority_receipt_path": (
                receipt_binding["receipt_path"]
            ),
            "source_authority_receipt_file_sha256": (
                receipt_binding["receipt_file_sha256"]
            ),
            "source_authority_receipt_content_sha256": (
                receipt_binding["receipt_content_sha256"]
            ),
            "source_release_commit": receipt_binding["git_commit"],
        }
        for by_kind in reservations.values():
            for entry in by_kind.values():
                entry["authority"].update(receipt_authority)

        source_paths = {
            "projection": projection_path,
            "capability": capability_path,
            "evaluator_control": evaluator_control_path,
            "bootstrap_projection": bootstrap_path,
            "preflight_amendment_control": preflight_control_path,
            "checkpoint": checkpoint_path,
            "checkpoint_exactness": exactness_path,
            "provider_call_journal": journal_path,
            "receipt": receipt_path,
        }
        case = ObservedP95SourceChain(
            repo_root=ROOT,
            raw_root=raw_root,
            contract=contract,
            capability_spec=capability_spec,
            preflight_spec=preflight_spec,
            paid=paid,
            config=config,
            provider=provider,
            checkpoint=checkpoint,
            exactness=exactness,
            source_paths=source_paths,
            source_bytes={
                name: path.read_bytes()
                for name, path in source_paths.items()
            },
            receipt=receipt,
            receipt_binding=receipt_binding,
            reservations=reservations,
        )
        yield case
    finally:
        shutil.rmtree(raw_root, ignore_errors=True)
