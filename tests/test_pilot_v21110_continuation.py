from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import json
import textwrap

import pytest

from verified_memory import pilot_contract
from verified_memory import pilot_v21110_continuation as continuation


ROOT = Path(__file__).resolve().parents[1]


def test_v21110_v2119_terminal_reader_enables_artifact_binding() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(continuation.verify_v2119_terminal_no_go))
    )
    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PilotRunLedger"
    ]
    assert len(constructors) == 1
    keywords = {keyword.arg: keyword.value for keyword in constructors[0].keywords}
    assert isinstance(keywords.get("bind_terminal_artifacts"), ast.Constant)
    assert keywords["bind_terminal_artifacts"].value is True


def test_v21110_frozen_v2119_boundary_matches_contract_validator() -> None:
    expected = pilot_contract._v2_11_10_expected_recovery_boundary(
        status="draft"
    )["failed_release_no_go"]
    assert continuation._expected_v2119_failed_release_no_go() == expected
    assert expected["run_ledger"]["status_counts"] == {
        "complete": 1,
        "failed": 86,
    }
    assert expected["budget_ledger"]["current_actual"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "storage_bytes": 800_162,
    }
    assert expected["acceptance_receipt_present"] is True
    assert expected["science_reservations"] == 36
    assert expected["provider_calls"] == 0
    assert expected["hosted_completions"] == 0


def test_v21110_parent_debit_is_exact_terminal_v2119_debit() -> None:
    assert continuation.V21110_PARENT_TERMINAL_EVIDENCE_SCOPE == (
        "preregistered_terminal_lineage_authority_import"
    )
    expected = {
        "parent_contract_sha256": continuation.V2119_CONTRACT_SHA256,
        "parent_run_ledger_sha256": continuation.V2119_RUN_LEDGER_SHA256,
        "parent_budget_ledger_sha256": continuation.V2119_BUDGET_LEDGER_SHA256,
        "stage_bucket": "parent_v2119",
        "cost_usd": 63.1196450625,
        "hosted_completions": 3_440,
        "storage_bytes": 270_993_662,
    }
    contract = SimpleNamespace(
        contract_id=continuation.V21110_CONTRACT_ID,
        v21110_recovery_boundary={"parent_budget_debit": expected},
    )
    debit = continuation.parent_budget_debit_for_v21110(contract)
    assert debit.to_dict() == {
        **expected,
        "schema_version": "finevo-parent-budget-debit-v1",
        "record_sha256": continuation.V21110_PARENT_DEBIT_RECORD_SHA256,
    }

    call = {
        "call_kind": "action",
        "decision_t": 4,
        "agent_id": 2,
        "prompt_hash": "a" * 64,
        "raw_output_hash": "b" * 64,
    }
    completion = {
        **call,
        "provider": "openai",
        "model": "gpt-5.2-2025-12-11",
        "completion_tokens": 17,
    }
    disposition = {
        **call,
        "parse_status": "success",
        "parse_mode": "exact_json",
        "accepted": True,
    }

    def project(
        completion_payload: Mapping[str, Any],
        disposition_payload: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        return continuation._v21110_journal_api_usage_projection(
            {
                "events": [
                    {
                        "event_type": "completion_received",
                        "payload": dict(completion_payload),
                    },
                    {
                        "event_type": "parse_disposition",
                        "payload": dict(disposition_payload),
                    },
                ]
            }
        )

    sealed = [{**completion, "action_parse_mode": "exact_json"}]
    assert project(completion, disposition) == sealed

    with pytest.raises(
        continuation.PilotV21110ContinuationError,
        match="action parse disposition",
    ):
        project(completion, {**disposition, "accepted": False})
    with pytest.raises(
        continuation.PilotV21110ContinuationError,
        match="action parse disposition",
    ):
        project(completion, {**disposition, "raw_output_hash": "c" * 64})

    parse_mode_drift = project(
        completion,
        {**disposition, "parse_mode": "fenced_recovery"},
    )
    provider_field_drift = project(
        {**completion, "completion_tokens": 18},
        disposition,
    )
    assert pilot_contract.canonical_sha256(parse_mode_drift) != (
        pilot_contract.canonical_sha256(sealed)
    )
    assert pilot_contract.canonical_sha256(provider_field_drift) != (
        pilot_contract.canonical_sha256(sealed)
    )


def _contract_pin_source(
    *, canonical: str, manifest_file: str, manifest_content: str
) -> str:
    return "\n".join(
        (
            f"PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256 = {canonical}",
            f"PILOT_V2_11_10_SOURCE_MANIFEST_FILE_SHA256 = {manifest_file}",
            f"PILOT_V2_11_10_SOURCE_MANIFEST_CONTENT_SHA256 = {manifest_content}",
        )
    )


@pytest.mark.parametrize(
    ("canonical", "manifest_file", "manifest_content"),
    (
        ("None", "None", "None"),
        ("None", repr("a" * 64), repr("b" * 64)),
        (repr("c" * 64), repr("a" * 64), repr("b" * 64)),
    ),
)
def test_v21110_contract_cycle_normalization_accepts_exact_bootstrap_states(
    tmp_path: Path,
    canonical: str,
    manifest_file: str,
    manifest_content: str,
) -> None:
    path = tmp_path / "pilot_contract.py"
    path.write_text(
        _contract_pin_source(
            canonical=canonical,
            manifest_file=manifest_file,
            manifest_content=manifest_content,
        ),
        encoding="utf-8",
    )
    binding = continuation._normalized_contract_module_ast_binding(
        path,
        require_v21110_cycle_pins=True,
    )
    assert "bootstrap_none" not in binding
    assert set(binding["replaced_cycle_pins"]) == (
        continuation._CYCLIC_V21110_CONTRACT_PIN_NAMES
    )


def test_v21110_contract_cycle_normalization_rejects_half_sealed_source_pair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pilot_contract.py"
    path.write_text(
        _contract_pin_source(
            canonical="None",
            manifest_file=repr("a" * 64),
            manifest_content="None",
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        continuation.PilotV21110ContinuationError,
        match="source pins must both",
    ):
        continuation._normalized_contract_module_ast_binding(
            path,
            require_v21110_cycle_pins=True,
        )


def test_v21110_contract_cycle_normalization_rejects_pinned_canonical_with_unsealed_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pilot_contract.py"
    path.write_text(
        _contract_pin_source(
            canonical=repr("c" * 64),
            manifest_file="None",
            manifest_content="None",
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        continuation.PilotV21110ContinuationError,
        match="source pins must both",
    ):
        continuation._normalized_contract_module_ast_binding(
            path,
            require_v21110_cycle_pins=True,
        )


def _ci_anchor_source(*, file_pin: str, content_pin: str) -> str:
    return (
        "SCIENTIFIC_SOURCE_MANIFEST_ANCHORS = {\n"
        "    'v21110': {\n"
        "        'path': "
        f"{continuation.V21110_SOURCE_MANIFEST_PATH.as_posix()!r},\n"
        f"        'file_sha256': {file_pin},\n"
        f"        'content_sha256': {content_pin},\n"
        "    },\n"
        "}\n"
    )


@pytest.mark.parametrize(
    ("file_pin", "content_pin"),
    (
        ("None", "None"),
        (repr("d" * 64), repr("e" * 64)),
    ),
)
def test_v21110_ci_anchor_normalization_accepts_paired_states(
    tmp_path: Path,
    file_pin: str,
    content_pin: str,
) -> None:
    path = tmp_path / "ci_release_receipt.py"
    path.write_text(
        _ci_anchor_source(file_pin=file_pin, content_pin=content_pin),
        encoding="utf-8",
    )
    binding = continuation._normalized_ci_release_module_ast_binding(path)
    assert "bootstrap_none" not in binding
    assert binding["replaced_cycle_pins"] == ["content_sha256", "file_sha256"]


def test_v21110_ci_anchor_normalization_rejects_half_sealed_pair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ci_release_receipt.py"
    path.write_text(
        _ci_anchor_source(file_pin=repr("d" * 64), content_pin="None"),
        encoding="utf-8",
    )
    with pytest.raises(
        continuation.PilotV21110ContinuationError,
        match="must both be None",
    ):
        continuation._normalized_ci_release_module_ast_binding(path)


def test_v21110_normalized_bindings_are_identical_across_three_freeze_phases(
    tmp_path: Path,
) -> None:
    contract_bindings = []
    for index, pins in enumerate(
        (
            ("None", "None", "None"),
            ("None", repr("a" * 64), repr("b" * 64)),
            (repr("c" * 64), repr("a" * 64), repr("b" * 64)),
        )
    ):
        path = tmp_path / f"pilot_contract_{index}.py"
        path.write_text(
            _contract_pin_source(
                canonical=pins[0],
                manifest_file=pins[1],
                manifest_content=pins[2],
            ),
            encoding="utf-8",
        )
        contract_bindings.append(
            continuation._normalized_contract_module_ast_binding(
                path,
                require_v21110_cycle_pins=True,
            )
        )
    assert contract_bindings == [contract_bindings[0]] * 3

    ci_bindings = []
    for index, pins in enumerate(
        (("None", "None"), (repr("d" * 64), repr("e" * 64)))
    ):
        path = tmp_path / f"ci_release_receipt_{index}.py"
        path.write_text(
            _ci_anchor_source(file_pin=pins[0], content_pin=pins[1]),
            encoding="utf-8",
        )
        ci_bindings.append(
            continuation._normalized_ci_release_module_ast_binding(path)
        )
    assert ci_bindings == [ci_bindings[0]] * 2


def test_v21110_source_manifest_fixture_keeps_failed_release_and_authority_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = json.loads(
        (ROOT / "experiments/pilot_v2_11_10.yaml").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        pilot_contract,
        "PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256",
        pilot_contract.science_design_sha256(document),
    )
    contract = pilot_contract.PilotContract.from_dict(document)
    child = tmp_path / "child"
    failed = tmp_path / "failed-v2119"
    authority = tmp_path / "authority-v2115"
    for root in (child, failed, authority):
        root.mkdir()

    mapping_sha256 = contract.v21110_recovery_boundary["continuation_matrix"][
        "canonical_86_row_mapping_sha256"
    ]
    state = {
        "failed_release": {
            "git_tag": continuation.V2119_SCIENCE_TAG,
            "git_commit": continuation.V2119_SCIENCE_COMMIT,
            "tag_object_type": "tag",
        },
        "raw_inventory": {
            "evidence": continuation._expected_v2119_failed_release_no_go()[
                "raw_inventory"
            ],
            "complete": continuation._expected_v2119_failed_release_no_go()[
                "complete_raw_inventory"
            ],
        },
        "authority": {
            "contract": SimpleNamespace(),
            "release": {},
            "raw_inventory": {"fixture": "sealed-v2115-authority"},
        },
    }
    monkeypatch.setattr(
        continuation,
        "verify_v2119_terminal_no_go",
        lambda **_kwargs: state,
    )
    monkeypatch.setattr(
        continuation,
        "_canonical_remaining_cell_mapping",
        lambda *_args, **_kwargs: {
            "schema_version": "fixture-v1",
            "row_count": 86,
            "mapping_sha256": mapping_sha256,
            "rows": [],
        },
    )
    monkeypatch.setattr(
        continuation,
        "_current_runtime_source_bindings",
        lambda *_args, **_kwargs: {
            "fixture": "complete-v21110-runtime-source-surface"
        },
    )
    monkeypatch.setattr(
        continuation,
        "_remaining_science_implementation_equivalence",
        lambda *_args, **_kwargs: {
            "fixture": "science-core-equal-except-p95-layer-adapter"
        },
    )

    manifest = continuation.build_v21110_source_manifest(
        contract=contract,
        repo_root=child,
        failed_repo_root=failed,
        authority_repo_root=authority,
    )
    assert manifest["schema_version"] == (
        continuation.V21110_SOURCE_MANIFEST_SCHEMA_VERSION
    )
    assert manifest["failed_terminal_no_go"] == (
        continuation._expected_v2119_failed_release_no_go()
    )
    assert manifest["failed_raw_inventory"] == state["raw_inventory"]["evidence"]
    assert manifest["authority_release"]["contract_id"] == (
        continuation.v2117.V2115_CONTRACT_ID
    )
    assert manifest["observed_p95_authority_adapter_recovery"] == {
        "failed_release_contract_id": continuation.V2119_CONTRACT_ID,
        "failure_error_type": "ValueError",
        "failure_message_by_model": dict(
            continuation.V2119_FAILURE_MESSAGE_BY_MODEL
        ),
        "producer_core_authority_field_count": 13,
        "runner_envelope_authority_field_count": 17,
        "runner_receipt_envelope_field_count": 4,
        "authority_release_contract_id": continuation.v2117.V2115_CONTRACT_ID,
        "authority_gate_path": (
            continuation.V2115_RAW_ROOT
            / "long-context-preflight/post_gate_authority.json"
        ).as_posix(),
        "authority_gate_content_sha256": continuation.V2115_POST_GATE_CONTENT_SHA256,
        "repair_changes_scientific_design": False,
        "scientific_outcomes_inspected_for_repair": False,
        "additional_source_release_required": False,
        "provider_construction": False,
        "provider_calls": 0,
    }
    assert manifest["observation_boundary"]["failed_v2119_effect_rows_imported"] == 0
    assert manifest["observation_boundary"]["provider_calls"] == 0
    assert len(manifest["integrity"]["content_sha256"]) == 64
