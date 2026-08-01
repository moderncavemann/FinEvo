from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest

import scripts.build_v2115_reviewer_trace as trace_cli
from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
import verified_memory.reviewer_closed_loop_trace as trace_module
from verified_memory.reviewer_closed_loop_trace import (
    JsonlRecord,
    ReviewerTraceError,
    ReviewerTraceUnavailable,
    TRACE_POLICY_ID,
    TRACE_SCHEMA_VERSION,
    V2115_CONTRACT_CANONICAL_SHA256,
    V2115_CONTRACT_FILE_SHA256,
    V2115_SOURCE_COMMIT,
    V2115_SOURCE_TAG,
    V2115_SOURCE_TAG_OBJECT,
    _absolute_without_symlink,
    _build_trace_artifact_verified,
    _claim_boundary,
    _fixed_run_manifest_path,
    _git_blob_oid,
    _load_jsonl,
    _recompute_rule_classification,
    _scan_publication_bytes,
    _seal_artifact,
    _select_record,
    _validate_against_runtime_schema,
    _verify_publisher_checkout,
    _verify_source_checkout,
    _write_trace_package,
    derive_selection,
    validate_trace_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"
SCHEMA_PATH = ROOT / "schemas" / "reviewer_closed_loop_trace_v1.schema.json"
MODULE_PATH = ROOT / "verified_memory" / "reviewer_closed_loop_trace.py"


def _publisher_provenance() -> dict[str, Any]:
    return {
        "publisher_id": "publication-consumer:reviewer-trace-v1",
        "git_commit": "1" * 40,
        "tracked_worktree_clean": True,
        "implementation_base_commit": (
            "34134f2624833e45f0e1f559332b0d11ea1942d6"
        ),
        "required_tracked_head_blobs": {
            "schemas/reviewer_closed_loop_trace_v1.schema.json": _git_blob_oid(
                SCHEMA_PATH.read_bytes()
            )
        },
        "publication_provider_calls": 0,
    }


def _source_provenance() -> dict[str, Any]:
    return {
        "source_id": "sealed-science:pilot-v2.11.5-science",
        "git_tag": V2115_SOURCE_TAG,
        "tag_object": V2115_SOURCE_TAG_OBJECT,
        "resolved_git_commit": V2115_SOURCE_COMMIT,
        "detached_head": True,
        "tracked_worktree_clean": True,
    }


def _unavailable_artifact(
    publisher: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = load_pilot_contract(CONTRACT_PATH)
    _, selection = derive_selection(contract)
    return _seal_artifact(
        {
            "schema_version": TRACE_SCHEMA_VERSION,
            "artifact_id": "finevo-pilot-v2.11.5--reviewer-closed-loop--unavailable",
            "status": "unavailable",
            "publication_provider_calls": 0,
            "selection_policy": selection,
            "evidence_scope": {
                "diagnostic_only": True,
                "descriptive_only": True,
                "effectiveness_evidence": False,
                "frozen_source_provider_call_scope": (
                    "historical-observations-read-from-sealed-logs"
                ),
                "publication_provider_calls": 0,
                "stage_authoritative": False,
            },
            "provenance": {
                "contract_id": contract.contract_id,
                "contract_file_sha256": V2115_CONTRACT_FILE_SHA256,
                "contract_canonical_sha256": V2115_CONTRACT_CANONICAL_SHA256,
                "science_source": _source_provenance(),
                "publisher": publisher or _publisher_provenance(),
            },
            "source_files": [],
            "source_records": [],
            "trace": None,
            "failure_reason": "fixed coordinate absent; no fallback attempted",
            "link_checks": {"all_pass": False, "checks": {}},
            "claim_boundary": _claim_boundary(),
        }
    )


def test_contract_indexed_selection_is_first_recovery_and_has_no_fallback() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    spec, selection = derive_selection(contract)

    assert selection["policy_id"] == TRACE_POLICY_ID
    assert selection["selection_timing"] == "publication-time-post-seal"
    assert selection["preregistered"] is False
    assert selection["human_prior_case_awareness"] is True
    assert selection["outcome_fields_used_by_selector"] is False
    assert selection["fallback_policy"] == "none-emit-unavailable"
    assert selection["selected_coordinates"] == {
        "stage_id": "experiment-a",
        "model_id": "gpt52_main",
        "arm_id": "full",
        "narrative_id": "none",
        "utility_profile_id": "stage0-selected",
        "seed": 1099057501,
        "agent_id": 0,
        "decision_t": 8,
        "outcome_t": 9,
        "next_decision_t": 9,
        "run_id": (
            "finevo-pilot-v2.11.5--experiment-a--gpt52_main--full--none--"
            "stage0-selected--s1099057501"
        ),
    }
    body = copy.deepcopy(selection)
    claimed = body.pop("policy_sha256")
    assert claimed == canonical_sha256(body)
    assert spec.run_id == selection["selected_coordinates"]["run_id"]


def test_fixed_coordinate_selector_never_falls_back_to_another_row() -> None:
    records = (
        JsonlRecord(
            line_number=1,
            raw_line_sha256="a" * 64,
            value={"agent_id": 1, "decision_t": 8},
        ),
        JsonlRecord(
            line_number=2,
            raw_line_sha256="b" * 64,
            value={"agent_id": 0, "decision_t": 7},
        ),
    )

    with pytest.raises(ReviewerTraceUnavailable, match="fixed coordinate"):
        _select_record(
            records,
            source_id="actions",
            selector={"agent_id": 0, "decision_t": 8},
            predicate=lambda row: row.get("agent_id") == 0
            and row.get("decision_t") == 8,
        )


def test_jsonl_binding_hashes_the_exact_raw_row_without_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    raw = b'{"decision_t":8,"agent_id":0}'
    path.write_bytes(raw + b"\n")

    records = _load_jsonl(path)

    assert len(records) == 1
    assert records[0].line_number == 1
    assert records[0].raw_line_sha256 == hashlib.sha256(raw).hexdigest()


def test_trace_schema_and_manual_validator_lock_claim_and_provider_boundaries() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    artifact = _unavailable_artifact()

    Draft202012Validator(schema).validate(artifact)
    validate_trace_artifact(artifact)
    assert artifact["publication_provider_calls"] == 0
    assert artifact["evidence_scope"]["frozen_source_provider_call_scope"] == (
        "historical-observations-read-from-sealed-logs"
    )
    assert artifact["claim_boundary"]["no_causal_attribution"] is True
    assert artifact["claim_boundary"]["stage_decision_unchanged"] is True

    tampered = copy.deepcopy(artifact)
    tampered["claim_boundary"]["no_causal_attribution"] = False
    with pytest.raises(ReviewerTraceError, match="invariant"):
        validate_trace_artifact(tampered)


def test_published_provenance_rejects_host_absolute_paths() -> None:
    payload = _unavailable_artifact()
    payload["provenance"]["science_source"]["source_repo_root"] = (
        "/Users/example/private-worktree"
    )

    with pytest.raises(ReviewerTraceError, match="invariant"):
        _seal_artifact(payload)


def test_same_commit_branch_and_detached_rebuilds_have_identical_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "publisher"
    checkout.mkdir()
    mode = "branch"
    calls: list[tuple[str, ...]] = []

    def fake_git(repo: Path, *args: str) -> str:
        calls.append(tuple(args))
        if args == ("rev-parse", "--show-toplevel"):
            return str(repo.resolve())
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return "3" * 40
        if args[:2] == ("merge-base", "--is-ancestor"):
            return ""
        if args[:3] == ("ls-files", "--error-unmatch", "--"):
            return args[3]
        if args[:3] == ("diff", "--quiet", "HEAD"):
            return ""
        if args[:2] == ("rev-parse", "--abbrev-ref"):
            return "codex/reviewer-trace-v1" if mode == "branch" else "HEAD"
        if args[0] == "rev-parse" and args[1].startswith("HEAD:"):
            return hashlib.sha1(args[1].encode("utf-8")).hexdigest()
        raise AssertionError(f"unexpected git query: {args}")

    monkeypatch.setattr(trace_module, "_git", fake_git)
    branch_provenance = _verify_publisher_checkout(checkout)
    branch_artifact = _unavailable_artifact(branch_provenance)
    mode = "detached"
    detached_provenance = _verify_publisher_checkout(checkout)
    detached_artifact = _unavailable_artifact(detached_provenance)

    assert branch_provenance == detached_provenance
    assert "branch" not in branch_provenance
    assert "repository_root" not in branch_provenance
    assert "/Users/" not in json.dumps(branch_artifact, sort_keys=True)
    assert json.dumps(branch_artifact, sort_keys=True) == json.dumps(
        detached_artifact, sort_keys=True
    )
    assert not any(
        args[:2] == ("rev-parse", "--abbrev-ref") for args in calls
    )


def test_builder_has_no_provider_or_environment_access_imports() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {"environ", "getenv"}

    assert imported_roots.isdisjoint(
        {
            "anthropic",
            "google",
            "httpx",
            "llm_providers",
            "openai",
            "requests",
        }
    )


def test_cli_routes_only_to_offline_package_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "trace-package"
    trace_path = output_dir / "reviewer_closed_loop_trace.json"
    calls: list[dict[str, Any]] = []

    def fake_builder(**kwargs: Any) -> Path:
        calls.append(kwargs)
        output_dir.mkdir()
        trace_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "integrity": {"content_sha256": "a" * 64},
                    "provenance": {
                        "stage_go": False,
                        "stage_status": "complete-with-no-go",
                    },
                }
            ),
            encoding="utf-8",
        )
        return trace_path

    monkeypatch.setattr(trace_cli, "build_trace_package", fake_builder)
    source = tmp_path / "sealed-source"
    publisher = tmp_path / "publisher"

    assert (
        trace_cli.main(
            [
                "--source-repo-root",
                str(source),
                "--output-dir",
                str(output_dir),
                "--publisher-repo-root",
                str(publisher),
            ]
        )
        == 0
    )
    assert calls == [
        {
            "source_repo_root": source,
            "output_dir": output_dir,
            "publisher_repo_root": publisher,
        }
    ]
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["publication_provider_calls"] == 0
    assert stdout["source_provider_call_scope"] == (
        "historical-observations-read-from-sealed-logs"
    )
    assert stdout["stage_go"] is False


def test_missing_fixed_run_emits_unavailable_without_searching_alternative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    alternative = (
        raw_root
        / "experiment-a"
        / "runs"
        / "attractive-alternative-run"
        / "manifest.json"
    )
    alternative.parent.mkdir(parents=True)
    alternative.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ReviewerTraceUnavailable, match="no fallback"):
        _fixed_run_manifest_path(raw_root, "contract-selected-run")

    def fixed_run_missing(**_kwargs: Any) -> dict[str, Any]:
        _fixed_run_manifest_path(raw_root, "contract-selected-run")
        raise AssertionError("unreachable")

    monkeypatch.setattr(trace_module, "_extract_trace", fixed_run_missing)
    artifact = _build_trace_artifact_verified(
        source=ROOT,
        publisher_root=ROOT,
        source_provenance=_source_provenance(),
        publisher_provenance=_publisher_provenance(),
    )

    assert artifact["status"] == "unavailable"
    assert artifact["trace"] is None
    assert artifact["source_files"] == []
    assert artifact["source_records"] == []
    assert "no fallback" in artifact["failure_reason"]


def test_real_sealed_trace_closes_all_links_and_rebuilds_byte_identically(
    tmp_path: Path,
) -> None:
    source = ROOT.parent / "finevo-pilot-v2-11-5-science"
    if not source.is_dir():
        pytest.skip("local sealed V2.11.5 integration worktree is unavailable")
    source_provenance = _verify_source_checkout(source)
    publisher = _publisher_provenance()

    first = _build_trace_artifact_verified(
        source=source.resolve(),
        publisher_root=ROOT,
        source_provenance=source_provenance,
        publisher_provenance=publisher,
    )
    second = _build_trace_artifact_verified(
        source=source.resolve(),
        publisher_root=ROOT,
        source_provenance=source_provenance,
        publisher_provenance=publisher,
    )

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] == "complete"
    assert first["publication_provider_calls"] == 0
    assert len(first["link_checks"]["checks"]) == 17
    assert all(first["link_checks"]["checks"].values())
    assert len(first["source_files"]) == 17
    assert len(first["source_records"]) == 23
    assert first["provenance"]["stage_status"] == "complete-with-no-go"
    assert first["provenance"]["stage_go"] is False
    assert first["trace"]["memory_update"]["verifier_recomputation"] == {
        "scope_matches": True,
        "condition_matches": True,
        "guidance_compliant": True,
        "guidance_observed_value": 0.86,
        "outcome_metric": "utility_advantage",
        "outcome_observed_value": pytest.approx(-0.17769995756579202),
        "outcome_passed": False,
        "classification": "harmful_compliance",
    }
    rule = first["trace"]["retrieval"]["selected_rules"][0]
    assert rule["candidate_id"] == "cand-134878c05dd7d100237e"
    assert rule["activation_t"] == 4
    assert rule["activation_from_status"] == "provisional"
    assert rule["activation_to_status"] == "active"
    assert rule["retirement_t"] == rule["final_updated_at"] == 9
    assert rule["retirement_from_status"] == "active"
    assert rule["retirement_to_status"] == rule["final_status"] == "retired"
    assert "/Users/" not in json.dumps(first, sort_keys=True)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(first)
    first_path = _write_trace_package(
        payload=first,
        schema_source=SCHEMA_PATH,
        target=tmp_path / "package-a",
    )
    second_path = _write_trace_package(
        payload=second,
        schema_source=SCHEMA_PATH,
        target=tmp_path / "package-b",
    )
    for name in (
        "reviewer_closed_loop_trace.json",
        "reviewer_closed_loop_trace_v1.schema.json",
        "checksums.json",
    ):
        assert (first_path.parent / name).read_bytes() == (
            second_path.parent / name
        ).read_bytes()


def test_real_sealed_trace_rejects_pre_post_loader_receipt_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ROOT.parent / "finevo-pilot-v2-11-5-science"
    if not source.is_dir():
        pytest.skip("local sealed V2.11.5 integration worktree is unavailable")
    original = trace_module._central_loader_receipt
    calls = 0

    def drifting_receipt(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        receipt = dict(original(**kwargs))
        if calls == 2:
            receipt["records_sha256"] = "0" * 64
        return receipt

    monkeypatch.setattr(trace_module, "_central_loader_receipt", drifting_receipt)
    with pytest.raises(ReviewerTraceError, match="changed during extraction"):
        _build_trace_artifact_verified(
            source=source.resolve(),
            publisher_root=ROOT,
            source_provenance=_verify_source_checkout(source),
            publisher_provenance=_publisher_provenance(),
        )


def test_runtime_schema_rejects_nested_provenance_drift() -> None:
    artifact = _unavailable_artifact()
    del artifact["provenance"]["publisher"]["publisher_id"]

    with pytest.raises(ReviewerTraceError, match="/provenance/publisher"):
        _validate_against_runtime_schema(artifact, SCHEMA_PATH.read_bytes())


def test_package_schema_must_match_publisher_head_blob(tmp_path: Path) -> None:
    artifact = _unavailable_artifact()
    artifact["provenance"]["publisher"]["required_tracked_head_blobs"][
        "schemas/reviewer_closed_loop_trace_v1.schema.json"
    ] = "f" * 40
    artifact = _seal_artifact(artifact)

    with pytest.raises(ReviewerTraceError, match="publisher HEAD"):
        _write_trace_package(
            payload=artifact,
            schema_source=SCHEMA_PATH,
            target=tmp_path / "package",
        )


def test_atomic_directory_reservation_prevents_concurrent_overwrite(
    tmp_path: Path,
) -> None:
    artifact = _unavailable_artifact()
    target = tmp_path / "shared-package"

    def write_once() -> str:
        try:
            return str(
                _write_trace_package(
                    payload=artifact,
                    schema_source=SCHEMA_PATH,
                    target=target,
                )
            )
        except ReviewerTraceError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _item: write_once(), range(2)))

    assert outcomes.count("refused") == 1
    assert len([item for item in outcomes if item != "refused"]) == 1
    stored = json.loads(
        (target / "reviewer_closed_loop_trace.json").read_text(encoding="utf-8")
    )
    assert stored["integrity"] == artifact["integrity"]


def test_path_normalization_and_publication_scan_fail_closed(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    bypass = tmp_path / "missing" / ".." / "alias" / "source"

    with pytest.raises(ReviewerTraceError, match="symlink"):
        _absolute_without_symlink(bypass, name="test input")
    with pytest.raises(ReviewerTraceError, match="secret scan"):
        _scan_publication_bytes(b'"outside":"sk-ant-exampleSecret123456"', name="x")
    with pytest.raises(ReviewerTraceError, match="host-path scan"):
        _scan_publication_bytes(b'"outside":"/Users/example/private"', name="x")
    with pytest.raises(ReviewerTraceError, match="host-path scan"):
        _scan_publication_bytes(b'"outside":"/root/private"', name="x")
    with pytest.raises(ReviewerTraceError, match="host-path scan"):
        _scan_publication_bytes(b'"outside":"/opt/private"', name="x")
    with pytest.raises(ReviewerTraceError, match="host-path scan"):
        _scan_publication_bytes(b'"outside":"Z:\\\\arbitrary\\\\path"', name="x")


def test_harmful_compliance_is_recomputed_not_trusted_from_event_label() -> None:
    rule = {
        "context_scope": {"scope_id": "global", "predicates": []},
        "condition": {
            "field": "interest_rate",
            "operator": "==",
            "value": 0.03,
            "tolerance": 0.0,
        },
        "action_guidance": {
            "target": "consumption_fraction",
            "direction": "at_least",
            "threshold": 0.8,
            "tolerance": 0.0,
        },
        "outcome_criterion": {
            "metric": "utility_advantage",
            "operator": ">",
            "value": 0.0,
            "tolerance": 0.0,
        },
    }
    episode = {
        "pre_state": {"interest_rate": 0.03},
        "executed_action": {"consumption_fraction": 0.86},
        "utility_advantage": -0.1,
        "outcome": {},
    }

    assert _recompute_rule_classification(rule, episode)["classification"] == (
        "harmful_compliance"
    )
    tampered = copy.deepcopy(episode)
    tampered["executed_action"]["consumption_fraction"] = 0.2
    assert _recompute_rule_classification(rule, tampered)["classification"] == (
        "alternative_failure"
    )


def test_module_contains_no_duplicate_literal_dict_keys() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    duplicates: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        duplicates.extend(
            (node.lineno, key) for key in sorted(set(keys)) if keys.count(key) > 1
        )

    assert duplicates == []
