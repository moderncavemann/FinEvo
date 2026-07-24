import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from verified_memory.pilot_release_attestation import CommandResult
from verified_memory.scientific_release_attestation import (
    CI_JOB_RECEIPT_SCHEMA_VERSION,
    SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION,
    ScientificReleaseAttestationError,
    adapt_contract_release_requirements,
    build_scientific_contract_binding,
    canonical_contract_sha256,
    canonical_sha256,
    dry_verify_scientific_launch_input,
    prepare_scientific_launch_input,
    resolve_scientific_ci_run_selection,
    scientific_policy_pointer_sets,
    verify_scientific_release_attestation,
)


HEAD = "1" * 40
TAG_OBJECT = "2" * 40
WORKFLOW_BLOB = "3" * 40
CONTRACT_BLOB = "4" * 40
REPOSITORY = "owner/finevo"
RUN_ID = 83109
ATTEMPT = 2
JOB_NAMES = (
    "Python 3.12.7 / ubuntu-24.04",
    "Python 3.12.7 / macos-14",
)
JOB_IDS = (98101, 98102)
INVENTORY_SHA = "5" * 64


def _contract() -> dict:
    value = {
        "schema_version": "finevo-pilot-contract-v2",
        "contract_id": "finevo-pilot-v2",
        "provider_profiles": {
            "gpt": {
                "price_snapshot": {"input_per_million": 1.75}
            },
            "local": {
                "price_snapshot": {"input_per_million": 0.0}
            },
        },
        "parameter_dispatch_policy": {
            "unsupported_parameter_policy": "omit-and-record"
        },
        "task_output_contracts": {"actor": {"max_tokens": 2048}},
        "model_roles": {"gpt": {"role": "primary"}},
        "budgets": {"total_usd": 25.0, "reserve_usd": 1.0},
        "denominator_policy": {"registered_cells_are_itt": True},
        "stop_go": {"core": {"complete_pairs_min": 4}},
        "release_requirements": _requirements(),
        "integrity": {
            "canonicalization": "json-sort-keys-v1",
            "declared_sha256": "",
        },
    }
    value["integrity"]["declared_sha256"] = canonical_contract_sha256(
        value
    )
    return value


def _resolve_pointer(contract: dict, pointer: str):
    value = contract
    for token in pointer.removeprefix("/").split("/"):
        value = value[token.replace("~1", "/").replace("~0", "~")]
    return value


def _policy_binding(contract: dict, pointers: tuple[str, ...]) -> dict:
    return {
        "pointers": list(pointers),
        "sha256": canonical_sha256(
            {
                pointer: _resolve_pointer(contract, pointer)
                for pointer in pointers
            }
        ),
    }


def _requirements(**updates) -> dict:
    value = {
        "remote": "origin",
        "branch": "main",
        "tag": "pilot-v2",
        "workflow_file": ".github/workflows/scientific-ci.yml",
        "workflow_name": "Scientific pilot CI",
        "required_job_names": list(JOB_NAMES),
        "expected_ci": {
            "test_count": 519,
            "test_collection_sha256": "6" * 64,
            "compiled_source_count": 37,
            "compiled_source_inventory_sha256": "7" * 64,
            "sealed_manifest_inventory_sha256": INVENTORY_SHA,
        },
    }
    value.update(updates)
    return value


def _selection(**updates) -> dict:
    value = {
        "run_id": RUN_ID,
        "run_attempt": ATTEMPT,
        "jobs": [
            {"name": name, "database_id": database_id}
            for name, database_id in zip(JOB_NAMES, JOB_IDS, strict=True)
        ],
    }
    value.update(updates)
    return value


def _binding(root: Path, contract: dict, **updates) -> dict:
    contract_path = root / "experiments" / "pilot_v2.yaml"
    contract_bytes = json.dumps(
        contract, sort_keys=True, separators=(",", ":")
    ).encode()
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_bytes(contract_bytes)
    policy_pointers = scientific_policy_pointer_sets(contract)
    value = {
        "contract_path": "experiments/pilot_v2.yaml",
        "contract_file_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "contract_canonical_sha256": canonical_contract_sha256(contract),
        "policies": {
            name: _policy_binding(contract, policy_pointers[name])
            for name in (
                "provider_policy",
                "price_policy",
                "budget_policy",
            )
        },
        "sealed_manifest_paths": [
            "artifacts/verified_runs/example/manifest.json"
        ],
        "sealed_manifest_inventory_sha256": INVENTORY_SHA,
    }
    value.update(updates)
    return value


def _ci_receipt(
    *,
    job_name: str,
    workflow_sha256: str,
    **updates,
) -> dict:
    payload = {
        "schema_version": CI_JOB_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "repository": REPOSITORY,
        "head_sha": HEAD,
        "run_id": RUN_ID,
        "run_attempt": ATTEMPT,
        "job_name": job_name,
        "job_key": "verify",
        "runner_os": (
            "Linux" if "ubuntu" in job_name else "macOS"
        ),
        "workflow_name": "Scientific pilot CI",
        "workflow_file": ".github/workflows/scientific-ci.yml",
        "workflow_ref": (
            f"{REPOSITORY}/.github/workflows/scientific-ci.yml@refs/heads/main"
        ),
        "workflow_source_sha": HEAD,
        "workflow_file_sha256": workflow_sha256,
        "workflow_blob_oid": WORKFLOW_BLOB,
        "test_count": 519,
        "test_collection_sha256": "6" * 64,
        "skipped_test_count": 0,
        "compiled_source_count": 37,
        "compiled_source_inventory_sha256": "7" * 64,
        "sealed_manifest_count": 1,
        "sealed_manifest_inventory_sha256": INVENTORY_SHA,
    }
    payload.update(updates)
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


class FakeRunner:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, argv, cwd):
        argv = tuple(argv)
        self.calls.append((argv, cwd))
        if argv[:3] == ("git", "rev-parse", "--verify"):
            expression = argv[3]
            keys = {
                "HEAD^{commit}": "head",
                "refs/tags/pilot-v2^{tag}": "tag_object",
                "refs/tags/pilot-v2^{commit}": "tag_commit",
                "HEAD:.github/workflows/scientific-ci.yml": "workflow_blob",
                "HEAD:experiments/pilot_v2.yaml": "contract_blob",
            }
            key = keys[expression]
        elif argv[:2] == ("git", "status"):
            key = "status"
        elif argv[:3] == ("git", "remote", "get-url"):
            key = "origin"
        elif argv[:2] == ("git", "ls-remote"):
            key = "remote"
        elif argv[:2] == ("git", "check-ignore"):
            key = "ignored"
        elif argv[:3] == ("gh", "api", "--method"):
            key = "jobs" if argv[-1].endswith("jobs?per_page=100") else "run"
        elif argv[:3] == ("gh", "run", "view"):
            job_id = int(argv[argv.index("--job") + 1])
            key = f"log-{job_id}"
        else:  # pragma: no cover - guards fixture drift
            raise AssertionError(argv)
        return self.responses[key]


def _responses(workflow_sha256: str) -> dict:
    run_url = f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}"
    jobs = [
        {
            "id": database_id,
            "name": name,
            "status": "completed",
            "conclusion": "success",
            "run_attempt": ATTEMPT,
            "html_url": f"{run_url}/job/{database_id}",
        }
        for name, database_id in zip(JOB_NAMES, JOB_IDS, strict=True)
    ]
    result = {
        "head": CommandResult(f"{HEAD}\n".encode()),
        "status": CommandResult(b""),
        "tag_object": CommandResult(f"{TAG_OBJECT}\n".encode()),
        "tag_commit": CommandResult(f"{HEAD}\n".encode()),
        "workflow_blob": CommandResult(f"{WORKFLOW_BLOB}\n".encode()),
        "contract_blob": CommandResult(f"{CONTRACT_BLOB}\n".encode()),
        "ignored": CommandResult(b""),
        "origin": CommandResult(
            f"https://github.com/{REPOSITORY}.git\n".encode()
        ),
        "remote": CommandResult(
            (
                f"{HEAD}\trefs/heads/main\n"
                f"{TAG_OBJECT}\trefs/tags/pilot-v2\n"
                f"{HEAD}\trefs/tags/pilot-v2^{{}}\n"
            ).encode()
        ),
        "run": CommandResult(
            json.dumps(
                {
                    "id": RUN_ID,
                    "run_attempt": ATTEMPT,
                    "head_sha": HEAD,
                    "head_branch": "main",
                    "status": "completed",
                    "conclusion": "success",
                    "name": "Scientific pilot CI",
                    "path": ".github/workflows/scientific-ci.yml",
                    "event": "push",
                    "html_url": run_url,
                }
            ).encode()
        ),
        "jobs": CommandResult(
            json.dumps({"total_count": 2, "jobs": jobs}).encode()
        ),
    }
    for name, database_id in zip(JOB_NAMES, JOB_IDS, strict=True):
        receipt = _ci_receipt(
            job_name=name, workflow_sha256=workflow_sha256
        )
        result[f"log-{database_id}"] = CommandResult(
            (
                "safe prefix from gh\n"
                "step\t"
                "FINEVO_CI_RELEASE_RECEIPT_JSON="
                + json.dumps(receipt, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
        )
    return result


def _attest(
    tmp_path: Path,
    monkeypatch,
    *,
    responses=None,
    binding=None,
    requirements=None,
    selection=None,
    compatibility=None,
):
    workflow = tmp_path / ".github" / "workflows" / "scientific-ci.yml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text("name: Scientific pilot CI\n", encoding="utf-8")
    contract = _contract()
    binding_value = binding or _binding(tmp_path, contract)
    workflow_sha = hashlib.sha256(workflow.read_bytes()).hexdigest()
    runner = FakeRunner(responses or _responses(workflow_sha))
    monkeypatch.setattr(
        "verified_memory.scientific_release_attestation."
        "sealed_manifest_inventory",
        lambda root, paths: (
            (
                {
                    "path": paths[0],
                    "manifest_sha256": "8" * 64,
                    "artifact_count": 11,
                },
            ),
            INVENTORY_SHA,
        ),
    )
    attestation = verify_scientific_release_attestation(
        tmp_path,
        release_requirements=requirements or _requirements(),
        ci_run_selection=selection or _selection(),
        contract_binding=binding_value,
        release_compatibility=compatibility,
        runner=runner,
    )
    return attestation, runner


def test_v2_binds_contract_driven_release_exact_ci_and_policy_hashes(
    tmp_path: Path, monkeypatch
):
    attestation, runner = _attest(tmp_path, monkeypatch)
    payload = attestation.to_dict()

    assert (
        payload["schema_version"]
        == SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION
    )
    assert payload["head_commit"] == HEAD
    assert payload["local_tag"]["name"] == "pilot-v2"
    assert payload["remote"]["branch_commit"] == HEAD
    assert payload["ci_run_selection"]["run_id"] == RUN_ID
    assert payload["ci_run_selection"]["run_attempt"] == ATTEMPT
    assert payload["github_actions"]["run"]["database_id"] == RUN_ID
    assert payload["github_actions"]["run"]["attempt"] == ATTEMPT
    assert [
        row["database_id"] for row in payload["github_actions"]["jobs"]
    ] == list(JOB_IDS)
    assert payload["github_actions"]["ci_measurements"] == {
        "test_count": 519,
        "test_collection_sha256": "6" * 64,
        "compiled_source_count": 37,
        "compiled_source_inventory_sha256": "7" * 64,
    }
    assert payload["workflow"]["blob_oid"] == WORKFLOW_BLOB
    assert payload["contract"]["blob_oid"] == CONTRACT_BLOB
    assert payload["contract"]["provider_policy_sha256"]
    assert (
        payload["sealed_manifest_inventory"]["inventory_sha256"]
        == INVENTORY_SHA
    )
    unsigned = dict(payload)
    observed = unsigned.pop("attestation_sha256")
    assert observed == canonical_sha256(unsigned)
    attestation.verify_hash()
    with pytest.raises(dataclasses.FrozenInstanceError):
        attestation.attestation_sha256 = "0" * 64

    api_call = next(
        argv for argv, _ in runner.calls if argv[:3] == ("gh", "api", "--method")
    )
    assert f"/{RUN_ID}/attempts/{ATTEMPT}" in api_call[-1]
    log_calls = [
        argv for argv, _ in runner.calls if argv[:3] == ("gh", "run", "view")
    ]
    assert {int(argv[argv.index("--job") + 1]) for argv in log_calls} == set(
        JOB_IDS
    )
    assert all(
        argv[argv.index("--attempt") + 1] == str(ATTEMPT)
        for argv in log_calls
    )


def test_scientific_release_rejects_non_push_workflow_run(
    tmp_path: Path, monkeypatch
) -> None:
    workflow = tmp_path / ".github" / "workflows" / "scientific-ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: Scientific pilot CI\n", encoding="utf-8")
    workflow_sha = hashlib.sha256(workflow.read_bytes()).hexdigest()
    responses = _responses(workflow_sha)
    run = json.loads(responses["run"].stdout)
    run["event"] = "workflow_dispatch"
    responses["run"] = CommandResult(json.dumps(run).encode())

    with pytest.raises(ScientificReleaseAttestationError, match="event"):
        _attest(tmp_path, monkeypatch, responses=responses)


def test_ci_receipt_hash_and_cross_os_measurements_fail_closed(
    tmp_path: Path, monkeypatch
):
    workflow = tmp_path / ".github" / "workflows" / "scientific-ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: Scientific pilot CI\n", encoding="utf-8")
    workflow_sha = hashlib.sha256(workflow.read_bytes()).hexdigest()
    responses = _responses(workflow_sha)
    log = json.loads(
        responses[f"log-{JOB_IDS[1]}"].stdout.decode().split(
            "FINEVO_CI_RELEASE_RECEIPT_JSON=", 1
        )[1]
    )
    log["test_count"] = 518
    body = (
        "FINEVO_CI_RELEASE_RECEIPT_JSON="
        + json.dumps(log, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    responses[f"log-{JOB_IDS[1]}"] = CommandResult(body.encode())
    with pytest.raises(
        ScientificReleaseAttestationError, match="self-hash mismatch"
    ):
        _attest(tmp_path, monkeypatch, responses=responses)

    log.pop("receipt_sha256")
    log["receipt_sha256"] = canonical_sha256(log)
    responses[f"log-{JOB_IDS[1]}"] = CommandResult(
        (
            "FINEVO_CI_RELEASE_RECEIPT_JSON="
            + json.dumps(log, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
    )
    with pytest.raises(
        ScientificReleaseAttestationError, match="static contract"
    ):
        _attest(tmp_path, monkeypatch, responses=responses)


def test_contract_file_policy_and_exact_job_identity_fail_closed(
    tmp_path: Path, monkeypatch
):
    workflow = tmp_path / ".github" / "workflows" / "scientific-ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: Scientific pilot CI\n", encoding="utf-8")
    workflow_sha = hashlib.sha256(workflow.read_bytes()).hexdigest()
    responses = _responses(workflow_sha)
    responses["jobs"] = CommandResult(
        json.dumps(
            {
                "total_count": 2,
                "jobs": [
                    {
                        "id": JOB_IDS[0],
                        "name": "wrong job",
                        "status": "completed",
                        "conclusion": "success",
                        "run_attempt": ATTEMPT,
                        "html_url": (
                            f"https://github.com/{REPOSITORY}/actions/runs/"
                            f"{RUN_ID}/job/{JOB_IDS[0]}"
                        ),
                    },
                    {
                        "id": JOB_IDS[1],
                        "name": JOB_NAMES[1],
                        "status": "completed",
                        "conclusion": "success",
                        "run_attempt": ATTEMPT,
                        "html_url": (
                            f"https://github.com/{REPOSITORY}/actions/runs/"
                            f"{RUN_ID}/job/{JOB_IDS[1]}"
                        ),
                    },
                ],
            }
        ).encode()
    )
    with pytest.raises(ScientificReleaseAttestationError, match="name mismatches"):
        _attest(tmp_path, monkeypatch, responses=responses)

    other = tmp_path / "other"
    other.mkdir()
    contract = _contract()
    binding = _binding(other, contract)
    binding["policies"]["budget_policy"]["sha256"] = "0" * 64
    with pytest.raises(
        ScientificReleaseAttestationError, match="budget_policy SHA-256"
    ):
        _attest(other, monkeypatch, binding=binding)


def test_policy_pointer_coverage_cannot_be_reduced_with_a_valid_hash(
    tmp_path: Path, monkeypatch
):
    contract = _contract()
    reductions = {
        "provider_policy": ("/parameter_dispatch_policy",),
        "price_policy": (
            "/provider_profiles/gpt/price_snapshot",
        ),
        "budget_policy": ("/budgets",),
    }
    for policy_name, reduced in reductions.items():
        binding = _binding(tmp_path, contract)
        binding["policies"][policy_name] = _policy_binding(
            contract, reduced
        )
        with pytest.raises(
            ScientificReleaseAttestationError,
            match=(
                f"{policy_name} pointers do not match the mandatory "
                "scientific coverage set"
            ),
        ):
            _attest(tmp_path, monkeypatch, binding=binding)


def test_prepare_launch_resolves_job_ids_in_contract_order_and_dry_verifies(
    tmp_path: Path, monkeypatch
):
    workflow = tmp_path / ".github" / "workflows" / "scientific-ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: Scientific pilot CI\n", encoding="utf-8")
    contract = _contract()
    _binding(tmp_path, contract)
    workflow_sha = hashlib.sha256(workflow.read_bytes()).hexdigest()
    responses = _responses(workflow_sha)
    jobs_payload = json.loads(responses["jobs"].stdout)
    jobs_payload["jobs"].reverse()
    responses["jobs"] = CommandResult(json.dumps(jobs_payload).encode())
    runner = FakeRunner(responses)
    monkeypatch.setattr(
        "verified_memory.scientific_release_attestation."
        "discover_scientific_manifest_paths",
        lambda root, runner=None: (
            "artifacts/verified_runs/example/manifest.json",
        ),
    )
    monkeypatch.setattr(
        "verified_memory.scientific_release_attestation."
        "sealed_manifest_inventory",
        lambda root, paths: (
            (
                {
                    "path": paths[0],
                    "manifest_sha256": "8" * 64,
                    "artifact_count": 11,
                },
            ),
            INVENTORY_SHA,
        ),
    )

    selection = resolve_scientific_ci_run_selection(
        tmp_path,
        release_requirements=_requirements(),
        run_id=RUN_ID,
        run_attempt=ATTEMPT,
        runner=runner,
    )
    assert [row["name"] for row in selection["jobs"]] == list(JOB_NAMES)
    assert [row["database_id"] for row in selection["jobs"]] == list(
        JOB_IDS
    )

    binding = build_scientific_contract_binding(
        tmp_path,
        contract_path="experiments/pilot_v2.yaml",
        runner=runner,
    )
    assert binding["policies"] == {
        name: _policy_binding(
            contract, scientific_policy_pointer_sets(contract)[name]
        )
        for name in (
            "provider_policy",
            "price_policy",
            "budget_policy",
        )
    }

    receipt = prepare_scientific_launch_input(
        tmp_path,
        contract_path="experiments/pilot_v2.yaml",
        run_id=RUN_ID,
        run_attempt=ATTEMPT,
        runner=runner,
    )
    output = tmp_path / receipt["output"]
    assert receipt["status"] == "pass"
    assert receipt["provider_calls"] == 0
    assert output.is_file()
    launch = json.loads(output.read_text(encoding="utf-8"))
    assert launch["ci_run_selection"]["jobs"] == _selection()["jobs"]
    unsigned = dict(launch)
    assert unsigned.pop("launch_input_sha256") == canonical_sha256(unsigned)
    verified = dry_verify_scientific_launch_input(
        tmp_path,
        contract_path="experiments/pilot_v2.yaml",
        launch_input=output,
        runner=runner,
    )
    assert verified.attestation_sha256


def test_static_release_requirements_must_be_embedded_in_contract(
    tmp_path: Path, monkeypatch
):
    with pytest.raises(
        ScientificReleaseAttestationError, match="differ from the scientific contract"
    ):
        _attest(
            tmp_path,
            monkeypatch,
            requirements=_requirements(workflow_name="Different CI"),
        )


@pytest.mark.parametrize(
    "requirements",
    [
        _requirements(required_job_names=[]),
        _requirements(
            required_job_names=[JOB_NAMES[0], JOB_NAMES[0]]
        ),
        _requirements(tag="../unsafe"),
        _requirements(workflow_file="ci.yml"),
    ],
)
def test_release_requirements_are_exact_and_safe(
    tmp_path: Path, requirements
):
    with pytest.raises(ScientificReleaseAttestationError):
        verify_scientific_release_attestation(
            tmp_path,
            release_requirements=requirements,
            ci_run_selection=_selection(),
            contract_binding={},
            runner=lambda argv, cwd: pytest.fail("must fail before commands"),
        )


def test_current_contract_release_shape_adapts_only_with_explicit_static_fill():
    current = {
        "required_branch": "main",
        "required_annotated_tag": "pilot-v2",
        "required_workflow": "scientific-ci.yml",
        "required_platforms": ["ubuntu", "macos"],
        "peeled_commit": None,
        "workflow_run_id": None,
        "ubuntu_job_id": None,
        "macos_job_id": None,
        "test_count": None,
        "manifest_sha256": None,
        "contract_sha256": None,
        "catalog_sha256": None,
        "price_snapshot_sha256": None,
        "budget_sha256": None,
    }
    compatibility = {
        "remote": "origin",
        "workflow_name": "Scientific pilot CI",
        "required_job_names": list(JOB_NAMES),
        "expected_ci": {
            "test_count": 519,
            "test_collection_sha256": "6" * 64,
            "compiled_source_count": 37,
            "compiled_source_inventory_sha256": "7" * 64,
            "sealed_manifest_inventory_sha256": INVENTORY_SHA,
        },
    }
    normalized = adapt_contract_release_requirements(
        current, compatibility=compatibility
    )
    assert normalized == _requirements()

    with pytest.raises(
        ScientificReleaseAttestationError, match="not scientifically frozen"
    ):
        adapt_contract_release_requirements(current)
    with pytest.raises(
        ScientificReleaseAttestationError,
        match="dynamic CI IDs must remain outside",
    ):
        adapt_contract_release_requirements(
            {**current, "workflow_run_id": RUN_ID},
            compatibility=compatibility,
        )
