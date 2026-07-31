import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import verified_memory.ci_release_receipt as ci_receipts
from verified_memory.ci_release_receipt import (
    CIReleaseReceiptError,
    PUBLICATION_CONSUMER_CI_AUTHORITY_RELATIVE,
    PUBLICATION_CONSUMER_CI_AUTHORITY_SCHEMA_VERSION,
    PUBLICATION_CONSUMER_CI_RECEIPT_SCHEMA_VERSION,
    SCIENTIFIC_SOURCE_MANIFEST_ANCHORS,
    SCIENTIFIC_SOURCE_MANIFEST_INVENTORY_SCHEMA_VERSION,
    V2115_SCIENCE_TAG_OBJECT,
    build_ci_job_receipt,
    build_collection_inventory,
    build_publication_consumer_ci_receipt,
    build_scientific_source_manifest_inventory,
    build_source_inventory,
    parse_junit_summary,
    load_publication_consumer_ci_authority,
    verify_contract_ci_receipt,
    verify_expected_ci_matches_receipt,
    verify_publication_consumer_ci_receipt,
)
from verified_memory.scientific_release_attestation import (
    CI_JOB_RECEIPT_SCHEMA_VERSION,
    canonical_sha256,
)


HEAD = "1" * 40
WORKFLOW_BLOB = "2" * 40
INVENTORY_SHA = "3" * 64
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CI = {
    "test_count": 2,
    "test_collection_sha256": "4" * 64,
    "compiled_source_count": 3,
    "compiled_source_inventory_sha256": "5" * 64,
    "sealed_manifest_inventory_sha256": "6" * 64,
}
V2115_SCIENCE_COMMIT = "2351ac2283f9fedb9dce70067174020be56ed9cc"
V2115_CONSUMER_HEAD = "7" * 40


def _publication_authority(
    tmp_path: Path,
    *,
    science_commit: str = V2115_SCIENCE_COMMIT,
    extra_top_level: bool = False,
) -> Path:
    value = {
        "schema_version": PUBLICATION_CONSUMER_CI_AUTHORITY_SCHEMA_VERSION,
        "status": "frozen",
        "authority_id": "finevo-pilot-v2.11.5-evidence-consumer-ci",
        "science_anchor": {
            "contract_path": "experiments/pilot_v2_11_5.yaml",
            "contract_id": "finevo-pilot-v2.11.5",
            "contract_sha256": (
                "e1ecdec43e3f7a7b9a3d0977e2522d95861e826fc68781377d7eaceeb5e6e2ef"
            ),
            "git_tag": "pilot-v2.11.5-science",
            "git_tag_object": V2115_SCIENCE_TAG_OBJECT,
            "git_commit": science_commit,
        },
        "scope": {
            "purpose": "publication-consumer-ci",
            "scientific_evidence": False,
            "provider_calls": 0,
            "science_dispatch_authority": False,
        },
        "expected_ci": dict(EXPECTED_CI),
        "integrity": {"canonicalization": "json-sort-keys-utf8-v1"},
    }
    if extra_top_level:
        value["unexpected"] = "self-rehashed but invalid"
    value["integrity"]["content_sha256"] = canonical_sha256(value)
    path = tmp_path / PUBLICATION_CONSUMER_CI_AUTHORITY_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    workflow = tmp_path / ".github/workflows/verified-memory-ci.yml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text("name: Verified memory CI\n", encoding="utf-8")
    return path


def _publication_ci_job(
    tmp_path: Path,
    *,
    head: str = V2115_CONSUMER_HEAD,
    runner_os: str = "Linux",
) -> dict[str, object]:
    workflow = tmp_path / ".github/workflows/verified-memory-ci.yml"
    payload: dict[str, object] = {
        "schema_version": CI_JOB_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "repository": "moderncavemann/FinEvo",
        "head_sha": head,
        "run_id": 19821,
        "run_attempt": 3,
        "job_name": (
            "Python 3.12.7 / ubuntu-24.04"
            if runner_os == "Linux"
            else "Python 3.12.7 / macos-14"
        ),
        "job_key": "verify",
        "runner_os": runner_os,
        "workflow_name": "Verified memory CI",
        "workflow_file": ".github/workflows/verified-memory-ci.yml",
        "workflow_ref": (
            "moderncavemann/FinEvo/.github/workflows/"
            "verified-memory-ci.yml@refs/heads/main"
        ),
        "workflow_source_sha": head,
        "workflow_file_sha256": hashlib.sha256(workflow.read_bytes()).hexdigest(),
        "workflow_blob_oid": WORKFLOW_BLOB,
        "test_count": EXPECTED_CI["test_count"],
        "test_collection_sha256": EXPECTED_CI["test_collection_sha256"],
        "skipped_test_count": 0,
        "compiled_source_count": EXPECTED_CI["compiled_source_count"],
        "compiled_source_inventory_sha256": EXPECTED_CI[
            "compiled_source_inventory_sha256"
        ],
        "sealed_manifest_count": 6,
        "sealed_manifest_inventory_sha256": EXPECTED_CI[
            "sealed_manifest_inventory_sha256"
        ],
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def _mock_publication_authority_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tag_object: str = V2115_SCIENCE_TAG_OBJECT,
) -> None:
    contract = SimpleNamespace(
        status="frozen",
        contract_id="finevo-pilot-v2.11.5",
        canonical_hash=(
            "e1ecdec43e3f7a7b9a3d0977e2522d95861e826fc68781377d7eaceeb5e6e2ef"
        ),
        release_requirements=SimpleNamespace(tag="pilot-v2.11.5-science"),
    )
    monkeypatch.setattr(ci_receipts, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        ci_receipts,
        "discover_tracked_files",
        lambda _root, patterns: tuple(patterns),
    )
    monkeypatch.setattr(ci_receipts, "_git_success", lambda *_args: None)

    def git_line(_root: Path, argv: tuple[str, ...]) -> str:
        if argv[1:4] == (
            "cat-file",
            "-t",
            "refs/tags/pilot-v2.11.5-science",
        ):
            return "tag"
        if "refs/tags/pilot-v2.11.5-science^{object}" in argv:
            return tag_object
        if "refs/tags/pilot-v2.11.5-science^{commit}" in argv:
            return V2115_SCIENCE_COMMIT
        if argv[-1] == "HEAD^{commit}":
            return V2115_CONSUMER_HEAD
        if argv[-1] == "HEAD:.github/workflows/verified-memory-ci.yml":
            return WORKFLOW_BLOB
        raise AssertionError(argv)

    monkeypatch.setattr(ci_receipts, "_git_line", git_line)


def _environment() -> dict[str, str]:
    return {
        "GITHUB_REPOSITORY": "owner/finevo",
        "GITHUB_SHA": HEAD,
        "GITHUB_RUN_ID": "19821",
        "GITHUB_RUN_ATTEMPT": "3",
        "GITHUB_JOB": "verify",
        "GITHUB_WORKFLOW": "Verified memory CI",
        "GITHUB_WORKFLOW_REF": (
            "owner/finevo/.github/workflows/" "verified-memory-ci.yml@refs/heads/main"
        ),
        "GITHUB_WORKFLOW_SHA": HEAD,
        "RUNNER_OS": "Linux",
        "FINEVO_CI_JOB_NAME": "Python 3.12.7 / ubuntu-24.04",
    }


def _junit(tmp_path: Path, body: str) -> Path:
    output = tmp_path / "results.xml"
    output.write_text(body, encoding="utf-8")
    return output


def _source_manifest_fixture(
    tmp_path: Path, *, tracked: bool = True
) -> tuple[dict[str, str], Path]:
    payload = {
        "schema_version": "finevo-test-source-manifest-v1",
        "authority": {"effect_blind": True},
        "integrity": {"canonicalization": "json-sort-keys-utf8-v1"},
    }
    payload["integrity"]["content_sha256"] = canonical_sha256(payload)
    raw = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    relative = "experiments/test_source_manifest.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True, capture_output=True)
    if tracked:
        subprocess.run(
            ("git", "add", "--", relative),
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    return (
        {
            "path": relative,
            "schema_version": payload["schema_version"],
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": payload["integrity"]["content_sha256"],
        },
        path,
    )


def test_collection_and_source_inventory_are_order_sensitive_and_hashed():
    collection = build_collection_inventory(
        ("tests/test_a.py::test_one", "tests/test_b.py::test_two")
    )
    assert collection["test_count"] == 2
    assert collection["test_collection_sha256"] == canonical_sha256(
        ["tests/test_a.py::test_one", "tests/test_b.py::test_two"]
    )
    sources = build_source_inventory(("run_pilot.py", "verified_memory/runner.py"))
    assert sources["compiled_source_count"] == 2
    assert sources["compiled_source_inventory_sha256"] == canonical_sha256(
        ["run_pilot.py", "verified_memory/runner.py"]
    )
    with pytest.raises(CIReleaseReceiptError, match="duplicate"):
        build_collection_inventory(("same::test", "same::test"))
    with pytest.raises(CIReleaseReceiptError, match="sorted"):
        build_source_inventory(("z.py", "a.py"))


def test_scientific_source_manifest_inventory_is_separate_and_hash_bound(
    tmp_path: Path,
):
    anchor, path = _source_manifest_fixture(tmp_path)
    inventory = build_scientific_source_manifest_inventory(tmp_path, anchors=(anchor,))

    assert inventory == {
        "schema_version": SCIENTIFIC_SOURCE_MANIFEST_INVENTORY_SCHEMA_VERSION,
        "source_manifest_count": 1,
        "source_manifests": [anchor],
        "source_manifest_inventory_sha256": canonical_sha256([anchor]),
    }

    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(CIReleaseReceiptError, match="file hash drifted"):
        build_scientific_source_manifest_inventory(tmp_path, anchors=(anchor,))


def test_scientific_source_manifest_inventory_requires_git_tracking(
    tmp_path: Path,
):
    anchor, _ = _source_manifest_fixture(tmp_path, tracked=False)
    with pytest.raises(CIReleaseReceiptError, match="exactly tracked"):
        build_scientific_source_manifest_inventory(tmp_path, anchors=(anchor,))


def test_scientific_source_manifest_rejects_reanchored_bad_content_seal(
    tmp_path: Path,
):
    anchor, path = _source_manifest_fixture(tmp_path)
    value = json.loads(path.read_bytes())
    value["authority"]["effect_blind"] = False
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(raw)
    anchor["file_sha256"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(CIReleaseReceiptError, match="content seal drifted"):
        build_scientific_source_manifest_inventory(tmp_path, anchors=(anchor,))


def test_scientific_source_manifest_rejects_duplicate_json_key(
    tmp_path: Path,
):
    anchor, path = _source_manifest_fixture(tmp_path)
    raw = (
        b'{"schema_version":"finevo-test-source-manifest-v1",'
        b'"schema_version":"finevo-test-source-manifest-v1"}'
    )
    path.write_bytes(raw)
    anchor["file_sha256"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(CIReleaseReceiptError, match="duplicate JSON key"):
        build_scientific_source_manifest_inventory(tmp_path, anchors=(anchor,))


def test_scientific_source_manifest_rejects_symlink_path(tmp_path: Path):
    anchor, path = _source_manifest_fixture(tmp_path)
    raw = path.read_bytes()
    target = tmp_path / "source-manifest-target.json"
    target.write_bytes(raw)
    path.unlink()
    path.symlink_to(target)
    subprocess.run(
        ("git", "add", "--", anchor["path"]),
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    with pytest.raises(CIReleaseReceiptError, match="contains a symlink"):
        build_scientific_source_manifest_inventory(tmp_path, anchors=(anchor,))


def test_scientific_source_manifest_anchors_match_release_bytes():
    assert [anchor["path"] for anchor in SCIENTIFIC_SOURCE_MANIFEST_ANCHORS] == [
        "experiments/pilot_v2_11_3_source_manifest.json",
        "experiments/pilot_v2_11_4_source_manifest.json",
        "experiments/pilot_v2_11_5_source_manifest.json",
    ]
    for anchor in SCIENTIFIC_SOURCE_MANIFEST_ANCHORS:
        raw = (ROOT / anchor["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == anchor["file_sha256"]
        value = json.loads(raw)
        assert value["schema_version"] == anchor["schema_version"]
        assert value["integrity"]["content_sha256"] == anchor["content_sha256"]


def test_verify_source_manifests_cli_smoke(tmp_path: Path):
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True, capture_output=True)
    for anchor in SCIENTIFIC_SOURCE_MANIFEST_ANCHORS:
        source = ROOT / anchor["path"]
        target = tmp_path / anchor["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        subprocess.run(
            ("git", "add", "--", anchor["path"]),
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    output = tmp_path / "source-manifest-inventory.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "verified_memory.ci_release_receipt",
            "verify-source-manifests",
            "--output",
            str(output),
        ),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    inventory = json.loads(output.read_text(encoding="utf-8"))
    assert inventory["source_manifest_count"] == len(
        SCIENTIFIC_SOURCE_MANIFEST_ANCHORS
    )
    assert inventory["source_manifests"] == list(SCIENTIFIC_SOURCE_MANIFEST_ANCHORS)


def test_junit_summary_counts_cases_and_rejects_failures(tmp_path: Path):
    summary = parse_junit_summary(
        _junit(
            tmp_path,
            (
                '<testsuites tests="2"><testsuite tests="2">'
                '<testcase classname="a" name="one"/>'
                '<testcase classname="b" name="two"><skipped/></testcase>'
                "</testsuite></testsuites>"
            ),
        )
    )
    assert summary == {
        "executed_test_count": 2,
        "failure_count": 0,
        "error_count": 0,
        "skipped_count": 1,
    }
    with pytest.raises(CIReleaseReceiptError, match="failed/error"):
        parse_junit_summary(
            _junit(
                tmp_path,
                (
                    '<testsuite tests="1"><testcase name="bad">'
                    "<failure/></testcase></testsuite>"
                ),
            )
        )


def test_ci_job_receipt_binds_workflow_tests_sources_and_manifests(
    tmp_path: Path, monkeypatch
):
    workflow = tmp_path / ".github" / "workflows" / "verified-memory-ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: Verified memory CI\n", encoding="utf-8")
    monkeypatch.setattr(
        "verified_memory.ci_release_receipt._git_line",
        lambda root, argv: (HEAD if argv[-1] == "HEAD^{commit}" else WORKFLOW_BLOB),
    )
    monkeypatch.setattr(
        "verified_memory.ci_release_receipt.sealed_manifest_inventory",
        lambda root, paths: (
            (
                {
                    "path": paths[0],
                    "manifest_sha256": "4" * 64,
                    "artifact_count": 7,
                },
            ),
            INVENTORY_SHA,
        ),
    )
    receipt = build_ci_job_receipt(
        tmp_path,
        collection_inventory=build_collection_inventory(
            ("tests/test_a.py::test_one", "tests/test_b.py::test_two")
        ),
        source_inventory=build_source_inventory(
            ("run_pilot.py", "verified_memory/runner.py")
        ),
        junit_summary={
            "executed_test_count": 2,
            "failure_count": 0,
            "error_count": 0,
            "skipped_count": 0,
        },
        environment=_environment(),
        manifest_paths=("artifacts/run/manifest.json",),
    )

    assert receipt["schema_version"] == CI_JOB_RECEIPT_SCHEMA_VERSION
    assert receipt["run_id"] == 19821
    assert receipt["run_attempt"] == 3
    assert receipt["head_sha"] == HEAD
    assert receipt["workflow_blob_oid"] == WORKFLOW_BLOB
    assert (
        receipt["workflow_file_sha256"]
        == hashlib.sha256(workflow.read_bytes()).hexdigest()
    )
    assert receipt["test_count"] == 2
    assert receipt["compiled_source_count"] == 2
    assert receipt["sealed_manifest_inventory_sha256"] == INVENTORY_SHA
    unsigned = dict(receipt)
    observed = unsigned.pop("receipt_sha256")
    assert observed == canonical_sha256(unsigned)


def test_ci_receipt_fails_closed_on_count_or_workflow_sha_drift(
    tmp_path: Path, monkeypatch
):
    workflow = tmp_path / ".github" / "workflows" / "verified-memory-ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: Verified memory CI\n", encoding="utf-8")
    monkeypatch.setattr(
        "verified_memory.ci_release_receipt._git_line",
        lambda root, argv: (HEAD if argv[-1] == "HEAD^{commit}" else WORKFLOW_BLOB),
    )
    inputs = {
        "repo_root": tmp_path,
        "collection_inventory": build_collection_inventory(("a::one",)),
        "source_inventory": build_source_inventory(("run_pilot.py",)),
        "junit_summary": {
            "executed_test_count": 2,
            "failure_count": 0,
            "error_count": 0,
            "skipped_count": 0,
        },
        "environment": _environment(),
        "manifest_paths": ("artifacts/run/manifest.json",),
    }
    with pytest.raises(CIReleaseReceiptError, match="executed test count"):
        build_ci_job_receipt(**inputs)

    inputs["junit_summary"]["executed_test_count"] = 1
    inputs["environment"] = {
        **_environment(),
        "GITHUB_WORKFLOW_SHA": "9" * 40,
    }
    with pytest.raises(CIReleaseReceiptError, match="workflow source SHA"):
        build_ci_job_receipt(**inputs)


def test_frozen_expected_ci_matches_exact_receipt_inventory() -> None:
    assert verify_expected_ci_matches_receipt(EXPECTED_CI, EXPECTED_CI) == EXPECTED_CI


@pytest.mark.parametrize("field", sorted(EXPECTED_CI))
def test_frozen_expected_ci_rejects_every_inventory_field_drift(field: str) -> None:
    mutated = dict(EXPECTED_CI)
    mutated[field] = (
        int(mutated[field]) + 1
        if field in {"test_count", "compiled_source_count"}
        else "9" * 64
    )

    with pytest.raises(CIReleaseReceiptError, match=field):
        verify_expected_ci_matches_receipt(mutated, EXPECTED_CI)


def test_frozen_expected_ci_rejects_schema_and_null_values() -> None:
    extra = {**EXPECTED_CI, "unexpected": "field"}
    with pytest.raises(CIReleaseReceiptError, match="fields differ"):
        verify_expected_ci_matches_receipt(extra, EXPECTED_CI)

    missing = dict(EXPECTED_CI)
    missing["test_collection_sha256"] = None
    with pytest.raises(CIReleaseReceiptError, match="test_collection_sha256"):
        verify_expected_ci_matches_receipt(missing, EXPECTED_CI)


def test_ci_contract_gate_requires_frozen_contract_and_exact_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path = tmp_path / "pilot.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    contract = SimpleNamespace(
        status="frozen",
        release_requirements=SimpleNamespace(expected_ci=dict(EXPECTED_CI)),
        contract_id="finevo-test-contract",
        canonical_hash="7" * 64,
    )
    monkeypatch.setattr(
        "verified_memory.ci_release_receipt.load_pilot_contract",
        lambda path: contract,
    )

    assert verify_contract_ci_receipt(contract_path, EXPECTED_CI) == {
        "contract_id": "finevo-test-contract",
        "contract_sha256": "7" * 64,
        "status": "pass",
    }

    contract.status = "draft"
    with pytest.raises(CIReleaseReceiptError, match="frozen release contract"):
        verify_contract_ci_receipt(contract_path, EXPECTED_CI)


def test_publication_consumer_authority_binds_science_and_non_scientific_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path = _publication_authority(tmp_path)
    _mock_publication_authority_dependencies(monkeypatch)

    authority = load_publication_consumer_ci_authority(tmp_path, authority_path)

    assert authority["consumer_head_sha"] == V2115_CONSUMER_HEAD
    assert authority["science_anchor"]["git_commit"] == V2115_SCIENCE_COMMIT
    assert authority["expected_ci"] == EXPECTED_CI
    assert authority["authority_status"] == "frozen"
    assert authority["validation_status"] == "pass"
    assert authority["ci_execution_status"] == "unverified"
    assert authority["scientific_evidence"] is False
    assert authority["science_dispatch_authority"] is False
    assert authority["provider_calls"] == 0


def test_publication_consumer_authority_rejects_tampering_without_reseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path = _publication_authority(tmp_path)
    value = json.loads(authority_path.read_text(encoding="utf-8"))
    value["expected_ci"]["test_count"] += 1
    authority_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _mock_publication_authority_dependencies(monkeypatch)

    with pytest.raises(CIReleaseReceiptError, match="self-hash mismatch"):
        load_publication_consumer_ci_authority(tmp_path, authority_path)


def test_publication_consumer_authority_rejects_self_rehashed_extra_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path = _publication_authority(tmp_path, extra_top_level=True)
    _mock_publication_authority_dependencies(monkeypatch)

    with pytest.raises(CIReleaseReceiptError, match="keys mismatch"):
        load_publication_consumer_ci_authority(tmp_path, authority_path)


def test_publication_consumer_authority_rejects_wrong_science_anchor_after_reseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path = _publication_authority(tmp_path, science_commit="8" * 40)
    _mock_publication_authority_dependencies(monkeypatch)

    with pytest.raises(CIReleaseReceiptError, match="science anchor drifted"):
        load_publication_consumer_ci_authority(tmp_path, authority_path)


def test_publication_consumer_authority_rejects_same_commit_retag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path = _publication_authority(tmp_path)
    # The peeled commit remains exact; only the annotated tag object changed.
    _mock_publication_authority_dependencies(monkeypatch, tag_object="9" * 40)

    with pytest.raises(CIReleaseReceiptError, match="tag object drifted"):
        load_publication_consumer_ci_authority(tmp_path, authority_path)


@pytest.mark.parametrize("field", sorted(EXPECTED_CI))
def test_publication_consumer_receipt_rejects_every_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    authority_path = _publication_authority(tmp_path)
    _mock_publication_authority_dependencies(monkeypatch)
    receipt = _publication_ci_job(tmp_path)
    receipt.pop("receipt_sha256")
    receipt[field] = (
        receipt[field] + 1
        if field in {"test_count", "compiled_source_count"}
        else "9" * 64
    )
    receipt["receipt_sha256"] = canonical_sha256(receipt)

    with pytest.raises(CIReleaseReceiptError, match=field):
        verify_publication_consumer_ci_receipt(
            tmp_path,
            authority_path,
            receipt,
        )


def test_publication_consumer_receipt_has_distinct_non_scientific_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path = _publication_authority(tmp_path)
    _mock_publication_authority_dependencies(monkeypatch)
    ci_job = _publication_ci_job(tmp_path)

    receipt = build_publication_consumer_ci_receipt(
        tmp_path,
        authority_path=authority_path,
        ci_job_receipt=ci_job,
    )

    assert receipt["schema_version"] == PUBLICATION_CONSUMER_CI_RECEIPT_SCHEMA_VERSION
    assert receipt["scientific_evidence"] is False
    assert receipt["science_dispatch_authority"] is False
    assert receipt["provider_calls"] == 0
    assert receipt["authority"]["ci_execution_status"] == "current-job-pass"
    assert receipt["authority"]["verified_job"]["runner_os"] == "Linux"
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def test_publication_consumer_rejects_incomplete_ci_job_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path = _publication_authority(tmp_path)
    _mock_publication_authority_dependencies(monkeypatch)

    with pytest.raises(CIReleaseReceiptError, match="job receipt keys mismatch"):
        build_publication_consumer_ci_receipt(
            tmp_path,
            authority_path=authority_path,
            ci_job_receipt={**EXPECTED_CI, "head_sha": V2115_CONSUMER_HEAD},
        )
    extra = {**_publication_ci_job(tmp_path), "unexpected": "field"}
    with pytest.raises(CIReleaseReceiptError, match="job receipt keys mismatch"):
        build_publication_consumer_ci_receipt(
            tmp_path,
            authority_path=authority_path,
            ci_job_receipt=extra,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    (
        ("schema_version", "finevo-ci-job-receipt-v0", "identity or status"),
        ("status", "failure", "identity or status"),
        ("receipt_sha256", "9" * 64, "self-hash mismatch"),
    ),
)
def test_publication_consumer_rejects_ci_job_identity_and_seal_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
    error: str,
) -> None:
    authority_path = _publication_authority(tmp_path)
    _mock_publication_authority_dependencies(monkeypatch)
    receipt = _publication_ci_job(tmp_path)
    receipt[field] = replacement
    if field != "receipt_sha256":
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = canonical_sha256(receipt)

    with pytest.raises(CIReleaseReceiptError, match=error):
        build_publication_consumer_ci_receipt(
            tmp_path,
            authority_path=authority_path,
            ci_job_receipt=receipt,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    (
        ("repository", "attacker/fork", "workflow identity drifted"),
        ("workflow_ref", "attacker/fork/workflow.yml@refs/heads/main", "workflow ref drifted"),
        ("workflow_blob_oid", "9" * 40, "workflow blob drifted"),
    ),
)
def test_publication_consumer_rejects_ci_job_origin_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
    error: str,
) -> None:
    authority_path = _publication_authority(tmp_path)
    _mock_publication_authority_dependencies(monkeypatch)
    receipt = _publication_ci_job(tmp_path)
    receipt[field] = replacement
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(receipt)

    with pytest.raises(CIReleaseReceiptError, match=error):
        build_publication_consumer_ci_receipt(
            tmp_path,
            authority_path=authority_path,
            ci_job_receipt=receipt,
        )


def test_publication_consumer_receipt_rejects_wrong_consumer_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path = _publication_authority(tmp_path)
    _mock_publication_authority_dependencies(monkeypatch)

    with pytest.raises(CIReleaseReceiptError, match="receipt HEAD differs"):
        verify_publication_consumer_ci_receipt(
            tmp_path,
            authority_path,
            _publication_ci_job(tmp_path, head="8" * 40),
        )


def test_verified_memory_ci_uses_descendant_consumer_authority_not_science_contract() -> None:
    workflow = (ROOT / ".github/workflows/verified-memory-ci.yml").read_text(
        encoding="utf-8"
    )
    emit = workflow.split("- name: Emit publication consumer CI receipt", 1)[1]
    assert "emit-publication-consumer" in emit
    assert (
        "--authority experiments/pilot_v2_11_5_publication_consumer_ci.json"
        in emit
    )
    assert "--contract experiments/pilot_v2_11_5.yaml" not in emit
    assert "- ubuntu-24.04" in workflow
    assert "- macos-14" in workflow


def test_tracked_v2113_frozen_contract_accepts_its_exact_ci_inventory() -> None:
    contract_path = ROOT / "experiments/pilot_v2_11_3.yaml"
    value = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = value["release_requirements"]["expected_ci"]

    verified = verify_contract_ci_receipt(contract_path, expected)

    assert verified == {
        "contract_id": "finevo-pilot-v2.11.3",
        "contract_sha256": value["integrity"]["declared_sha256"],
        "status": "pass",
    }
