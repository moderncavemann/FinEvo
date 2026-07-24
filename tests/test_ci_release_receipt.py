import hashlib
from pathlib import Path

import pytest

from verified_memory.ci_release_receipt import (
    CIReleaseReceiptError,
    build_ci_job_receipt,
    build_collection_inventory,
    build_source_inventory,
    parse_junit_summary,
)
from verified_memory.scientific_release_attestation import (
    CI_JOB_RECEIPT_SCHEMA_VERSION,
    canonical_sha256,
)


HEAD = "1" * 40
WORKFLOW_BLOB = "2" * 40
INVENTORY_SHA = "3" * 64


def _environment() -> dict[str, str]:
    return {
        "GITHUB_REPOSITORY": "owner/finevo",
        "GITHUB_SHA": HEAD,
        "GITHUB_RUN_ID": "19821",
        "GITHUB_RUN_ATTEMPT": "3",
        "GITHUB_JOB": "verify",
        "GITHUB_WORKFLOW": "Verified memory CI",
        "GITHUB_WORKFLOW_REF": (
            "owner/finevo/.github/workflows/"
            "verified-memory-ci.yml@refs/heads/main"
        ),
        "GITHUB_WORKFLOW_SHA": HEAD,
        "RUNNER_OS": "Linux",
        "FINEVO_CI_JOB_NAME": "Python 3.12.7 / ubuntu-24.04",
    }


def _junit(tmp_path: Path, body: str) -> Path:
    output = tmp_path / "results.xml"
    output.write_text(body, encoding="utf-8")
    return output


def test_collection_and_source_inventory_are_order_sensitive_and_hashed():
    collection = build_collection_inventory(
        ("tests/test_a.py::test_one", "tests/test_b.py::test_two")
    )
    assert collection["test_count"] == 2
    assert collection["test_collection_sha256"] == canonical_sha256(
        ["tests/test_a.py::test_one", "tests/test_b.py::test_two"]
    )
    sources = build_source_inventory(
        ("run_pilot.py", "verified_memory/runner.py")
    )
    assert sources["compiled_source_count"] == 2
    assert sources["compiled_source_inventory_sha256"] == canonical_sha256(
        ["run_pilot.py", "verified_memory/runner.py"]
    )
    with pytest.raises(CIReleaseReceiptError, match="duplicate"):
        build_collection_inventory(("same::test", "same::test"))
    with pytest.raises(CIReleaseReceiptError, match="sorted"):
        build_source_inventory(("z.py", "a.py"))


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
    workflow = (
        tmp_path / ".github" / "workflows" / "verified-memory-ci.yml"
    )
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: Verified memory CI\n", encoding="utf-8")
    monkeypatch.setattr(
        "verified_memory.ci_release_receipt._git_line",
        lambda root, argv: (
            HEAD if argv[-1] == "HEAD^{commit}" else WORKFLOW_BLOB
        ),
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
    assert receipt["workflow_file_sha256"] == hashlib.sha256(
        workflow.read_bytes()
    ).hexdigest()
    assert receipt["test_count"] == 2
    assert receipt["compiled_source_count"] == 2
    assert receipt["sealed_manifest_inventory_sha256"] == INVENTORY_SHA
    unsigned = dict(receipt)
    observed = unsigned.pop("receipt_sha256")
    assert observed == canonical_sha256(unsigned)


def test_ci_receipt_fails_closed_on_count_or_workflow_sha_drift(
    tmp_path: Path, monkeypatch
):
    workflow = (
        tmp_path / ".github" / "workflows" / "verified-memory-ci.yml"
    )
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: Verified memory CI\n", encoding="utf-8")
    monkeypatch.setattr(
        "verified_memory.ci_release_receipt._git_line",
        lambda root, argv: (
            HEAD if argv[-1] == "HEAD^{commit}" else WORKFLOW_BLOB
        ),
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
