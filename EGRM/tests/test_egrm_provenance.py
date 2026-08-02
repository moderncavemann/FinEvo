import json
from pathlib import Path

import pytest

from egrm.provenance import validate_extraction


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance" / "SOURCE_PROVENANCE.json"


def test_extracted_modules_match_declared_hashes():
    receipt = validate_extraction(MANIFEST, source_repo_root=ROOT.parent)
    assert receipt["checked_module_count"] == 11
    assert receipt["checked_inherited_test_count"] == 8
    assert receipt["checked_count"] == 19
    assert receipt["new_module_count"] == 8
    assert receipt["source_verified"] is True
    assert receipt["scientific_evidence_imported"] is False


def test_manifest_cannot_admit_historical_scientific_evidence(tmp_path):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["scientific_evidence_imported"] = True
    changed = tmp_path / "source.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbid imported"):
        validate_extraction(changed, egrm_root=ROOT, source_repo_root=ROOT.parent)


@pytest.mark.parametrize("field", ["source_commit", "git_blob", "source_sha256"])
def test_manifest_cannot_forge_source_identity(tmp_path, field):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if field == "source_commit":
        payload[field] = "0" * 40
    else:
        payload["modules"][0][field] = "0" * len(payload["modules"][0][field])
    changed = tmp_path / f"forged-{field}.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source|Git"):
        validate_extraction(
            changed,
            egrm_root=ROOT,
            source_repo_root=ROOT.parent,
        )


def test_manifest_requires_exact_new_module_inventory(tmp_path):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["new_egrm_modules"].remove("src/egrm/metrics.py")
    changed = tmp_path / "missing-new-module.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="package inventory is not exact"):
        validate_extraction(
            changed,
            egrm_root=ROOT,
            source_repo_root=ROOT.parent,
        )
