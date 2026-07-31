from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from tests import test_pilot_v211_gate as gate_fixtures
from verified_memory import observed_p95_authority as authority
from verified_memory import pilot_v211_gate as gate


def _write_receipt(
    repo_root: Path,
    relative: str,
    receipt: dict[str, Any],
) -> Path:
    target = repo_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _valid_receipt() -> dict[str, Any]:
    return gate.build_v211_post_gate_authority(**gate_fixtures._inputs())


def _reseal(receipt: dict[str, Any]) -> dict[str, Any]:
    unsigned = deepcopy(receipt)
    unsigned.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = gate.canonical_sha256(unsigned)
    return receipt


def test_v211_schema_is_registered_without_replacing_historical_adapters() -> None:
    registry = authority.DEDICATED_OBSERVED_P95_BINDING_SCHEMA_REGISTRY

    assert registry[gate.V211_GATE_SCHEMA_VERSION] == (
        "v2.11-post-gate-authority"
    )
    assert registry[
        "finevo-pilot-v2.10.1-resealed-observed-p95-authority-v1"
    ] == "v2.10.1-resealed-with-sibling-projection"
    assert registry[
        "finevo-pilot-v2.10.2-resealed-observed-p95-authority-v1"
    ] == "v2.10.2-resealed-with-sibling-projection"


def test_v211_gate_reaches_generic_flat_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = (
        "experiment_results/pilot-v2.11/raw/post-gate/gate.json"
    )
    _write_receipt(tmp_path, relative, _valid_receipt())
    dedicated_calls: list[str] = []
    original = gate.verified_v211_gate_authority_binding

    def recorded(
        receipt_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        dedicated_calls.append(receipt_path)
        return original(receipt_path, **kwargs)

    monkeypatch.setattr(
        gate,
        "verified_v211_gate_authority_binding",
        recorded,
    )
    generic = authority.verified_observed_p95_authority_binding(
        relative,
        repo_root=tmp_path,
        expected_git_commit=gate_fixtures.RELEASE_COMMIT,
    )
    verified_rows = authority.verify_observed_p95_authority_receipt(
        relative,
        repo_root=tmp_path,
        expected_git_commit=gate_fixtures.RELEASE_COMMIT,
    )

    assert dedicated_calls == [relative, relative]
    assert set(generic) == {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
        "git_commit",
        "reservations",
    }
    assert generic["receipt_path"] == relative
    assert generic["git_commit"] == gate_fixtures.RELEASE_COMMIT
    assert verified_rows == generic["reservations"]
    assert set(generic["reservations"]) == {
        "openai/gpt-5.2-2025-12-11",
        "openai/gpt-5.6-sol",
    }
    for by_kind in generic["reservations"].values():
        assert set(by_kind) == {"action", "semantic"}
        for entry in by_kind.values():
            assert set(entry) == {"authority", "reservation"}


def test_v211_generic_adapter_rejects_unknown_and_tampered_receipts(
    tmp_path: Path,
) -> None:
    relative = (
        "experiment_results/pilot-v2.11/raw/post-gate/gate.json"
    )
    unknown = _valid_receipt()
    unknown["schema_version"] = (
        "finevo-pilot-v2.11-unknown-post-gate-authority-v1"
    )
    _write_receipt(tmp_path, relative, _reseal(unknown))

    with pytest.raises(authority.ObservedP95AuthorityError):
        authority.verified_observed_p95_authority_binding(
            relative,
            repo_root=tmp_path,
            expected_git_commit=gate_fixtures.RELEASE_COMMIT,
        )

    tampered = _valid_receipt()
    tampered["projection"]["new_hosted_completions"] -= 1
    _write_receipt(tmp_path, relative, tampered)
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="self-hash mismatch",
    ):
        authority.verified_observed_p95_authority_binding(
            relative,
            repo_root=tmp_path,
            expected_git_commit=gate_fixtures.RELEASE_COMMIT,
        )


def test_v211_generic_adapter_rejects_sibling_binding_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = (
        "experiment_results/pilot-v2.11/raw/post-gate/gate.json"
    )
    sibling = (
        "experiment_results/pilot-v2.11/raw/post-gate/sibling.json"
    )
    _write_receipt(tmp_path, selected, _valid_receipt())
    sibling_inputs = gate_fixtures._inputs()
    sibling_inputs["ledger_event_chain_head"] = "9" * 64
    _write_receipt(
        tmp_path,
        sibling,
        gate.build_v211_post_gate_authority(**sibling_inputs),
    )
    original = gate.verified_v211_gate_authority_binding

    def substitute_sibling(
        _receipt_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return original(sibling, **kwargs)

    monkeypatch.setattr(
        gate,
        "verified_v211_gate_authority_binding",
        substitute_sibling,
    )
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="changed during dedicated verification",
    ):
        authority.verified_observed_p95_authority_binding(
            selected,
            repo_root=tmp_path,
            expected_git_commit=gate_fixtures.RELEASE_COMMIT,
        )


def test_v211_generic_adapter_rejects_symlinked_receipt(
    tmp_path: Path,
) -> None:
    target_relative = (
        "experiment_results/pilot-v2.11/raw/post-gate/real.json"
    )
    target = _write_receipt(tmp_path, target_relative, _valid_receipt())
    alias_relative = (
        "experiment_results/pilot-v2.11/raw/post-gate/gate.json"
    )
    alias = tmp_path / alias_relative
    alias.symlink_to(target.name)

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="cannot be opened safely",
    ):
        authority.verified_observed_p95_authority_binding(
            alias_relative,
            repo_root=tmp_path,
            expected_git_commit=gate_fixtures.RELEASE_COMMIT,
        )
