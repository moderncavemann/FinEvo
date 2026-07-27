from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory import observed_p95_authority as authority
from verified_memory import pilot_checkpoint
from verified_memory.pilot_contract import canonical_sha256


def _policy(case: Any) -> authority.HistoricalCheckpointVerificationPolicy:
    return authority.HistoricalCheckpointVerificationPolicy(
        source_repo_root=case.repo_root,
        source_annotated_tag=case.paid.git_tag,
        expected_tag_object="a" * 40,
        expected_peeled_commit=case.paid.head_commit,
    )


def _build(case: Any, *, historical: bool) -> dict[str, Any]:
    return authority.build_observed_p95_authority_receipt(
        repo_root=case.repo_root,
        contract_path="experiments/pilot_v2_3.yaml",
        raw_root=case.raw_root.relative_to(case.repo_root).as_posix(),
        model_id="gpt52_main",
        expected_git_commit=case.paid.head_commit,
        historical_checkpoint_policy=(
            _policy(case) if historical else None
        ),
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
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


def test_default_parent_authority_path_remains_strict(
    observed_p95_source_chain: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = observed_p95_source_chain
    original = pilot_checkpoint.verify_closed_loop_preflight_checkpoint
    calls: list[bool] = []

    def strict_replay(
        checkpoint: pilot_checkpoint.PilotCheckpoint,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        strict = kwargs.get("strict_code_binding", True)
        calls.append(strict)
        assert strict is True
        return original(checkpoint, *args, **kwargs)

    def forbidden_historical(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError(
            "default V2.4 parent path must not opt into historical replay"
        )

    monkeypatch.setattr(
        pilot_checkpoint,
        "verify_closed_loop_preflight_checkpoint",
        strict_replay,
    )
    monkeypatch.setattr(
        pilot_checkpoint,
        "verify_historical_closed_loop_preflight_checkpoint",
        forbidden_historical,
    )

    rebuilt = _build(case, historical=False)

    assert calls == [True]
    assert rebuilt == case.receipt


def test_explicit_historical_policy_is_the_only_compatibility_path(
    observed_p95_source_chain: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = observed_p95_source_chain
    historical_calls: list[dict[str, Any]] = []

    def forbidden_strict(*_args: Any, **_kwargs: Any) -> None:
        raise pilot_checkpoint.PilotCheckpointError(
            "checkpoint code binding differs from current code"
        )

    def historical_replay(
        checkpoint: pilot_checkpoint.PilotCheckpoint,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert checkpoint.checkpoint_hash == case.checkpoint.checkpoint_hash
        assert kwargs == {
            "source_repo_root": case.repo_root,
            "source_annotated_tag": case.paid.git_tag,
            "expected_tag_object": "a" * 40,
            "expected_peeled_commit": case.paid.head_commit,
            "frozen_exactness_receipt": case.exactness["exactness"],
        }
        historical_calls.append(deepcopy(kwargs))
        return {
            "schema_version": (
                "finevo-historical-checkpoint-verification-receipt-v1"
            ),
            "exactness": deepcopy(case.exactness["exactness"]),
        }

    monkeypatch.setattr(
        pilot_checkpoint,
        "verify_closed_loop_preflight_checkpoint",
        forbidden_strict,
    )
    monkeypatch.setattr(
        pilot_checkpoint,
        "verify_historical_closed_loop_preflight_checkpoint",
        historical_replay,
    )

    rebuilt = _build(case, historical=True)

    assert len(historical_calls) == 1
    # Verification context is not serialized into the parent receipt.
    assert rebuilt == case.receipt
    assert json.dumps(
        rebuilt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) == json.dumps(
        case.receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.parametrize("reseal_wrapper", [False, True])
def test_historical_policy_rejects_exactness_tampering(
    observed_p95_source_chain: Any,
    monkeypatch: pytest.MonkeyPatch,
    reseal_wrapper: bool,
) -> None:
    case = observed_p95_source_chain
    exactness_path = case.source_paths["checkpoint_exactness"]
    tampered = deepcopy(case.exactness)
    tampered["exactness"]["component_hashes"][
        "environment_hash"
    ] = "9" * 64
    if reseal_wrapper:
        tampered["integrity"]["content_sha256"] = (
            authority._bound_content_sha256(tampered)
        )
    _write_json(exactness_path, tampered)

    def historical_replay(
        _checkpoint: pilot_checkpoint.PilotCheckpoint,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert kwargs["frozen_exactness_receipt"] == tampered["exactness"]
        raise pilot_checkpoint.PilotCheckpointError(
            "current replay exactness differs from frozen historical exactness"
        )

    monkeypatch.setattr(
        pilot_checkpoint,
        "verify_historical_closed_loop_preflight_checkpoint",
        historical_replay,
    )
    try:
        if reseal_wrapper:
            message = "replay exactness differs"
        else:
            message = "checkpoint exactness receipt self-hash mismatch"
        with pytest.raises(authority.ObservedP95AuthorityError, match=message):
            _build(case, historical=True)
    finally:
        case.restore_sources()


def test_historical_policy_rejects_rehashed_receipt_tampering(
    observed_p95_source_chain: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = observed_p95_source_chain
    tampered = deepcopy(case.receipt)
    tampered["evidence_use"] = "tampered authority claim"
    unsigned = deepcopy(tampered)
    unsigned.pop("integrity")
    tampered["integrity"]["content_sha256"] = canonical_sha256(unsigned)

    def historical_replay(
        _checkpoint: pilot_checkpoint.PilotCheckpoint,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert kwargs["frozen_exactness_receipt"] == (
            case.exactness["exactness"]
        )
        return {
            "schema_version": (
                "finevo-historical-checkpoint-verification-receipt-v1"
            ),
            "exactness": deepcopy(case.exactness["exactness"]),
        }

    monkeypatch.setattr(
        pilot_checkpoint,
        "verify_historical_closed_loop_preflight_checkpoint",
        historical_replay,
    )
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="receipt differs from its reverified source chain",
    ):
        authority.verify_observed_p95_authority_receipt(
            tampered,
            repo_root=case.repo_root,
            expected_git_commit=case.paid.head_commit,
            historical_checkpoint_policy=_policy(case),
        )
