from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess

import pytest

from diagnose_provider_interface import (
    _safe_error_type as _safe_cli_error_type,
    build_parser,
)
from llm_providers import ProviderErrorDetails, StructuredCompletion
from verified_memory.budget import UsageRecord
from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory.provider_diagnostics import (
    DEFAULT_INTERFACE_PROBE_MAX_TOKENS,
    DIAGNOSTIC_CUMULATIVE_CAP_USD,
    DIAGNOSTIC_OUTPUT_RELATIVE_ROOT,
    PRIOR_MANUAL_DIAGNOSTIC_RESERVE_USD,
    ProviderDiagnosticError,
    _assert_diagnostic_output_path,
    _cumulative_reservation_cost,
    _estimated_usage,
    _ledger_path,
    _reserve_cumulative_diagnostic_budget,
    run_provider_interface_probe,
    verify_debug_provenance,
    verify_provider_interface_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v1.yaml"


@pytest.fixture
def diagnostic_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(
        ["git", "init", "-q"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    (root / ".gitignore").write_text(
        "experiment_results/\n.env\nlogs/\n",
        encoding="utf-8",
    )
    return root


def _output(root: Path, name: str) -> Path:
    return root / DIAGNOSTIC_OUTPUT_RELATIVE_ROOT / name


class _FixtureProvider:
    def __init__(
        self,
        profile,
        *,
        usage: UsageRecord | None = None,
    ) -> None:
        self.profile = profile
        self.calls = 0
        self.usage = usage or UsageRecord(10, 4, 0.001)

    def get_model_name(self) -> str:
        transport = (
            "thirdparty"
            if self.profile.transport == "openrouter"
            else self.profile.transport
        )
        return f"{transport}/{self.profile.requested_model}"

    def get_completion(self, *args, **kwargs):
        raise AssertionError("tuple API is not expected")

    def get_structured_completion(self, messages, **kwargs):
        self.calls += 1
        assert len(messages) == 1
        assert kwargs["max_retries"] == 1
        expected_seed = (
            None
            if self.profile.seed_capability == "unsupported"
            else 2010922376
        )
        assert kwargs["seed"] == expected_seed
        common = {
            "text": '{"ok": true}',
            "usage": self.usage,
            "model": self.profile.requested_model,
            "attempts": 1,
            "latency_seconds": 0.01,
            "request_seed": kwargs["seed"],
            "response_model": self.profile.served_model,
            "request_profile_id": self.profile.profile_id,
            "request_provider_pin": tuple(self.profile.provider_pin),
            "request_artifact_identity": tuple(
                self.profile.artifact_identity
            ),
            "request_price_snapshot_source": (
                self.profile.price_snapshot.source
            ),
            "request_price_snapshot_captured_at": (
                self.profile.price_snapshot.captured_at
            ),
            "finish_reason": "stop",
            "native_finish_reason": "stop",
            "response_completed": True,
        }
        if self.profile.transport == "openai":
            return StructuredCompletion(
                provider="openai",
                response_provider="OpenAI-direct",
                response_route="direct",
                provider_sdk_name="openai-python",
                provider_sdk_version="2.46.0",
                request_parameters=(
                    "max_completion_tokens",
                    "messages",
                    "model",
                    "reasoning_effort",
                    "response_format",
                    "seed",
                    "top_p",
                ),
                temperature_dispatch="omitted_unsupported",
                **common,
            )
        if self.profile.transport == "ollama":
            return StructuredCompletion(
                provider="ollama",
                response_provider="local-ollama",
                response_route="local",
                provider_sdk_name="requests",
                provider_sdk_version="2.34.2",
                request_parameters=(
                    "format",
                    "messages",
                    "model",
                    "options",
                    "stream",
                ),
                temperature_dispatch="explicit",
                **common,
            )
        if self.profile.transport == "openrouter":
            return StructuredCompletion(
                provider="thirdparty",
                response_provider=self.profile.provider_pin[0],
                response_route=dict(self.profile.artifact_identity)[
                    "served_snapshot"
                ],
                provider_sdk_name="openai-python",
                provider_sdk_version="2.46.0",
                route_attestation_code="OR_RA_PASS",
                route_attestation_source="inline-attribute",
                request_parameters=(
                    "extra_body",
                    "max_tokens",
                    "messages",
                    "model",
                    "response_format",
                    "seed",
                    "temperature",
                    "top_p",
                ),
                temperature_dispatch="explicit",
                **common,
            )
        raise AssertionError("fixture received an unsupported transport")


def _provenance(_root, *, required_tag):
    return {
        "git_tag": required_tag,
        "head_commit": "a" * 40,
        "tag_commit": "a" * 40,
        "tag_object_id": "b" * 40,
        "tag_object_type": "tag",
        "worktree_clean": True,
    }


def _run(
    root: Path,
    *,
    name: str,
    model_id: str = "gpt52_main",
    provider_factory=None,
    max_tokens: int | None = 32,
    max_cost_usd: float = 0.05,
    force_json_object: bool = False,
):
    providers = []

    def default_factory(profile):
        provider = _FixtureProvider(profile)
        providers.append(provider)
        return provider

    kwargs = {
        "contract_path": CONTRACT_PATH,
        "model_id": model_id,
        "output_path": _output(root, name),
        "repo_root": root,
        "required_tag": f"pilot-v2-debug-{name}",
        "max_cost_usd": max_cost_usd,
        "force_json_object": force_json_object,
        "provider_factory": provider_factory or default_factory,
        "provenance_verifier": _provenance,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    result = run_provider_interface_probe(
        **kwargs,
    )
    return result, providers


def _read_ledger(root: Path) -> dict:
    return json.loads(_ledger_path(root).read_text(encoding="utf-8"))


def _assert_ledger_hash(ledger: dict) -> None:
    copied = dict(ledger)
    expected = copied.pop("ledger_sha256")
    assert expected == canonical_sha256(copied)


def test_single_probe_writes_one_safe_receipt_and_final_ledger_hash(
    diagnostic_root: Path,
) -> None:
    result, providers = _run(diagnostic_root, name="gpt52.json")
    output = _output(diagnostic_root, "gpt52.json")

    assert result["status"] == "pass"
    assert result["scientific_evidence"] is False
    assert result["diagnostic_only"] is True
    assert result["denominator_inclusion"] is False
    assert result["dispatch"] == {
        "provider_constructed": True,
        "provider_call_attempted": True,
    }
    assert result["checks"] and all(result["checks"].values())
    assert providers[0].calls == 1
    assert result["budget"]["completed_calls"] == 1
    assert "text" not in result["completion"]
    assert result["completion"]["output_bytes"] == len(b'{"ok": true}')
    assert "cumulative_budget_final" not in result

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == result
    assert '{"ok": true}' not in output.read_text(encoding="utf-8")
    verify_provider_interface_receipt(persisted)

    ledger = _read_ledger(diagnostic_root)
    _assert_ledger_hash(ledger)
    assert ledger["scientific_evidence"] is False
    assert ledger["diagnostic_only"] is True
    assert ledger["entries"][0]["status"] == "pass"
    assert ledger["entries"][0]["receipt_sha256"] == result["receipt_sha256"]


def test_reasoning_safe_default_avoids_an_80_token_interface_false_negative(
    diagnostic_root: Path,
) -> None:
    parsed = build_parser().parse_args(
        [
            "--model-id",
            "gemini35_flash_sentinel",
            "--required-tag",
            "pilot-v2-debug-default",
            "--output",
            str(_output(diagnostic_root, "default.json")),
        ]
    )
    signature_default = inspect.signature(
        run_provider_interface_probe
    ).parameters["max_tokens"].default
    assert parsed.max_tokens == DEFAULT_INTERFACE_PROBE_MAX_TOKENS == 128
    assert signature_default == DEFAULT_INTERFACE_PROBE_MAX_TOKENS

    contract = load_pilot_contract(CONTRACT_PATH)
    profile = contract.provider_profiles["gemini35_flash_sentinel"]
    assert (
        _estimated_usage(
            profile,
            max_tokens=DEFAULT_INTERFACE_PROBE_MAX_TOKENS,
        ).cost_usd
        < 0.05
    )

    class ReasoningHeavyProvider(_FixtureProvider):
        def get_structured_completion(self, messages, **kwargs):
            observed_max_tokens.append(kwargs["max_tokens"])
            result = super().get_structured_completion(messages, **kwargs)
            if kwargs["max_tokens"] <= 80:
                return replace(
                    result,
                    text='{"ok',
                    usage=UsageRecord(15, 80, 0.001),
                    error_type="IncompleteCompletionError",
                    provider_error_details=ProviderErrorDetails(
                        error_type="IncompleteCompletionError",
                        stage="openrouter.response.finish",
                        sdk_name="openai-python",
                        sdk_version="2.46.0",
                        code="completion_length",
                    ),
                    reasoning_tokens=74,
                    finish_reason="length",
                    native_finish_reason="MAX_TOKENS",
                    response_completed=False,
                    output_disposition="discarded_incomplete",
                )
            return replace(
                result,
                usage=UsageRecord(15, 86, 0.0011),
                reasoning_tokens=74,
            )

    created = []
    observed_max_tokens = []

    def reasoning_factory(profile):
        provider = ReasoningHeavyProvider(profile)
        created.append(provider)
        return provider

    truncated, _ = _run(
        diagnostic_root,
        name="gemini-80.json",
        model_id="gemini35_flash_sentinel",
        provider_factory=reasoning_factory,
        max_tokens=80,
    )
    assert truncated["status"] == "no-go"
    assert truncated["checks"]["finish_stop"] is False
    assert truncated["checks"]["strict_json_valid"] is False
    assert truncated["completion"]["reasoning_tokens"] == 74
    assert created[-1].calls == 1

    passed, _ = _run(
        diagnostic_root,
        name="gemini-default.json",
        model_id="gemini35_flash_sentinel",
        provider_factory=reasoning_factory,
        max_tokens=None,
    )
    assert passed["status"] == "pass"
    assert passed["completion"]["reasoning_tokens"] == 74
    assert created[-1].calls == 1
    assert observed_max_tokens == [80, DEFAULT_INTERFACE_PROBE_MAX_TOKENS]
    assert (
        passed["request"]["max_tokens"]
        == DEFAULT_INTERFACE_PROBE_MAX_TOKENS
    )
    assert (
        passed["budget"]["limits"]["max_completion_tokens"]
        == DEFAULT_INTERFACE_PROBE_MAX_TOKENS
    )

    for receipt in (truncated, passed):
        assert receipt["scientific_evidence"] is False
        assert receipt["diagnostic_only"] is True
        assert receipt["denominator_inclusion"] is False
        verify_provider_interface_receipt(receipt)

    ledger = _read_ledger(diagnostic_root)
    _assert_ledger_hash(ledger)
    assert [entry["status"] for entry in ledger["entries"]] == [
        "no-go",
        "pass",
    ]
    assert all(
        entry["reserved_cost_usd"] == 0.05
        for entry in ledger["entries"]
    )
    assert ledger["hard_cap_usd"] == DIAGNOSTIC_CUMULATIVE_CAP_USD == 0.30


def test_exact_stop_and_strict_json_are_independent_interface_gates(
    diagnostic_root: Path,
) -> None:
    class FinishOnlyFaultProvider(_FixtureProvider):
        def get_structured_completion(self, messages, **kwargs):
            return replace(
                super().get_structured_completion(messages, **kwargs),
                finish_reason="length",
                native_finish_reason="MAX_TOKENS",
            )

    class JsonOnlyFaultProvider(_FixtureProvider):
        def get_structured_completion(self, messages, **kwargs):
            return replace(
                super().get_structured_completion(messages, **kwargs),
                text="not-json",
            )

    created = []

    def factory(provider_type):
        def build(profile):
            provider = provider_type(profile)
            created.append(provider)
            return provider

        return build

    finish_fault, _ = _run(
        diagnostic_root,
        name="finish-only.json",
        provider_factory=factory(FinishOnlyFaultProvider),
    )
    assert finish_fault["status"] == "no-go"
    assert {
        name for name, passed in finish_fault["checks"].items() if not passed
    } == {"finish_stop"}
    assert created[-1].calls == 1

    json_fault, _ = _run(
        diagnostic_root,
        name="json-only.json",
        provider_factory=factory(JsonOnlyFaultProvider),
    )
    assert json_fault["status"] == "no-go"
    assert {
        name for name, passed in json_fault["checks"].items() if not passed
    } == {"strict_json_valid"}
    assert created[-1].calls == 1

    for receipt in (finish_fault, json_fault):
        verify_provider_interface_receipt(receipt)


def test_probe_rejects_incomplete_dispatched_parameter_shape(
    diagnostic_root: Path,
) -> None:
    class MissingJsonModeProvider(_FixtureProvider):
        def get_structured_completion(self, messages, **kwargs):
            result = super().get_structured_completion(messages, **kwargs)
            return replace(
                result,
                request_parameters=tuple(
                    name
                    for name in result.request_parameters
                    if name != "response_format"
                ),
            )

    result, _ = _run(
        diagnostic_root,
        name="missing-json-mode.json",
        provider_factory=MissingJsonModeProvider,
    )

    assert result["status"] == "no-go"
    assert result["checks"]["request_parameters_exact"] is False
    assert result["checks"]["strict_json_valid"] is True
    assert result["dispatch"]["provider_call_attempted"] is True


def test_output_is_confined_ignored_json_non_overwriting_and_internal_safe(
    diagnostic_root: Path,
) -> None:
    allowed = _output(diagnostic_root, "valid.json")
    _assert_diagnostic_output_path(diagnostic_root, allowed)

    for forbidden in (
        diagnostic_root / ".env",
        diagnostic_root / "logs" / "probe.json",
        diagnostic_root / "experiment_results" / "pilot-v1" / "probe.json",
    ):
        with pytest.raises(ProviderDiagnosticError, match="must be inside"):
            _assert_diagnostic_output_path(diagnostic_root, forbidden)

    with pytest.raises(ProviderDiagnosticError, match="internal ledger"):
        _assert_diagnostic_output_path(
            diagnostic_root,
            _ledger_path(diagnostic_root),
        )
    with pytest.raises(ProviderDiagnosticError, match="json suffix"):
        _assert_diagnostic_output_path(
            diagnostic_root,
            _output(diagnostic_root, "probe.txt"),
        )

    first, providers = _run(diagnostic_root, name="once.json")
    assert first["status"] == "pass"
    with pytest.raises(ProviderDiagnosticError, match="cannot be overwritten"):
        _run(
            diagnostic_root,
            name="once.json",
            provider_factory=lambda profile: pytest.fail(
                "provider must not be constructed for an existing receipt"
            ),
        )
    assert providers[0].calls == 1


def test_provider_construction_failure_writes_redacted_pre_dispatch_no_go(
    diagnostic_root: Path,
) -> None:
    secret = "SECRET_PROVIDER_CONSTRUCTOR_sk-private"

    def fail(_profile):
        raise RuntimeError(secret)

    result, _ = _run(
        diagnostic_root,
        name="constructor-no-go.json",
        provider_factory=fail,
    )
    serialized = json.dumps(result, sort_keys=True)

    assert result["status"] == "no-go"
    assert result["dispatch"] == {
        "provider_constructed": False,
        "provider_call_attempted": False,
    }
    assert result["completion"] is None
    assert result["budget"]["completed_calls"] == 0
    assert result["pre_dispatch_failure"]["error_type"] == "RuntimeError"
    assert (
        result["pre_dispatch_failure"]["stage"]
        == "diagnostic.provider.construct"
    )
    assert secret not in serialized
    verify_provider_interface_receipt(result)

    ledger = _read_ledger(diagnostic_root)
    _assert_ledger_hash(ledger)
    entry = ledger["entries"][0]
    assert entry["status"] == "pre-dispatch-no-go"
    assert entry["accounted_cost_usd"] == 0.0
    assert entry["receipt_sha256"] == result["receipt_sha256"]


def test_hostile_exception_type_is_not_persisted(
    diagnostic_root: Path,
) -> None:
    hostile_name = "SensitiveTypeNotAllowlisted"
    hostile_error = type(hostile_name, (RuntimeError,), {})

    def fail(_profile):
        raise hostile_error("sensitive constructor body")

    result, _ = _run(
        diagnostic_root,
        name="hostile-type-no-go.json",
        provider_factory=fail,
    )
    serialized = json.dumps(result, sort_keys=True)

    assert result["status"] == "no-go"
    assert result["pre_dispatch_failure"]["error_type"] == "ProviderError"
    assert hostile_name not in serialized
    assert "sensitive constructor body" not in serialized
    assert _safe_cli_error_type(hostile_error("body")) == "ProviderError"


def test_estimate_above_caller_cap_is_a_durable_zero_call_no_go(
    diagnostic_root: Path,
) -> None:
    result, _ = _run(
        diagnostic_root,
        name="estimate-no-go.json",
        max_cost_usd=0.0001,
        provider_factory=lambda profile: pytest.fail(
            "provider must not be constructed above the caller cost cap"
        ),
    )

    assert result["status"] == "no-go"
    assert result["dispatch"]["provider_call_attempted"] is False
    assert result["budget"]["completed_calls"] == 0
    assert (
        result["pre_dispatch_failure"]["stage"]
        == "diagnostic.budget.estimate"
    )
    ledger = _read_ledger(diagnostic_root)
    assert ledger["entries"][0]["status"] == "pre-dispatch-no-go"
    assert ledger["entries"][0]["accounted_cost_usd"] == 0.0


def test_budget_overage_still_writes_one_call_no_go_receipt(
    diagnostic_root: Path,
) -> None:
    providers = []

    def factory(profile):
        provider = _FixtureProvider(
            profile,
            usage=UsageRecord(10, 33, 0.06),
        )
        providers.append(provider)
        return provider

    result, _ = _run(
        diagnostic_root,
        name="overage.json",
        provider_factory=factory,
        max_tokens=32,
        max_cost_usd=0.05,
    )

    assert providers[0].calls == 1
    assert result["status"] == "no-go"
    assert result["budget"]["completed_calls"] == 1
    assert set(result["budget_overage_reasons"]) == {
        "completion_token_limit",
        "cost_limit",
    }
    ledger = _read_ledger(diagnostic_root)
    assert ledger["entries"][0]["status"] == "no-go"
    assert ledger["entries"][0]["receipt_sha256"] == result["receipt_sha256"]


def test_process_interrupt_leaves_durable_dispatching_reservation(
    diagnostic_root: Path,
) -> None:
    class InterruptingProvider(_FixtureProvider):
        def get_structured_completion(self, messages, **kwargs):
            self.calls += 1
            raise KeyboardInterrupt()

    output = _output(diagnostic_root, "interrupted.json")
    with pytest.raises(KeyboardInterrupt):
        _run(
            diagnostic_root,
            name=output.name,
            provider_factory=InterruptingProvider,
        )

    assert not output.exists()
    ledger = _read_ledger(diagnostic_root)
    _assert_ledger_hash(ledger)
    entry = ledger["entries"][0]
    assert entry["status"] == "dispatching"
    assert entry["dispatch_started_at"]
    assert entry["accounted_cost_usd"] is None
    assert entry["receipt_sha256"] is None


def test_local_prompt_only_is_durable_no_go_and_override_reserves_zero(
    diagnostic_root: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    source = contract.provider_profiles["llama33_local_sentinel"]
    constructed = []

    result, _ = _run(
        diagnostic_root,
        name="local-no-override.json",
        model_id="llama33_local_sentinel",
        provider_factory=lambda profile: constructed.append(profile),
    )
    assert result["status"] == "no-go"
    assert result["dispatch"]["provider_call_attempted"] is False
    assert (
        result["pre_dispatch_failure"]["stage"]
        == "diagnostic.profile.validate"
    )
    assert constructed == []
    assert source.json_mode == "prompt_only"
    ledger = _read_ledger(diagnostic_root)
    assert ledger["entries"][0]["reserved_cost_usd"] == 0.0

    second_root = diagnostic_root.parent / "local-override-repo"
    second_root.mkdir()
    subprocess.run(
        ["git", "init", "-q"],
        cwd=second_root,
        check=True,
        capture_output=True,
        text=True,
    )
    (second_root / ".gitignore").write_text(
        "experiment_results/\n",
        encoding="utf-8",
    )
    passed, providers = _run(
        second_root,
        name="local-override.json",
        model_id="llama33_local_sentinel",
        force_json_object=True,
    )
    assert passed["status"] == "pass"
    assert providers[0].calls == 1
    assert passed["diagnostic_overrides"]["json_mode"]["source"] == "prompt_only"
    local_ledger = _read_ledger(second_root)
    assert local_ledger["entries"][0]["reserved_cost_usd"] == 0.0


def test_cumulative_cap_fits_five_default_hosted_cells_and_local_is_free(
    diagnostic_root: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    hosted = contract.provider_profiles["gpt52_main"]
    local = contract.provider_profiles["llama33_local_sentinel"]

    assert _cumulative_reservation_cost(
        hosted,
        max_cost_usd=0.05,
    ) == 0.05
    assert _cumulative_reservation_cost(
        local,
        max_cost_usd=0.05,
    ) == 0.0
    assert PRIOR_MANUAL_DIAGNOSTIC_RESERVE_USD + 5 * 0.05 <= (
        DIAGNOSTIC_CUMULATIVE_CAP_USD
    )

    for index in range(5):
        _reserve_cumulative_diagnostic_budget(
            diagnostic_root,
            required_tag=f"tag-{index}",
            model_id=f"hosted-{index}",
            profile_id=f"profile-{index}",
            output_path=_output(diagnostic_root, f"hosted-{index}.json"),
            reserved_cost_usd=0.05,
            force_json_object=False,
        )
    with pytest.raises(ProviderDiagnosticError, match="budget is exhausted"):
        _reserve_cumulative_diagnostic_budget(
            diagnostic_root,
            required_tag="tag-six",
            model_id="hosted-six",
            profile_id="profile-six",
            output_path=_output(diagnostic_root, "hosted-six.json"),
            reserved_cost_usd=0.05,
            force_json_object=False,
        )


def test_receipt_verifier_detects_hash_and_boundary_tampering(
    diagnostic_root: Path,
) -> None:
    result, _ = _run(diagnostic_root, name="tamper.json")

    tampered = json.loads(json.dumps(result))
    tampered["status"] = "forged"
    with pytest.raises(ProviderDiagnosticError, match="hash mismatch"):
        verify_provider_interface_receipt(tampered)

    boundary = json.loads(json.dumps(result))
    boundary["scientific_evidence"] = True
    boundary.pop("receipt_sha256")
    boundary["receipt_sha256"] = canonical_sha256(boundary)
    with pytest.raises(ProviderDiagnosticError, match="scientific boundary"):
        verify_provider_interface_receipt(boundary)

    endpoint = json.loads(json.dumps(result))
    endpoint["request"]["endpoint_identity"] = "untrusted-endpoint"
    endpoint.pop("receipt_sha256")
    endpoint["receipt_sha256"] = canonical_sha256(endpoint)
    with pytest.raises(ProviderDiagnosticError, match="endpoint identity"):
        verify_provider_interface_receipt(endpoint)


def test_clean_annotated_tag_provenance_records_tag_object(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tagged"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=FinEvo Test",
            "-c",
            "user.email=finevo@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=FinEvo Test",
            "-c",
            "user.email=finevo@example.invalid",
            "tag",
            "-am",
            "diagnostic",
            "pilot-v2-debug-test",
        ],
        cwd=root,
        check=True,
    )

    provenance = verify_debug_provenance(
        root,
        required_tag="pilot-v2-debug-test",
    )
    assert provenance["head_commit"] == provenance["tag_commit"]
    assert provenance["tag_object_type"] == "tag"
    assert provenance["tag_object_id"] != provenance["tag_commit"]
    assert provenance["worktree_clean"] is True

    (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ProviderDiagnosticError, match="clean worktree"):
        verify_debug_provenance(
            root,
            required_tag="pilot-v2-debug-test",
        )
